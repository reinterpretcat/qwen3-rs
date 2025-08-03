use crate::tensor_reader::TensorReader;
use anyhow::Result;
use log::{debug, warn};
use rayon::prelude::*;

/// LoraMerger applies standard LoRA merge logic on tensors: W = W_base + α / r * (B @ A)
///
/// Assumptions:
/// - LoRA uses a low-rank update to fine-tune a frozen base model weight.
/// - A and B are learned matrices with shapes:
///     - A: (r, in_features)
///     - B: (out_features, r)
/// - The base weight matrix W_base has shape (out_features, in_features) and is stored in row-major 1D layout.
/// - The rank `r` is a small integer (e.g., 4, 8, 16) << min(in_features, out_features).
/// - `alpha` is a scalar hyperparameter; `scaling = alpha / r`.
///
/// Merge formula:
///   W = W_base + scaling * (B @ A)
/// Where:
/// - `B @ A` produces a matrix of shape (out_features, in_features)
/// - `scaling` modulates the update magnitude
/// - The update is applied elementwise to the flattened base tensor
///
/// Notes:
/// - This code assumes tensors are flattened 1D f32 buffers in row-major order.
/// - The caller is responsible for ensuring tensor shapes are consistent.
pub(crate) struct LoraMerger<'a> {
    tensor_reader: &'a TensorReader,
    scaling: f32,
    rank: usize,
}

impl<'a> LoraMerger<'a> {
    pub fn new(tensor_reader: &'a TensorReader, alpha: f32, rank: usize) -> Result<Self> {
        let scaling = alpha / rank as f32;

        if !scaling.is_finite() || scaling.is_nan() {
            anyhow::bail!("Invalid scaling factor: {scaling} (must be finite). Alpha: {alpha}, Rank: {rank}");
        }

        Ok(Self { tensor_reader, scaling, rank })
    }

    /// Try to merge LoRA adapters with base weights
    pub fn try_merge_lora_adapters(
        &self,
        base_weights: &[f32],
        component: &str,
        layer_idx: u32,
    ) -> Result<Option<Vec<f32>>> {
        // component is already clean (e.g., "self_attn.k_proj", "mlp.gate_proj")
        let (lora_a, lora_b) = self.discover_and_load_lora_pairs(component, layer_idx)?;

        if let (Some(a), Some(b)) = (lora_a, lora_b) {
            debug!("Merging LoRA adapters for {component} layer {layer_idx} with scaling {}", self.scaling);

            // Merge LoRA: W = W_base + scaling * (B @ A)
            let merged = self.merge_lora_weights(base_weights, &a, &b)?;
            Ok(Some(merged))
        } else {
            Ok(None)
        }
    }

    /// Dynamically discover and load LoRA adapter pairs
    fn discover_and_load_lora_pairs(
        &self,
        component: &str,
        layer_idx: u32,
    ) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>)> {
        // TODO: consider supporting different patterns?

        // Based on the actual tensor naming pattern:
        // Base: model.layers.{layer}.{component}.weight

        let lora_a_name = format!("base_model.model.model.layers.{layer_idx}.{component}.lora_A.weight");
        let lora_b_name = format!("base_model.model.model.layers.{layer_idx}.{component}.lora_B.weight");

        debug!("Looking for LoRA A: '{lora_a_name}'");
        let lora_a = self.tensor_reader.load_tensor(&lora_a_name)?;

        debug!("Looking for LoRA B: '{lora_b_name}'");
        let lora_b = self.tensor_reader.load_tensor(&lora_b_name)?;

        if lora_a.is_none() || lora_b.is_none() {
            debug!(
                "Could not find LoRA pair for layers.{layer_idx}.{component}, A found: {}, B found: {}",
                lora_a.is_some(),
                lora_b.is_some()
            );
        }

        Ok((lora_a, lora_b))
    }

    /// Merge LoRA weights: W = W_base + scaling * (B @ A)
    fn merge_lora_weights(&self, base: &[f32], lora_a: &[f32], lora_b: &[f32]) -> Result<Vec<f32>> {
        // Input validation
        if base.is_empty() || lora_a.is_empty() || lora_b.is_empty() {
            return Err(anyhow::anyhow!(
                "Empty tensors not allowed: base={}, A={}, B={}",
                base.len(),
                lora_a.len(),
                lora_b.len()
            ));
        }

        let rank = self.rank;
        let base_len = base.len();
        let a_len = lora_a.len();
        let b_len = lora_b.len();

        // Calculate dimensions using known rank
        let (in_features, out_features) = self.calculate_lora_dimensions(base_len, a_len, b_len)?;

        debug!(
            "Merging LoRA: base {out_features}×{in_features} ({base_len}), A {rank}×{in_features} ({a_len}), B {out_features}×{rank} ({b_len}), rank={rank}, scaling={:.6}",
            self.scaling
        );

        if self.scaling.abs() > 1e3 {
            warn!("Large scaling factor detected: {:.6}", self.scaling);
        }

        // apply merge: W = W_base + scaling * (B @ A)
        let mut result = base.to_vec();
        result.par_iter_mut().enumerate().for_each(|(idx, base_val)| {
            let out_idx = idx / in_features;
            let in_idx = idx % in_features;

            let mut delta_val = 0.0f32;
            for r in 0..self.rank {
                let b_val = lora_b[out_idx * self.rank + r];
                let a_val = lora_a[r * in_features + in_idx];
                delta_val += b_val * a_val;
            }

            *base_val += self.scaling * delta_val;
        });

        let (max_abs_delta, avg_abs_delta, max_abs_base, avg_abs_base) =
            self.compute_merge_statistics(base, &result)?;

        debug!(
            "LoRA merge complete: max_delta={max_abs_delta:.6}, avg_delta={avg_abs_delta:.6}, max_base={max_abs_base:.6}, avg_base={avg_abs_base:.6}, relative_change={:.3}%",
            if avg_abs_base > 1e-12 { (avg_abs_delta / avg_abs_base) * 100.0 } else { 0.0 }
        );

        Ok(result)
    }

    /// Calculate LoRA dimensions using the known rank from config
    /// Returns (in_features, out_features) or error if dimensions don't match
    fn calculate_lora_dimensions(&self, base_len: usize, a_len: usize, b_len: usize) -> Result<(usize, usize)> {
        // With known rank, we can directly calculate dimensions
        // LoRA format: A: (rank, in_features), B: (out_features, rank)

        if a_len % self.rank != 0 {
            anyhow::bail!("LoRA A tensor size ({}) is not divisible by rank ({})", a_len, self.rank);
        }

        if b_len % self.rank != 0 {
            anyhow::bail!("LoRA B tensor size ({}) is not divisible by rank ({})", b_len, self.rank);
        }

        let in_features = a_len / self.rank;
        let out_features = b_len / self.rank;

        // Verify that dimensions are consistent with base weight
        if in_features * out_features != base_len {
            anyhow::bail!(
                "Dimension mismatch: base tensor size ({base_len}) doesn't match calculated dimensions ({out_features}×{in_features} = {})",
                in_features * out_features
            );
        }

        if in_features == 0 || out_features == 0 {
            anyhow::bail!("Invalid dimensions: in_features={in_features}, out_features={out_features}",);
        }

        debug!(
            "Calculated LoRA dimensions: rank={}, in_features={in_features}, out_features={out_features}",
            self.rank,
        );

        Ok((in_features, out_features))
    }

    /// Compute statistics for LoRA merge validation and logging using parallel processing
    fn compute_merge_statistics(&self, base: &[f32], result: &[f32]) -> Result<(f32, f32, f32, f32)> {
        if base.len() != result.len() {
            anyhow::bail!("Base and result tensor lengths don't match: {} vs {}", base.len(), result.len());
        }

        // Parallel computation of statistics with overflow checking
        let stats_result: Result<(f32, f64, f32, f64)> = base
            .par_iter()
            .zip(result.par_iter())
            .enumerate()
            .map(|(idx, (&base_val, &result_val))| {
                // Check for overflow/NaN in result during statistics computation
                if !result_val.is_finite() {
                    anyhow::bail!("Non-finite value detected in result at index {idx}: {result_val}");
                }

                let delta = (result_val - base_val).abs();
                let abs_base = base_val.abs();
                Ok((delta, delta as f64, abs_base, abs_base as f64))
            })
            .try_reduce(
                || (0.0f32, 0.0f64, 0.0f32, 0.0f64),
                |acc, curr| {
                    let (max_delta_acc, sum_delta_acc, max_base_acc, sum_base_acc) = acc;
                    let (delta, delta_f64, abs_base, abs_base_f64) = curr;
                    Ok((
                        max_delta_acc.max(delta),
                        sum_delta_acc + delta_f64,
                        max_base_acc.max(abs_base),
                        sum_base_acc + abs_base_f64,
                    ))
                },
            );

        let (max_abs_delta, sum_abs_delta, max_abs_base, sum_abs_base) = stats_result?;

        let len = base.len() as f64;
        let avg_abs_delta = (sum_abs_delta / len) as f32;
        let avg_abs_base = (sum_abs_base / len) as f32;

        Ok((max_abs_delta, avg_abs_delta, max_abs_base, avg_abs_base))
    }
}
