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

        let base_len = base.len();
        let a_len = lora_a.len();
        let b_len = lora_b.len();

        // Try to infer dimensions from the tensor sizes using known rank
        // LoRA format: A: (rank, in_features), B: (out_features, rank)
        // Base weight: (out_features, in_features) - flattened to 1D
        let (in_features, out_features) = self.calculate_lora_dimensions(base_len, a_len, b_len)?;

        let rank = self.rank;
        debug!(
            "Merging LoRA: base {out_features}×{in_features} ({base_len}), A {rank}×{in_features} ({a_len}), B {out_features}×{rank} ({b_len}), rank={rank}, scaling={:.6}",
            self.scaling
        );

        // Check for potential numerical issues
        self.validate_tensors_for_merge(base, lora_a, lora_b)?;

        // Perform parallel matrix multiplication: delta_W = scaling * (B @ A)
        // B: (out_features, rank), A: (rank, in_features) -> delta_W: (out_features, in_features)
        let mut result = base.to_vec();

        // Compute deltas in parallel, then apply them
        let deltas = (0..base.len())
            .into_par_iter()
            .map(|result_idx| {
                let out_idx = result_idx / in_features;
                let in_idx = result_idx % in_features;

                let mut delta_val = 0.0f32;

                // Compute one element of B @ A
                for r in 0..self.rank {
                    let b_val = lora_b[out_idx * self.rank + r]; // B[out_idx, r]
                    let a_val = lora_a[r * in_features + in_idx]; // A[r, in_idx]
                    delta_val += b_val * a_val;
                }

                // Apply scaling
                let scaled_delta = self.scaling * delta_val;

                // Check for overflow/underflow
                if !scaled_delta.is_finite() {
                    anyhow::bail!(
                        "Numerical instability detected at position ({out_idx}, {in_idx}): delta={delta_val}, scaled={scaled_delta}"
                    );
                }

                Ok(scaled_delta)
            })
            .collect::<Result<Vec<f32>>>()?;

        // Apply deltas and validate results
        for (idx, &delta) in deltas.iter().enumerate() {
            result[idx] += delta;

            if !result[idx].is_finite() {
                let out_idx = idx / in_features;
                let in_idx = idx % in_features;
                anyhow::bail!(
                    "Result overflow at position ({out_idx}, {in_idx}): base={}, delta={delta}, result={}",
                    base[idx],
                    result[idx]
                );
            }
        }

        // Compute statistics for validation and logging
        let (max_abs_delta, avg_abs_delta, max_abs_base, avg_abs_base) =
            self.compute_merge_statistics(base, &result)?;

        debug!(
            "LoRA merge complete: max_delta={max_abs_delta:.6}, avg_delta={avg_abs_delta:.6}, max_base={max_abs_base:.6}, avg_base={avg_abs_base:.6}, relative_change={:.3}%",
            if avg_abs_base > 1e-12 { (avg_abs_delta / avg_abs_base) * 100.0 } else { 0.0 }
        );

        // Final sanity checks
        if result.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow::anyhow!("Final result contains non-finite values after LoRA merge"));
        }

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

    /// Validate tensors before LoRA merge to catch potential numerical issues
    fn validate_tensors_for_merge(&self, base: &[f32], lora_a: &[f32], lora_b: &[f32]) -> Result<()> {
        // Check for non-finite values in input tensors
        let base_has_invalid = base.par_iter().any(|&x| !x.is_finite());
        if base_has_invalid {
            anyhow::bail!("Base tensor contains non-finite values");
        }

        let a_has_invalid = lora_a.par_iter().any(|&x| !x.is_finite());
        if a_has_invalid {
            anyhow::bail!("LoRA A tensor contains non-finite values");
        }

        let b_has_invalid = lora_b.par_iter().any(|&x| !x.is_finite());
        if b_has_invalid {
            anyhow::bail!("LoRA B tensor contains non-finite values");
        }

        // Computation of maximum absolute values
        let max_base = base.par_iter().map(|&x| x.abs()).reduce(|| 0.0f32, f32::max);
        let max_a = lora_a.par_iter().map(|&x| x.abs()).reduce(|| 0.0f32, f32::max);
        let max_b = lora_b.par_iter().map(|&x| x.abs()).reduce(|| 0.0f32, f32::max);

        // Rough estimate of maximum possible delta magnitude
        let estimated_max_delta = max_a * max_b * self.scaling.abs();

        if estimated_max_delta > 1e6 {
            warn!(
                "Large LoRA delta values detected (estimated max: {estimated_max_delta:.2e}). \
                 This may indicate numerical instability. Base max: {max_base:.2e}, A max: {max_a:.2e}, B max: {max_b:.2e}, scaling: {:.6}",
                self.scaling
            );
        }

        // Check if the result might overflow f32
        if max_base + estimated_max_delta > f32::MAX / 2.0 {
            anyhow::bail!(
                "Potential f32 overflow detected: base_max={max_base:.2e}, estimated_delta_max={estimated_max_delta:.2e}",
            );
        }

        Ok(())
    }

    /// Compute statistics for LoRA merge validation and logging using parallel processing
    fn compute_merge_statistics(&self, base: &[f32], result: &[f32]) -> Result<(f32, f32, f32, f32)> {
        if base.len() != result.len() {
            anyhow::bail!("Base and result tensor lengths don't match: {} vs {}", base.len(), result.len());
        }

        // Parallel computation of statistics using reduce operations
        let (max_abs_delta, sum_abs_delta, max_abs_base, sum_abs_base) = base
            .par_iter()
            .zip(result.par_iter())
            .map(|(&base_val, &result_val)| {
                let delta = (result_val - base_val).abs();
                let abs_base = base_val.abs();
                (delta, delta as f64, abs_base, abs_base as f64)
            })
            .reduce(
                || (0.0f32, 0.0f64, 0.0f32, 0.0f64),
                |(max_delta_acc, sum_delta_acc, max_base_acc, sum_base_acc),
                 (delta, delta_f64, abs_base, abs_base_f64)| {
                    (
                        max_delta_acc.max(delta),
                        sum_delta_acc + delta_f64,
                        max_base_acc.max(abs_base),
                        sum_base_acc + abs_base_f64,
                    )
                },
            );

        let len = base.len() as f64;
        let avg_abs_delta = (sum_abs_delta / len) as f32;
        let avg_abs_base = (sum_abs_base / len) as f32;

        Ok((max_abs_delta, avg_abs_delta, max_abs_base, avg_abs_base))
    }
}
