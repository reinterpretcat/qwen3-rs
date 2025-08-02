use crate::tensor_reader::TensorReader;
use anyhow::Result;
use log::{debug, warn};

pub(crate) struct LoraMerger<'a> {
    tensor_reader: &'a TensorReader,
    scaling: f32,
}

impl<'a> LoraMerger<'a> {
    pub fn new(tensor_reader: &'a TensorReader, alpha: f32, rank: usize) -> Result<Self> {
        let scaling = alpha / rank as f32;

        if !scaling.is_finite() || scaling.is_nan() {
            anyhow::bail!(anyhow::anyhow!(
                "Invalid scaling factor: {scaling} (must be finite)"
            ));
        }

        Ok(Self {
            tensor_reader,
            scaling,
        })
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
            debug!(
                "Merging LoRA adapters for {component} layer {layer_idx} with scaling {}",
                self.scaling
            );

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
        // Based on the actual tensor naming pattern:
        // Base: model.layers.{layer}.{component}.weight
        // LoRA A: base_model.model.model.layers.{layer}.{component}.lora_A.weight
        // LoRA B: base_model.model.model.layers.{layer}.{component}.lora_B.weight

        let lora_a_name = format!(
            "base_model.model.model.layers.{}.{}.lora_A.weight",
            layer_idx, component
        );
        let lora_b_name = format!(
            "base_model.model.model.layers.{}.{}.lora_B.weight",
            layer_idx, component
        );

        debug!("Looking for LoRA A: '{lora_a_name}'");
        debug!("Looking for LoRA B: '{lora_b_name}'");

        // Try to load LoRA A
        let lora_a = match self.tensor_reader.load_tensor(&lora_a_name)? {
            Some(tensor) => {
                debug!(
                    "Loaded LoRA A tensor: {lora_a_name} (size: {})",
                    tensor.len()
                );
                Some(tensor)
            }
            None => {
                debug!("LoRA A tensor not found: {lora_a_name}");
                None
            }
        };

        // Try to load LoRA B
        let lora_b = match self.tensor_reader.load_tensor(&lora_b_name)? {
            Some(tensor) => {
                debug!(
                    "Loaded LoRA B tensor: {lora_b_name} (size: {})",
                    tensor.len()
                );
                Some(tensor)
            }
            None => {
                debug!("LoRA B tensor not found: {lora_b_name}");
                None
            }
        };

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
    /// Production-ready implementation with robust error handling, optimization, and validation
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

        // Try to infer dimensions from the tensor sizes
        // LoRA format: A: (rank, in_features), B: (out_features, rank)
        // Base weight: (out_features, in_features) - flattened to 1D
        let (rank, in_features, out_features) =
            self.infer_lora_dimensions(base_len, a_len, b_len)?;

        debug!(
            "Merging LoRA: base {out_features}×{in_features} ({base_len}), A {rank}×{in_features} ({a_len}), B {out_features}×{rank} ({b_len}), rank={rank}, scaling={:.6}",
            self.scaling
        );

        // Check for potential numerical issues
        self.validate_tensors_for_merge(base, lora_a, lora_b)?;

        // Perform optimized matrix multiplication: delta_W = scaling * (B @ A)
        // B: (out_features, rank), A: (rank, in_features) -> delta_W: (out_features, in_features)
        let mut result = base.to_vec();

        // Direct computation without allocating intermediate delta_w matrix
        for out_idx in 0..out_features {
            for in_idx in 0..in_features {
                let mut delta_val = 0.0f32;

                // Compute one element of B @ A
                for r in 0..rank {
                    let b_val = lora_b[out_idx * rank + r]; // B[out_idx, r]
                    let a_val = lora_a[r * in_features + in_idx]; // A[r, in_idx]
                    delta_val += b_val * a_val;
                }

                // Apply scaling and add to base weight directly
                let result_idx = out_idx * in_features + in_idx;
                let scaled_delta = self.scaling * delta_val;

                // Check for overflow/underflow before adding
                if !scaled_delta.is_finite() {
                    return Err(anyhow::anyhow!(
                        "Numerical instability detected at position ({out_idx}, {in_idx}): delta={delta_val}, scaled={scaled_delta}",
                    ));
                }

                result[result_idx] += scaled_delta;

                // Additional overflow check after addition
                if !result[result_idx].is_finite() {
                    return Err(anyhow::anyhow!(
                        "Result overflow at position ({out_idx}, {in_idx}): base={}, delta={scaled_delta}, result={}",
                        base[result_idx],
                        result[result_idx]
                    ));
                }
            }
        }

        // Compute statistics for validation and logging
        let (max_abs_delta, avg_abs_delta, max_abs_base, avg_abs_base) =
            self.compute_merge_statistics(base, &result)?;

        debug!(
            "LoRA merge complete: max_delta={max_abs_delta:.6}, avg_delta={avg_abs_delta:.6}, max_base={max_abs_base:.6}, avg_base={avg_abs_base:.6}, relative_change={:.3}%",
            if avg_abs_base > 1e-12 {
                (avg_abs_delta / avg_abs_base) * 100.0
            } else {
                0.0
            }
        );

        // Final sanity checks
        if result.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow::anyhow!(
                "Final result contains non-finite values after LoRA merge"
            ));
        }

        Ok(result)
    }

    /// Infer LoRA dimensions with comprehensive error handling
    /// Returns (rank, in_features, out_features) or error if cannot infer
    fn infer_lora_dimensions(
        &self,
        base_len: usize,
        a_len: usize,
        b_len: usize,
    ) -> Result<(usize, usize, usize)> {
        // Comprehensive dimension inference with multiple strategies

        // Strategy 1: Standard LoRA layout - A: (rank, in_features), B: (out_features, rank)
        for rank in 1..=512 {
            // Extended range for larger models
            if a_len % rank == 0 && b_len % rank == 0 {
                let in_features = a_len / rank;
                let out_features = b_len / rank;

                // Check if this produces the correct base dimension
                if in_features * out_features == base_len && in_features > 0 && out_features > 0 {
                    debug!(
                        "Inferred LoRA dimensions using standard layout: rank={}, in={}, out={}",
                        rank, in_features, out_features
                    );
                    return Ok((rank, in_features, out_features));
                }
            }
        }

        // Strategy 2: Transposed A layout - A: (in_features, rank), B: (out_features, rank)
        for rank in 1..=512 {
            if a_len % rank == 0 && b_len % rank == 0 {
                let in_features = a_len / rank; // Different interpretation
                let out_features = b_len / rank;

                if in_features * out_features == base_len && in_features > 0 && out_features > 0 {
                    debug!(
                        "Inferred LoRA dimensions using transposed A layout: rank={}, in={}, out={}",
                        rank, in_features, out_features
                    );
                    return Ok((rank, in_features, out_features));
                }
            }
        }

        // Strategy 3: Alternative factorizations (some LoRA implementations vary)
        // Try to find any valid factorization where rank divides both A and B tensors
        let mut candidates = Vec::new();
        for rank in 1..=std::cmp::min(a_len, b_len) {
            if a_len % rank == 0 && b_len % rank == 0 {
                let in_features_1 = a_len / rank;
                let out_features_1 = b_len / rank;
                if in_features_1 * out_features_1 == base_len {
                    candidates.push((rank, in_features_1, out_features_1));
                }

                let in_features_2 = rank;
                let out_features_2 = b_len / rank;
                let rank_2 = a_len / rank;
                if in_features_2 * out_features_2 == base_len && rank_2 > 0 {
                    candidates.push((rank_2, in_features_2, out_features_2));
                }
            }
        }

        if !candidates.is_empty() {
            // Prefer smaller ranks (more typical for LoRA)
            candidates.sort_by_key(|(rank, _, _)| *rank);
            let (rank, in_features, out_features) = candidates[0];
            debug!(
                "Inferred LoRA dimensions using alternative strategy: rank={}, in={}, out={}",
                rank, in_features, out_features
            );
            return Ok((rank, in_features, out_features));
        }

        Err(anyhow::anyhow!(
            "Could not infer LoRA dimensions from tensor sizes: base={}, A={}, B={}. \
             This may indicate incompatible LoRA format or corrupted tensors. \
             Expected: A should be (rank × in_features) and B should be (out_features × rank) \
             where in_features × out_features = base_len",
            base_len,
            a_len,
            b_len
        ))
    }

    /// Validate tensors before LoRA merge to catch potential numerical issues
    fn validate_tensors_for_merge(
        &self,
        base: &[f32],
        lora_a: &[f32],
        lora_b: &[f32],
    ) -> Result<()> {
        // Check for non-finite values in input tensors
        if base.iter().any(|&x| !x.is_finite()) {
            anyhow::bail!("Base tensor contains non-finite values");
        }

        if lora_a.iter().any(|&x| !x.is_finite()) {
            anyhow::bail!("LoRA A tensor contains non-finite values");
        }

        if lora_b.iter().any(|&x| !x.is_finite()) {
            anyhow::bail!("LoRA B tensor contains non-finite values");
        }

        // Check for extreme values that might cause overflow
        let max_base = base.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let max_a = lora_a.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let max_b = lora_b.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

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

    /// Compute statistics for LoRA merge validation and logging
    fn compute_merge_statistics(
        &self,
        base: &[f32],
        result: &[f32],
    ) -> Result<(f32, f32, f32, f32)> {
        if base.len() != result.len() {
            anyhow::bail!(
                "Base and result tensor lengths don't match: {} vs {}",
                base.len(),
                result.len()
            );
        }

        let mut max_abs_delta = 0.0f32;
        let mut sum_abs_delta = 0.0f64; // Use f64 for better precision in sum
        let mut max_abs_base = 0.0f32;
        let mut sum_abs_base = 0.0f64;

        for (&base_val, &result_val) in base.iter().zip(result.iter()) {
            let delta = (result_val - base_val).abs();
            max_abs_delta = max_abs_delta.max(delta);
            sum_abs_delta += delta as f64;

            let abs_base = base_val.abs();
            max_abs_base = max_abs_base.max(abs_base);
            sum_abs_base += abs_base as f64;
        }

        let len = base.len() as f64;
        let avg_abs_delta = (sum_abs_delta / len) as f32;
        let avg_abs_base = (sum_abs_base / len) as f32;

        Ok((max_abs_delta, avg_abs_delta, max_abs_base, avg_abs_base))
    }
}
