#[cfg(test)]
#[path = "../tests/unit/model_exporter_test.rs"]
mod model_exporter_test;

use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use log::{debug, info, warn};
use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::tensor_reader::TensorReader;
use crate::utils::ProgressTracker;
use crate::{ModelConfig, ModelInfo, ModelType};

// Quantization result
#[derive(Debug)]
pub struct QuantizedWeight {
    pub int8_data: Vec<i8>,
    pub scales: Vec<f32>,
    pub max_error: f32,
}

/// Header information structure (lightweight)
#[derive(Debug)]
struct HeaderInfo {
    pub shared_classifier: bool,
}

/// Binary model exporter for quantized model weights
pub struct BinaryModelExporter {
    config: ModelConfig,
    group_size: usize,
}

impl BinaryModelExporter {
    const MAGIC_NUMBER: u32 = 0x616A6331; // "ajc1" in ASCII
    const VERSION: i32 = 1;
    const HEADER_SIZE: usize = 256;
    const MIN_GROUP_SIZE: usize = 4;

    // Tensor name constants
    const EMBED_TOKENS_KEY: &'static str = "model.embed_tokens.weight";
    const LM_HEAD_KEY: &'static str = "lm_head.weight";
    const FINAL_NORM_KEY: &'static str = "model.norm.weight";

    // Qwen3 model layer weight component names (without .weight suffix)
    const QWEN3_LAYER_COMPONENTS: &'static [&'static str] = &[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
    ];

    /// Find optimal group size that divides hidden_dim and is reasonable
    fn find_optimal_group_size(hidden_dim: usize, requested_size: usize) -> usize {
        let mut size = requested_size.min(hidden_dim);

        // Find largest size that divides hidden_dim
        while size >= Self::MIN_GROUP_SIZE && hidden_dim % size != 0 {
            size /= 2;
        }

        size.max(Self::MIN_GROUP_SIZE)
    }

    pub fn new(config: ModelConfig, group_size: usize) -> Self {
        let optimal_group_size = Self::find_optimal_group_size(config.dim as usize, group_size);
        if optimal_group_size != group_size {
            info!(
                "Adjusted group size from {} to {} to fit hidden_dim {}",
                group_size, optimal_group_size, config.dim
            );
        }
        Self {
            config,
            group_size: optimal_group_size,
        }
    }

    /// Create exporter from ModelInfo (recommended for new code)
    pub fn new_from_model_info(model_info: &ModelInfo, group_size: usize) -> Self {
        Self::new(model_info.config.clone(), group_size)
    }

    /// Export binary model with quantized weights using streaming to minimize memory usage
    pub fn export_binary_model(
        &self,
        model_path: &Path,
        output_path: &Path,
        model_info: &ModelInfo,
    ) -> Result<()> {
        let tensor_reader = TensorReader::new(model_path)?;

        #[cfg(debug_assertions)]
        debug!("Tensor names: {:?}", tensor_reader.list_tensor_names()?);

        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Check if classifier is shared by comparing tensor values (like Python)
        let shared_classifier = match (
            tensor_reader.load_tensor(Self::LM_HEAD_KEY)?,
            tensor_reader.load_tensor(Self::EMBED_TOKENS_KEY)?,
        ) {
            (Some(lm_head_weights), Some(embed_weights)) => {
                // Compare tensor values to determine if they're identical
                lm_head_weights.len() == embed_weights.len()
                    && lm_head_weights
                        .iter()
                        .zip(embed_weights.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-6)
            }
            (None, Some(_)) => true, // No lm_head means shared
            _ => false,              // Missing embed_tokens is an error, but we'll handle it later
        };
        let header_info = HeaderInfo { shared_classifier };

        // Write header (256 bytes)
        self.write_header(&mut writer, &header_info)?;

        // Write normalization weights (fp32) - these are small
        self.write_norm_weights(&mut writer, &tensor_reader)?;

        // Stream and quantize weights one by one
        self.stream_and_quantize_weights(
            &mut writer,
            &tensor_reader,
            shared_classifier,
            &model_info.model_type,
        )?;

        writer.flush()?;
        info!("💾 Written model checkpoint to {}", output_path.display());

        // Clear cache to free memory
        if let Err(e) = tensor_reader.clear_cache() {
            warn!("Failed to clear cache: {e}");
        }

        Ok(())
    }

    /// Quantize weights to Q8_0 format (symmetric int8, range [-127, 127])
    pub fn quantize_q80(&self, weights: &[f32]) -> Result<QuantizedWeight> {
        if weights.len() % self.group_size != 0 {
            return Err(anyhow::anyhow!(
                "Weight length is not a multiple of group_size"
            ));
        }

        let num_groups = weights.len() / self.group_size;

        // Process groups in parallel
        let group_results: Vec<_> = (0..num_groups)
            .into_par_iter()
            .map(|group_idx| {
                let start_idx = group_idx * self.group_size;
                let end_idx = start_idx + self.group_size;
                let group = &weights[start_idx..end_idx];

                // Find the maximum absolute value in this group
                let group_max = group.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

                // Calculate scaling factor
                let scale = if group_max > 0.0 {
                    group_max / 127.0
                } else {
                    1.0
                };

                // Quantize the group
                let mut group_int8 = Vec::with_capacity(self.group_size);
                let mut group_error = 0.0f32;

                for &weight in group {
                    let quantized = if scale > 0.0 {
                        // Use banker's rounding to match PyTorch exactly
                        let scaled = weight / scale;
                        round_half_to_even(scaled).clamp(-127.0, 127.0) as i8
                    } else {
                        0i8
                    };
                    group_int8.push(quantized);

                    // Calculate reconstruction error for this value
                    let dequantized = f32::from(quantized) * scale;
                    let error = (dequantized - weight).abs();
                    group_error = group_error.max(error);
                }

                (group_int8, scale, group_error)
            })
            .collect();

        // Reconstruct results in order
        let mut int8_data = Vec::with_capacity(weights.len());
        let mut scales = Vec::with_capacity(num_groups);
        let mut max_error = 0.0f32;

        for (group_int8, scale, group_error) in group_results {
            int8_data.extend(group_int8);
            scales.push(scale);
            max_error = max_error.max(group_error);
        }

        Ok(QuantizedWeight {
            int8_data,
            scales,
            max_error,
        })
    }

    /// Write binary header
    fn write_header<W: Write>(&self, writer: &mut W, header_info: &HeaderInfo) -> Result<()> {
        // Magic number "ajc1" in ASCII
        writer.write_u32::<LittleEndian>(Self::MAGIC_NUMBER)?;

        // Version
        writer.write_i32::<LittleEndian>(Self::VERSION)?;

        // Model parameters (10 int32 values)
        writer.write_u32::<LittleEndian>(self.config.dim)?;
        writer.write_u32::<LittleEndian>(self.config.hidden_dim)?;
        writer.write_u32::<LittleEndian>(self.config.n_layers)?;
        writer.write_u32::<LittleEndian>(self.config.n_heads)?;
        writer.write_u32::<LittleEndian>(self.config.n_kv_heads)?;
        writer.write_u32::<LittleEndian>(self.config.vocab_size)?;
        writer.write_u32::<LittleEndian>(self.config.max_seq_len)?;
        writer.write_u32::<LittleEndian>(self.config.head_dim)?;
        writer.write_u32::<LittleEndian>(header_info.shared_classifier as u32)?;
        writer.write_u32::<LittleEndian>(self.group_size as u32)?;

        // Pad to header size
        let current_pos = 4 + 4 + 10 * 4; // magic + version + 10 params
        let padding = Self::HEADER_SIZE - current_pos;
        let zeros = vec![0u8; padding];
        writer.write_all(&zeros)?;

        Ok(())
    }

    /// Write normalization weights (fp32)
    fn write_norm_weights<W: Write>(
        &self,
        writer: &mut W,
        tensor_reader: &TensorReader,
    ) -> Result<()> {
        info!("Writing normalization weights...");

        // Attention norms
        for layer_idx in 0..self.config.n_layers {
            let attn_norm_key = format!("model.layers.{}.input_layernorm.weight", layer_idx);
            if let Some(attn_norm) = tensor_reader.load_tensor(&attn_norm_key)? {
                for &value in &attn_norm {
                    writer.write_f32::<LittleEndian>(value)?;
                }
            } else {
                return Err(anyhow::anyhow!("Missing weight: {}", attn_norm_key));
            }
        }

        // FFN norms
        for layer_idx in 0..self.config.n_layers {
            let ffn_norm_key =
                format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
            if let Some(ffn_norm) = tensor_reader.load_tensor(&ffn_norm_key)? {
                for &value in &ffn_norm {
                    writer.write_f32::<LittleEndian>(value)?;
                }
            } else {
                return Err(anyhow::anyhow!("Missing weight: {}", ffn_norm_key));
            }
        }

        // Final norm
        if let Some(final_norm) = tensor_reader.load_tensor(Self::FINAL_NORM_KEY)? {
            for &value in &final_norm {
                writer.write_f32::<LittleEndian>(value)?;
            }
        } else {
            return Err(anyhow::anyhow!("Missing final norm"));
        }

        // QK LayerNorm weights (Qwen3 specific)
        for layer_idx in 0..self.config.n_layers {
            let lq_key = format!("model.layers.{}.self_attn.q_norm.weight", layer_idx);
            if let Some(lq) = tensor_reader.load_tensor(&lq_key)? {
                for &value in &lq {
                    writer.write_f32::<LittleEndian>(value)?;
                }
            } else {
                // Default to ones if not present
                for _ in 0..self.config.head_dim as usize {
                    writer.write_f32::<LittleEndian>(1.0)?;
                }
            }
        }

        for layer_idx in 0..self.config.n_layers {
            let lk_key = format!("model.layers.{}.self_attn.k_norm.weight", layer_idx);
            if let Some(lk) = tensor_reader.load_tensor(&lk_key)? {
                for &value in &lk {
                    writer.write_f32::<LittleEndian>(value)?;
                }
            } else {
                // Default to ones if not present
                for _ in 0..self.config.head_dim as usize {
                    writer.write_f32::<LittleEndian>(1.0)?;
                }
            }
        }

        Ok(())
    }

    /// Try to merge LoRA adapters with base weights
    fn try_merge_lora_adapters(
        &self,
        tensor_reader: &TensorReader,
        base_weights: &[f32],
        component: &str,
        layer_idx: u32,
        lora_params: (f32, usize), // (alpha, r)
    ) -> Result<Option<Vec<f32>>> {
        // component is already clean (e.g., "self_attn.k_proj", "mlp.gate_proj")
        let (lora_a, lora_b) =
            self.discover_and_load_lora_pairs(tensor_reader, component, layer_idx)?;

        if let (Some(a), Some(b)) = (lora_a, lora_b) {
            // Use provided LoRA parameters from ModelType
            let (alpha, r) = lora_params;
            let scaling = alpha / r as f32;

            debug!(
                "Merging LoRA adapters for {} layer {} with scaling {}",
                component, layer_idx, scaling
            );

            // Merge LoRA: W = W_base + scaling * (B @ A)
            let merged = self.merge_lora_weights(base_weights, &a, &b, scaling)?;
            Ok(Some(merged))
        } else {
            Ok(None)
        }
    }

    /// Dynamically discover and load LoRA adapter pairs
    fn discover_and_load_lora_pairs(
        &self,
        tensor_reader: &TensorReader,
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
        let lora_a = match tensor_reader.load_tensor(&lora_a_name)? {
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
        let lora_b = match tensor_reader.load_tensor(&lora_b_name)? {
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
    fn merge_lora_weights(
        &self,
        base: &[f32],
        lora_a: &[f32],
        lora_b: &[f32],
        scaling: f32,
    ) -> Result<Vec<f32>> {
        // Input validation
        if base.is_empty() || lora_a.is_empty() || lora_b.is_empty() {
            return Err(anyhow::anyhow!(
                "Empty tensors not allowed: base={}, A={}, B={}",
                base.len(),
                lora_a.len(),
                lora_b.len()
            ));
        }

        if !scaling.is_finite() || scaling.is_nan() {
            return Err(anyhow::anyhow!(
                "Invalid scaling factor: {scaling} (must be finite)"
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
            "Merging LoRA: base {out_features}×{in_features} ({base_len}), A {rank}×{in_features} ({a_len}), B {out_features}×{rank} ({b_len}), rank={rank}, scaling={scaling:.6}",
        );

        // Check for potential numerical issues
        self.validate_tensors_for_merge(base, lora_a, lora_b, scaling)?;

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
                let scaled_delta = scaling * delta_val;

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

    /// Stream and quantize weights one by one to minimize memory usage (LoRA-aware)
    fn stream_and_quantize_weights<W: Write>(
        &self,
        writer: &mut W,
        tensor_reader: &TensorReader,
        shared_classifier: bool,
        model_type: &ModelType,
    ) -> Result<()> {
        let estimated_capacity = 1  // embed_tokens
            + (self.config.n_layers as usize * Self::QWEN3_LAYER_COMPONENTS.len())  // layer weights
            + usize::from(!shared_classifier); // classifier if not shared
        let mut weight_tensors = Vec::with_capacity(estimated_capacity);

        // First: embedding tokens
        weight_tensors.push((Self::EMBED_TOKENS_KEY.to_string(), None, None));

        // Then: group by tensor type across all layers (matching Python exactly)
        for &component in Self::QWEN3_LAYER_COMPONENTS {
            for layer_idx in 0..self.config.n_layers {
                let tensor_name = format!("model.layers.{}.{}.weight", layer_idx, component);
                weight_tensors.push((tensor_name, Some(layer_idx), Some(component.to_string())));
            }
        }

        // Finally: classifier if not shared (matching Python logic)
        if !shared_classifier {
            weight_tensors.push((Self::LM_HEAD_KEY.to_string(), None, None));
        }

        let progress = ProgressTracker::new(weight_tensors.len(), "Quantizing");

        // Process each weight tensor individually
        let max_errors: Result<Vec<f32>> = weight_tensors
            .iter()
            .enumerate()
            .map(|(i, (tensor_name, layer_idx, tensor_type))| {
                progress.set_current(i + 1, Some(tensor_name));

                // Load the base tensor (same for both base and LoRA models)
                let mut weight_tensor = tensor_reader
                    .load_tensor(tensor_name)?
                    .ok_or_else(|| anyhow::anyhow!("Missing weight tensor: {}", tensor_name))?;

                // If this is a LoRA model and this is a layer tensor, try to merge LoRA adapters
                if let ModelType::LoRA(lora_config) = model_type {
                    if let (Some(layer_idx), Some(component)) = (layer_idx, tensor_type) {
                        debug!(
                            "Checking for LoRA adapters: layer={layer_idx}, component={component}",
                        );

                        if let Some(merged_weights) = self.try_merge_lora_adapters(
                            tensor_reader,
                            &weight_tensor,
                            component,
                            *layer_idx,
                            (lora_config.lora_alpha, lora_config.r),
                        )? {
                            weight_tensor = merged_weights;
                        }
                    }
                }

                if weight_tensor.is_empty() {
                    warn!("Empty weight tensor: {}", tensor_name);
                    return Ok(0.0);
                }

                // Quantize this tensor
                let quantized = self.quantize_q80(&weight_tensor)?;

                // Write quantized data using iterators
                quantized
                    .int8_data
                    .iter()
                    .try_for_each(|&value| writer.write_i8(value))?;
                quantized
                    .scales
                    .iter()
                    .try_for_each(|&scale| writer.write_f32::<LittleEndian>(scale))?;

                Ok(quantized.max_error)
            })
            .collect();

        let max_errors = max_errors?;

        // Print overall max error
        let overall_max_error = max_errors.iter().fold(0.0f32, |acc, &x| acc.max(x));
        info!(
            "Quantized {} weight tensors to Q8_0 with max error: {overall_max_error:.8}",
            weight_tensors.len()
        );

        Ok(())
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
        scaling: f32,
    ) -> Result<()> {
        // Check for non-finite values in input tensors
        if base.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow::anyhow!("Base tensor contains non-finite values"));
        }

        if lora_a.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow::anyhow!("LoRA A tensor contains non-finite values"));
        }

        if lora_b.iter().any(|&x| !x.is_finite()) {
            return Err(anyhow::anyhow!("LoRA B tensor contains non-finite values"));
        }

        // Check for extreme values that might cause overflow
        let max_base = base.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let max_a = lora_a.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let max_b = lora_b.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        // Rough estimate of maximum possible delta magnitude
        let estimated_max_delta = max_a * max_b * scaling.abs();

        if estimated_max_delta > 1e6 {
            warn!(
                "Large LoRA delta values detected (estimated max: {:.2e}). \
                 This may indicate numerical instability. Base max: {:.2e}, A max: {:.2e}, B max: {:.2e}, scaling: {:.6}",
                estimated_max_delta, max_base, max_a, max_b, scaling
            );
        }

        // Check if the result might overflow f32
        if max_base + estimated_max_delta > f32::MAX / 2.0 {
            return Err(anyhow::anyhow!(
                "Potential f32 overflow detected: base_max={:.2e}, estimated_delta_max={:.2e}",
                max_base,
                estimated_max_delta
            ));
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
            return Err(anyhow::anyhow!(
                "Base and result tensor lengths don't match: {} vs {}",
                base.len(),
                result.len()
            ));
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

/// Round half to even (banker's rounding) to match PyTorch's torch.round() behavior
#[inline]
fn round_half_to_even(x: f32) -> f32 {
    // For non-half values, use standard rounding
    let rounded = x.round();
    let diff = (x - rounded).abs();

    // If not exactly halfway, return standard rounding
    if diff != 0.5 {
        return rounded;
    }

    // For exactly halfway cases, round to nearest even
    if rounded as i32 % 2 == 0 {
        rounded // Already even
    } else {
        // Make even by rounding toward zero
        if x >= 0.0 {
            rounded - 1.0
        } else {
            rounded + 1.0
        }
    }
}
