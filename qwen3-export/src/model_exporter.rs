#[cfg(test)]
#[path = "../tests/unit/model_exporter_test.rs"]
mod model_exporter_test;

use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use log::{info, warn};
use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::ModelConfig;
use crate::tensor_reader::TensorReader;
use crate::utils::ProgressTracker;

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

    const LAYER_WEIGHT_PATTERNS: &'static [&'static str] = &[
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
        "mlp.up_proj.weight",
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

    /// Export binary model with quantized weights using streaming to minimize memory usage
    pub fn export_binary_model(&self, model_path: &Path, output_path: &Path) -> Result<()> {
        let mut tensor_reader = TensorReader::new(model_path)?;

        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Check if classifier is shared by comparing tensor values (like Python)
        // If lm_head.weight and model.embed_tokens.weight are identical, classifier is shared
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
        self.write_norm_weights(&mut writer, &mut tensor_reader)?;

        // Stream and quantize weights one by one
        self.stream_and_quantize_weights(&mut writer, &mut tensor_reader, shared_classifier)?;

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
        tensor_reader: &mut TensorReader,
    ) -> Result<()> {
        info!("Writing normalization weights...");

        // Attention norms
        for layer_idx in 0..self.config.n_layers {
            let attn_norm_key = format!("model.layers.{layer_idx}.input_layernorm.weight");
            if let Some(attn_norm) = tensor_reader.load_tensor(&attn_norm_key)? {
                for &value in &attn_norm {
                    writer.write_f32::<LittleEndian>(value)?;
                }
            } else {
                return Err(anyhow::anyhow!("Missing weight: {attn_norm_key}"));
            }
        }

        // FFN norms
        for layer_idx in 0..self.config.n_layers {
            let ffn_norm_key = format!("model.layers.{layer_idx}.post_attention_layernorm.weight");
            if let Some(ffn_norm) = tensor_reader.load_tensor(&ffn_norm_key)? {
                for &value in &ffn_norm {
                    writer.write_f32::<LittleEndian>(value)?;
                }
            } else {
                return Err(anyhow::anyhow!("Missing weight: {ffn_norm_key}"));
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
            let lq_key = format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
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
            let lk_key = format!("model.layers.{layer_idx}.self_attn.k_norm.weight");
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

    /// Stream and quantize weights one by one to minimize memory usage
    fn stream_and_quantize_weights<W: Write>(
        &self,
        writer: &mut W,
        tensor_reader: &mut TensorReader,
        shared_classifier: bool,
    ) -> Result<()> {
        // Build weight tensor list to match Python ordering EXACTLY
        // Python: embed_tokens, [all q_proj], [all k_proj], [all v_proj], [all o_proj], [all gate_proj], [all down_proj], [all up_proj], [lm_head if not shared]
        let estimated_capacity = 1  // embed_tokens
            + (self.config.n_layers as usize * Self::LAYER_WEIGHT_PATTERNS.len())  // layer weights
            + usize::from(!shared_classifier); // classifier if not shared
        let mut weight_tensors = Vec::with_capacity(estimated_capacity);

        // First: embedding tokens
        weight_tensors.push(Self::EMBED_TOKENS_KEY.to_string());

        // Then: group by tensor type across all layers (matching Python exactly)
        for &tensor_type in Self::LAYER_WEIGHT_PATTERNS {
            for layer_idx in 0..self.config.n_layers {
                weight_tensors.push(format!("model.layers.{layer_idx}.{tensor_type}"));
            }
        }

        // Finally: classifier if not shared (matching Python logic)
        if !shared_classifier {
            weight_tensors.push(Self::LM_HEAD_KEY.to_string());
        }

        let progress = ProgressTracker::new(weight_tensors.len(), "Quantizing");

        // Process each weight tensor individually
        let max_errors: Result<Vec<f32>> = weight_tensors
            .iter()
            .enumerate()
            .map(|(i, tensor_name)| {
                progress.set_current(i + 1);

                let weight_tensor = tensor_reader
                    .load_tensor(tensor_name)?
                    .ok_or_else(|| anyhow::anyhow!("Missing weight tensor: {tensor_name}"))?;

                if weight_tensor.is_empty() {
                    warn!("Empty weight tensor: {tensor_name}");
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
