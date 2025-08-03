#[cfg(test)]
#[path = "../tests/unit/model_exporter_test.rs"]
mod model_exporter_test;

use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use log::{info, warn};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::lora_merger::LoraMerger;
use crate::models::{Architecture, HeaderInfo, NormWeightLayer, WeightLayer, create_architecture};
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

    pub fn new(config: ModelConfig, group_size: usize) -> Self {
        let optimal_group_size = Self::find_optimal_group_size(config.dim as usize, group_size);
        if optimal_group_size != group_size {
            info!("Adjusted group size from {} to {} to fit hidden_dim {}", group_size, optimal_group_size, config.dim);
        }
        Self { config, group_size: optimal_group_size }
    }

    /// Find optimal group size that divides hidden_dim and is reasonable
    fn find_optimal_group_size(hidden_dim: usize, requested_size: usize) -> usize {
        let mut size = requested_size.min(hidden_dim);

        // Find largest size that divides hidden_dim
        while size >= Self::MIN_GROUP_SIZE && hidden_dim % size != 0 {
            size /= 2;
        }

        size.max(Self::MIN_GROUP_SIZE)
    }

    /// Create exporter from ModelInfo (recommended for new code)
    pub fn new_from_model_info(model_info: &ModelInfo, group_size: usize) -> Self {
        Self::new(model_info.config.clone(), group_size)
    }

    /// Export binary model with quantized weights using streaming to minimize memory usage
    pub fn export_binary_model(&self, model_path: &Path, output_path: &Path, model_info: &ModelInfo) -> Result<()> {
        let tensor_reader = TensorReader::new(model_path)?;

        #[cfg(debug_assertions)]
        log::debug!("Tensor names: {:?}", tensor_reader.list_tensor_names()?);

        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        let architecture = create_architecture(model_info, &tensor_reader);
        let header_info = architecture.header()?;

        // Write header (256 bytes)
        self.write_header(&mut writer, &header_info)?;

        // Write normalization weights (fp32) - these are small
        self.write_norm_weights(architecture.as_ref(), &mut writer, &tensor_reader)?;

        // Stream and quantize weights one by one
        self.stream_and_quantize_weights(
            architecture.as_ref(),
            &mut writer,
            &tensor_reader,
            header_info.shared_classifier,
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
            return Err(anyhow::anyhow!("Weight length is not a multiple of group_size"));
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
                let scale = if group_max > 0.0 { group_max / 127.0 } else { 1.0 };

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

        Ok(QuantizedWeight { int8_data, scales, max_error })
    }

    /// Write binary header
    fn write_header<W: Write>(&self, writer: &mut W, header_info: &HeaderInfo) -> Result<()> {
        // Magic number "ajc1" in ASCII
        writer.write_u32::<LittleEndian>(Self::MAGIC_NUMBER)?;

        // Version
        writer.write_i32::<LittleEndian>(Self::VERSION)?;

        // Model parameters (10 int32 values)
        writer.write_u32::<LittleEndian>(header_info.architecture_id as u32)?;
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
        let current_pos = 4 + 4 + 4 + 10 * 4; // magic + version + architecture_id + 10 params
        let padding = Self::HEADER_SIZE - current_pos;
        let zeros = vec![0u8; padding];
        writer.write_all(&zeros)?;

        Ok(())
    }

    /// Write normalization weights (fp32).
    fn write_norm_weights<W: Write>(
        &self,
        architecture: &dyn Architecture,
        writer: &mut W,
        tensor_reader: &TensorReader,
    ) -> Result<()> {
        info!("Writing normalization weights...");

        let mut write_fn = |tensor_name: &str, is_required| -> Result<()> {
            match (tensor_reader.load_tensor(tensor_name)?, is_required) {
                (Some(attn_norm), _) => {
                    for &value in &attn_norm {
                        writer.write_f32::<LittleEndian>(value)?;
                    }
                }
                (None, false) => {
                    for _ in 0..self.config.head_dim as usize {
                        writer.write_f32::<LittleEndian>(1.0)?;
                    }
                }
                (None, true) => anyhow::bail!("Missing weight for tensor_name: '{tensor_name}'"),
            }

            Ok(())
        };

        architecture.norm_weight_layers().iter().try_for_each(|&NormWeightLayer { name, layered, is_required }| {
            if layered {
                for layer_idx in 0..self.config.n_layers {
                    let layer_name = name.replace("{}", &layer_idx.to_string());
                    write_fn(&layer_name, is_required)?;
                }
            } else {
                write_fn(name, is_required)?;
            }

            Ok(())
        })
    }

    /// Stream and quantize weights one by one to minimize memory usage (LoRA-aware)
    fn stream_and_quantize_weights<W: Write>(
        &self,
        architecture: &dyn Architecture,
        writer: &mut W,
        tensor_reader: &TensorReader,
        shared_classifier: bool,
        model_type: &ModelType,
    ) -> Result<()> {
        let estimated_capacity = 1  // embed_tokens
        + architecture.weight_layers().len()  // layer weights
        + usize::from(!shared_classifier); // classifier if not shared

        let mut weight_tensors = Vec::with_capacity(estimated_capacity);

        // First: embedding tokens
        weight_tensors.push((architecture.embed_tokens_layer().to_string(), None, None));

        // Then: layer weights
        for WeightLayer { tensor_name, component, layer_idx } in architecture.weight_layers() {
            weight_tensors.push((tensor_name.clone(), Some(component.to_string()), Some(*layer_idx)));
        }

        // Then Classifier if not shared
        if !shared_classifier {
            weight_tensors.push((architecture.lm_head_layer().to_string(), None, None));
        }

        let lora_merger = if let ModelType::LoRA(lora_config) = model_type {
            Some(LoraMerger::new(tensor_reader, lora_config.lora_alpha, lora_config.r)?)
        } else {
            None
        };

        let progress = ProgressTracker::new(weight_tensors.len(), "Quantizing");

        // Process each weight tensor individually
        let max_errors: Result<Vec<f32>> = weight_tensors
            .iter()
            .enumerate()
            .map(|(i, (tensor_name, tensor_type, layer_idx))| {
                progress.set_current(i + 1, Some(tensor_name));

                // Load the base tensor (same for both base and LoRA models)
                let mut weight_tensor = tensor_reader
                    .load_tensor(tensor_name)?
                    .ok_or_else(|| anyhow::anyhow!("Missing weight tensor: {}", tensor_name))?;

                // If LoRA is used, try to merge adapters
                if let (Some(lora_merger), Some(layer_idx), Some(component)) =
                    (lora_merger.as_ref(), layer_idx, tensor_type.as_ref())
                {
                    if let Some(merged_weights) =
                        lora_merger.try_merge_lora_adapters(&weight_tensor, component, *layer_idx)?
                    {
                        weight_tensor = merged_weights;
                    }
                }

                if weight_tensor.is_empty() {
                    warn!("Empty weight tensor: {}", tensor_name);
                    return Ok(0.0);
                }

                // Quantize this tensor
                let quantized = self.quantize_q80(&weight_tensor)?;

                // Write quantized data using iterators
                quantized.int8_data.iter().try_for_each(|&value| writer.write_i8(value))?;
                quantized.scales.iter().try_for_each(|&scale| writer.write_f32::<LittleEndian>(scale))?;

                Ok(quantized.max_error)
            })
            .collect();

        let max_errors = max_errors?;

        // Print overall max error
        let overall_max_error = max_errors.iter().fold(0.0f32, |acc, &x| acc.max(x));
        info!("Quantized {} weight tensors to Q8_0 with max error: {overall_max_error:.8}", weight_tensors.len());

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
        if x >= 0.0 { rounded - 1.0 } else { rounded + 1.0 }
    }
}
