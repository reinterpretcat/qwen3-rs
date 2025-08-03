use std::fs::File;

use crate::{
    configuration::{ModelConfig, read_config},
    tensor::QuantizedTensor,
    utils::MemoryMapper,
};
use anyhow::{Context, Result};

mod qwen3;

/// Contains the main inference logic for the Transformer model.
pub trait Transformer {
    /// Runs forward pass for the Transformer model.
    fn forward(&mut self, token: usize, pos: usize) -> &[f32];

    fn get_config(&self) -> &ModelConfig;
}

#[non_exhaustive]
pub enum Transformers {
    Qwen3(qwen3::Qwen3Transformer),
}

impl Transformer for Transformers {
    fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        match self {
            Transformers::Qwen3(model) => model.forward(token, pos),
        }
    }

    fn get_config(&self) -> &ModelConfig {
        match self {
            Transformers::Qwen3(model) => model.get_config(),
        }
    }
}

/// Builder pattern for creating transformer models
pub struct TransformerBuilder {
    checkpoint_path: String,
    ctx_length: Option<usize>,
}

impl TransformerBuilder {
    pub fn new(checkpoint_path: &str) -> Self {
        Self { checkpoint_path: checkpoint_path.to_string(), ctx_length: None }
    }

    pub fn with_ctx_length(mut self, ctx_length: Option<usize>) -> Self {
        self.ctx_length = ctx_length;
        self
    }

    pub fn build(self) -> Result<Transformers> {
        let file = File::open(&self.checkpoint_path)
            .with_context(|| format!("Failed to open checkpoint: {}", self.checkpoint_path))?;

        let mut mapper = MemoryMapper::new(file)?;

        // Read config from the first part of the file
        let mut config = read_config(&mut mapper)?;

        // Apply context length override if provided
        if let Some(ctx_len) = self.ctx_length {
            config.seq_len = ctx_len.min(config.seq_len);
        }

        match config.architecture_id {
            1 => Ok(Transformers::Qwen3(qwen3::Qwen3Transformer::new(config, mapper)?)),
            x => anyhow::bail!("Unknown architecture_id: {x}"),
        }
    }
}

/// Reads multiple quantized tensors from memory mapper.
///
/// Each quantized tensor consists of:
/// 1. Quantized weights (i8 values)
/// 2. Scale factors (f32 values)
///
/// The scale factors are grouped according to the quantization group size.
pub(crate) fn create_quantized_tensors(
    mapper: &mut MemoryMapper,
    n_tensors: usize,
    size_each: usize,
    group_size: usize,
) -> Result<Vec<QuantizedTensor>> {
    (0..n_tensors)
        .map(|i| {
            // Read quantized values
            let q_bytes =
                mapper.get_bytes(size_each).with_context(|| format!("Failed to read quantized tensor {i} data"))?;

            // Convert bytes to i8 (avoiding copy by using unsafe)
            let q_slice = unsafe { std::slice::from_raw_parts(q_bytes.as_ptr() as *const i8, size_each) };

            // Calculate and read scale factors
            let s_len = size_each / group_size;
            let s_slice =
                mapper.get_f32_slice(s_len).with_context(|| format!("Failed to read scale factors for tensor {i}"))?;

            // Convert to 'static lifetime using unsafe transmute
            let q_static = unsafe { std::mem::transmute::<&[i8], &'static [i8]>(q_slice) };
            let s_static = unsafe { std::mem::transmute::<&[f32], &'static [f32]>(s_slice) };

            Ok(QuantizedTensor::from_slices(q_static, s_static))
        })
        .collect()
}
