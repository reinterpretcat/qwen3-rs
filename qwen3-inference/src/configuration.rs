use std::io::Cursor;

use crate::utils::MemoryMapper;
use anyhow::{Context, Error, Result};
use byteorder::{LittleEndian, ReadBytesExt};

/// Magic number for validating checkpoint files
const CHECKPOINT_MAGIC: i32 = 0x616a6331;
/// Expected checkpoint version
const CHECKPOINT_VERSION: i32 = 1;
/// Size of the checkpoint header in bytes
const HEADER_SIZE: usize = 256;
/// Size of config structure in bytes (12 i32 fields)
const CONFIG_SIZE: usize = 48;

/// Configuration struct for transformer models.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub group_size: usize,
    pub shared_classifier: bool,
}

/// Configuration struct for reading model parameters from checkpoint files.
#[derive(Debug, Clone, Copy)]
struct Config {
    pub magic_number: i32,
    pub version: i32,
    pub dim: i32,
    pub hidden_dim: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub vocab_size: i32,
    pub seq_len: i32,
    pub head_dim: i32,
    pub shared_classifier: i32,
    pub group_size: i32,
}

impl TryInto<ModelConfig> for Config {
    type Error = Error;

    fn try_into(self) -> Result<ModelConfig> {
        validate_config(&self).with_context(|| "Invalid model configuration")?;

        Ok(ModelConfig {
            dim: self.dim as usize,
            hidden_dim: self.hidden_dim as usize,
            n_layers: self.n_layers as usize,
            n_heads: self.n_heads as usize,
            n_kv_heads: self.n_kv_heads as usize,
            head_dim: self.head_dim as usize,
            seq_len: self.seq_len as usize,
            vocab_size: self.vocab_size as usize,
            group_size: self.group_size as usize,
            shared_classifier: self.shared_classifier != 0,
        })
    }
}

/// Reads and validates the model configuration from checkpoint data (mapper).
///
/// The configuration is stored as 12 consecutive i32 values in little-endian format.
/// This function performs bounds checking and validates the magic number and version.
pub fn read_config(mapper: &mut MemoryMapper) -> Result<ModelConfig> {
    let data = mapper.get_bytes(CONFIG_SIZE)?;

    if data.len() != CONFIG_SIZE {
        anyhow::bail!(
            "Insufficient data for config: need {} bytes, got {}",
            CONFIG_SIZE,
            data.len()
        );
    }

    let mut cursor = Cursor::new(data);

    // Use a macro to reduce repetitive error handling
    macro_rules! read_i32 {
        ($field:literal) => {
            cursor
                .read_i32::<LittleEndian>()
                .with_context(|| format!("Failed to read {}", $field))?
        };
    }

    let config = Config {
        magic_number: read_i32!("magic number"),
        version: read_i32!("version"),
        dim: read_i32!("dimension"),
        hidden_dim: read_i32!("hidden dimension"),
        n_layers: read_i32!("number of layers"),
        n_heads: read_i32!("number of heads"),
        n_kv_heads: read_i32!("number of KV heads"),
        vocab_size: read_i32!("vocabulary size"),
        seq_len: read_i32!("sequence length"),
        head_dim: read_i32!("head dimension"),
        shared_classifier: read_i32!("shared classifier flag"),
        group_size: read_i32!("group size"),
    };

    // prepare to load model weights (skip header).
    mapper.skip(HEADER_SIZE - CONFIG_SIZE)?;

    config.try_into()
}

/// Validates the model configuration to ensure it's supported.
fn validate_config(config: &Config) -> Result<()> {
    match config.magic_number {
        CHECKPOINT_MAGIC => {}
        actual => anyhow::bail!(
            "Invalid checkpoint magic number: expected {:#x}, got {:#x}",
            CHECKPOINT_MAGIC,
            actual
        ),
    }

    match config.version {
        CHECKPOINT_VERSION => {}
        actual => anyhow::bail!(
            "Unsupported checkpoint version: expected {}, got {}",
            CHECKPOINT_VERSION,
            actual
        ),
    }

    // Validate positive dimensions
    let dimensions = [
        ("dim", config.dim),
        ("n_layers", config.n_layers),
        ("n_heads", config.n_heads),
        ("n_kv_heads", config.n_kv_heads),
        ("vocab_size", config.vocab_size),
        ("seq_len", config.seq_len),
        ("head_dim", config.head_dim),
    ];

    for (name, value) in dimensions {
        if value <= 0 {
            anyhow::bail!("Invalid {}: must be positive, got {}", name, value);
        }
    }

    Ok(())
}
