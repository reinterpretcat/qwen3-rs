use anyhow::{Context, Result};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs::File, io::Read, path::Path};

/// Configuration structure matching the Python ModelArgs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dim: u32,
    pub hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub head_dim: u32,
    pub norm_eps: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

/// Load model configuration from HuggingFace format.
pub fn load_hf_config(model_path: &str) -> Result<ModelConfig> {
    let config_path = Path::new(model_path).join("config.json");
    let mut file = File::open(&config_path)
        .with_context(|| format!("Failed to open config.json at {config_path:?}"))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let config: Value =
        serde_json::from_str(&contents).with_context(|| "Failed to parse config.json")?;

    let get_required_i64 = |field: &str| -> Result<i64> {
        config
            .get(field)
            .and_then(|v| v.as_i64())
            .with_context(|| format!("Missing or invalid field '{field}' in config.json"))
    };

    let get_required_f64 = |field: &str| -> Result<f64> {
        config
            .get(field)
            .and_then(|v| v.as_f64())
            .with_context(|| format!("Missing or invalid field '{field}' in config.json"))
    };

    let get_optional_u64 = |field: &str, default: u64| -> u64 {
        config
            .get(field)
            .and_then(|v| v.as_u64())
            .unwrap_or(default)
    };

    let dim = get_required_i64("hidden_size")? as u32;
    let n_heads = get_required_i64("num_attention_heads")? as u32;

    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_i64())
        .unwrap_or(dim as i64 / n_heads as i64) as u32;

    let config = ModelConfig {
        dim,
        hidden_dim: get_required_i64("intermediate_size")? as u32,
        n_layers: get_required_i64("num_hidden_layers")? as u32,
        n_heads,
        n_kv_heads: get_required_i64("num_key_value_heads")? as u32,
        vocab_size: get_required_i64("vocab_size")? as u32,
        max_seq_len: get_required_i64("max_position_embeddings")? as u32,
        head_dim,
        norm_eps: get_required_f64("rms_norm_eps")? as f32,
        bos_token_id: get_optional_u64("bos_token_id", 0) as u32,
        eos_token_id: get_optional_u64("eos_token_id", 0) as u32,
    };

    info!("Model configuration loaded:");
    info!("   • Dimensions: {}", config.dim);
    info!("   • Layers: {}", config.n_layers);
    info!("   • Attention heads: {}", config.n_heads);
    info!("   • KV heads: {}", config.n_kv_heads);
    info!("   • Vocabulary size: {}", config.vocab_size);
    info!("   • Max sequence length: {}", config.max_seq_len);
    info!("   • Head dimension: {}", config.head_dim);
    info!("");

    Ok(config)
}
