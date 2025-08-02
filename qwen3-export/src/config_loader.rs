#[cfg(test)]
#[path = "../tests/unit/config_loader_test.rs"]
mod config_loader_test;

use anyhow::{Context, Result};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, path::Path};

use crate::models::ArchitectureId;

/// Model type detection with embedded LoRA configuration
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Base,             // Standard base model
    LoRA(LoRAConfig), // LoRA fine-tuned model with full config
}

/// Enhanced model information that includes type and configs
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_type: ModelType,
    pub config: ModelConfig,
}

/// Configuration structure matching the Python ModelArgs
#[derive(Debug, Clone)]
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
    pub architecture: ArchitectureId,
}

/// LoRA configuration from adapter_config.json
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoRAConfig {
    pub lora_alpha: f32,
    pub r: usize,
    pub target_modules: Vec<String>,
    pub base_model_name_or_path: Option<String>,
}

/// Auto-detect model type and load appropriate configuration
/// This is the main entry point that replaces load_hf_config
pub fn load_model_info(model_path: &str) -> Result<ModelInfo> {
    let model_path = Path::new(model_path);

    // Detect model type based on config files
    let model_type = detect_model_type(model_path)?;

    let (config, _) = match &model_type {
        ModelType::Base => {
            info!("Detected Base model type (no LoRA)");
            let config = load_base_model_config(model_path)?;
            (config, ())
        }
        ModelType::LoRA(lora_config) => {
            let config = load_base_model_config(model_path)?;
            info!("Detected Base model type with LoRA configuration:");
            info!("   • Alpha: {}", lora_config.lora_alpha);
            info!("   • Rank (r): {}", lora_config.r);
            info!("   • Target modules: {:?}", lora_config.target_modules);
            if let Some(ref base_model) = lora_config.base_model_name_or_path {
                info!("   • Base model: {}", base_model);
            }
            info!("");
            (config, ())
        }
    };

    Ok(ModelInfo { model_type, config })
}

/// Detect model type based on presence of config files.
/// For LoRA models, loads and embeds the LoRA configuration.
fn detect_model_type(model_path: &Path) -> Result<ModelType> {
    let has_adapter_config = model_path.join("adapter_config.json").exists();
    let has_base_config = model_path.join("config.json").exists();

    match (has_base_config, has_adapter_config) {
        (true, true) => {
            // LoRA model - load adapter config and embed the full config
            let lora_config = load_lora_config(model_path)?;
            Ok(ModelType::LoRA(lora_config))
        }
        (true, false) => Ok(ModelType::Base),
        (false, true) => anyhow::bail!(
            "Only LoRA config is found in {}. Make sure to have base model files in the same directory",
            model_path.display()
        ),
        _ => anyhow::bail!(
            "No valid configuration files found in {}",
            model_path.display()
        ),
    }
}

/// Load base model configuration - handles both direct config.json and LoRA case
fn load_base_model_config(model_path: &Path) -> Result<ModelConfig> {
    let config_path = model_path.join("config.json");

    if config_path.exists() {
        // Direct config.json exists
        load_hf_config(&config_path)
    } else {
        // For LoRA models, we might need to look elsewhere or use defaults
        // For now, return an error to let user know they need base model config
        anyhow::bail!(
            "Base model config.json not found in {}. For LoRA models, ensure the base model config is available.",
            model_path.display()
        )
    }
}

/// Load model configuration from HuggingFace format.
fn load_hf_config(config_path: &Path) -> Result<ModelConfig> {
    let mut file = File::open(&config_path)
        .with_context(|| format!("Failed to open config.json at {config_path:?}"))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    #[derive(Debug, Deserialize)]
    struct HFConfig {
        hidden_size: u32,
        intermediate_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        vocab_size: u32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        #[serde(default)]
        head_dim: Option<u32>,
        #[serde(default)]
        bos_token_id: Option<u32>,
        #[serde(default)]
        eos_token_id: Option<u32>,
        #[serde(default)]
        architectures: Option<Vec<String>>,
    }

    let hf_config: HFConfig = serde_json::from_str(&contents)
        .map_err(|err| anyhow::anyhow!("Failed to parse config.json: {}", err))?;

    let head_dim = hf_config
        .head_dim
        .unwrap_or(hf_config.hidden_size / hf_config.num_attention_heads);

    // Try to determine architecture
    let architectures = hf_config.architectures.as_ref();
    let architecture = match (architectures, architectures.and_then(|a| a.first())) {
        (Some(architectures), Some(first)) if architectures.len() == 1 => {
            ArchitectureId::try_from(first.as_str())?
        }
        (Some(architectures), _) => {
            anyhow::bail!("Multiple architectures are not supported: {architectures:?}")
        }
        _ => anyhow::bail!("Cannot determine architecture"),
    };

    let config = ModelConfig {
        dim: hf_config.hidden_size,
        hidden_dim: hf_config.intermediate_size,
        n_layers: hf_config.num_hidden_layers,
        n_heads: hf_config.num_attention_heads,
        n_kv_heads: hf_config.num_key_value_heads,
        vocab_size: hf_config.vocab_size,
        max_seq_len: hf_config.max_position_embeddings,
        norm_eps: hf_config.rms_norm_eps,
        head_dim,
        bos_token_id: hf_config.bos_token_id.unwrap_or(0),
        eos_token_id: hf_config.eos_token_id.unwrap_or(0),
        architecture,
    };

    info!("Model configuration loaded:");
    info!("   • Architecture: {:?}", config.architecture);
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

/// Load LoRA configuration from adapter_config.json
fn load_lora_config(model_path: &Path) -> Result<LoRAConfig> {
    let config_path = model_path.join("adapter_config.json");
    let mut file = File::open(&config_path).with_context(|| {
        format!(
            "Failed to open adapter_config.json at {}",
            config_path.display()
        )
    })?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let config: LoRAConfig = serde_json::from_str(&contents)
        .map_err(|err| anyhow::anyhow!("Failed to parse adapter_config.json: {}", err))?;

    Ok(config)
}
