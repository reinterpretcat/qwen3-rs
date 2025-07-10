//! Integration tests for config loader functionality

use anyhow::Result;
use qwen3_export::{ModelConfig, load_hf_config};
use std::fs;
use tempfile::TempDir;

/// Helper to create a minimal config.json for testing
fn create_test_config_json(temp_dir: &TempDir) -> Result<()> {
    let config_content = r#"{
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "head_dim": 32,
        "bos_token_id": 1,
        "eos_token_id": 2
    }"#;

    let config_path = temp_dir.path().join("config.json");
    fs::write(config_path, config_content)?;
    Ok(())
}

#[test]
fn test_load_hf_config_valid() -> Result<()> {
    let temp_dir = TempDir::new()?;
    create_test_config_json(&temp_dir)?;

    let config = load_hf_config(temp_dir.path().to_str().unwrap())?;

    // Verify all fields are loaded correctly
    assert_eq!(config.dim, 256);
    assert_eq!(config.hidden_dim, 1024);
    assert_eq!(config.n_layers, 4);
    assert_eq!(config.n_heads, 8);
    assert_eq!(config.n_kv_heads, 8);
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.max_seq_len, 512);
    assert_eq!(config.head_dim, 32);
    assert!((config.norm_eps - 1e-6).abs() < 1e-9);
    assert_eq!(config.bos_token_id, 1);
    assert_eq!(config.eos_token_id, 2);

    Ok(())
}

#[test]
fn test_load_hf_config_missing_file() {
    let temp_dir = TempDir::new().unwrap();
    let result = load_hf_config(temp_dir.path().to_str().unwrap());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Failed to open config.json")
    );
}

#[test]
fn test_load_hf_config_invalid_json() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config_path = temp_dir.path().join("config.json");
    fs::write(config_path, "invalid json")?;

    let result = load_hf_config(temp_dir.path().to_str().unwrap());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Failed to parse config.json")
    );

    Ok(())
}

#[test]
fn test_load_hf_config_missing_required_field() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Config missing required "hidden_size" field
    let config_content = r#"{
        "intermediate_size": 1024,
        "num_hidden_layers": 4
    }"#;

    let config_path = temp_dir.path().join("config.json");
    fs::write(config_path, config_content)?;

    let result = load_hf_config(temp_dir.path().to_str().unwrap());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Missing or invalid field 'hidden_size'")
    );

    Ok(())
}

#[test]
fn test_load_hf_config_with_defaults() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Config without optional fields (bos_token_id, eos_token_id, head_dim)
    let config_content = r#"{
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6
    }"#;

    let config_path = temp_dir.path().join("config.json");
    fs::write(config_path, config_content)?;

    let config = load_hf_config(temp_dir.path().to_str().unwrap())?;

    // Check defaults are applied
    assert_eq!(config.bos_token_id, 0); // default
    assert_eq!(config.eos_token_id, 0); // default
    assert_eq!(config.head_dim, 256 / 8); // calculated: dim / n_heads

    Ok(())
}

#[test]
fn test_config_serialization() -> Result<()> {
    let config = ModelConfig {
        dim: 128,
        hidden_dim: 512,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 500,
        max_seq_len: 256,
        head_dim: 32,
        norm_eps: 1e-5,
        bos_token_id: 1,
        eos_token_id: 2,
    };

    // Test that config can be serialized and deserialized
    let json = serde_json::to_string(&config)?;
    let deserialized: ModelConfig = serde_json::from_str(&json)?;

    // Verify all fields match
    assert_eq!(config.dim, deserialized.dim);
    assert_eq!(config.hidden_dim, deserialized.hidden_dim);
    assert_eq!(config.n_layers, deserialized.n_layers);
    assert_eq!(config.n_heads, deserialized.n_heads);
    assert_eq!(config.n_kv_heads, deserialized.n_kv_heads);
    assert_eq!(config.vocab_size, deserialized.vocab_size);
    assert_eq!(config.max_seq_len, deserialized.max_seq_len);
    assert_eq!(config.head_dim, deserialized.head_dim);
    assert!((config.norm_eps - deserialized.norm_eps).abs() < 1e-9);
    assert_eq!(config.bos_token_id, deserialized.bos_token_id);
    assert_eq!(config.eos_token_id, deserialized.eos_token_id);

    Ok(())
}
