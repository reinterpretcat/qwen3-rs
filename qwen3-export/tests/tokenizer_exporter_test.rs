//! Integration tests for tokenizer exporter functionality

use byteorder::{LittleEndian, ReadBytesExt};
use qwen3_export::tokenizer_exporter::TokenizerExporter;
use serde_json::{Value, json};
use std::fs::File;
use std::io::{Read, Write};
use tempfile::TempDir;

const DEFAULT_SCORE: f32 = -1e6;

fn create_tokenizer_file(temp_dir: &TempDir, tokenizer_data: &Value) -> std::io::Result<()> {
    let tokenizer_path = temp_dir.path().join("tokenizer.json");
    let mut file = File::create(&tokenizer_path)?;
    write!(file, "{tokenizer_data}")?;
    Ok(())
}

fn create_config_file(temp_dir: &TempDir, config_data: &Value) -> std::io::Result<()> {
    let config_path = temp_dir.path().join("tokenizer_config.json");
    let mut file = File::create(&config_path)?;
    write!(file, "{config_data}")?;
    Ok(())
}

/// Test complete tokenizer export pipeline
#[test]
fn test_complete_tokenizer_export() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    // Create a complete tokenizer.json with vocabulary and merges
    // Include tokens that actually match the merge patterns to test score calculation
    let tokenizer_data = json!({
        "model": {
            "vocab": {
                "hello": 1,
                "world": 2,
                "!": 3,
                "h e": 4,      // This token matches merge[0], should get score -ln(1) = 0
                "l l": 5,      // This token matches merge[1], should get score -ln(2) ≈ -0.693
                "other": 6     // This token has no merge, should get DEFAULT_SCORE
            },
            "merges": [
                "h e",         // rank 0
                "l l"          // rank 1
            ]
        }
    });

    create_tokenizer_file(&temp_dir, &tokenizer_data)?;

    // Create tokenizer_config.json with special tokens
    let config_data = json!({
        "added_tokens_decoder": {
            "100": {
                "content": "<|endoftext|>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false,
                "special": true
            },
            "101": {
                "content": "<|startoftext|>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false,
                "special": true
            }
        }
    });

    create_config_file(&temp_dir, &config_data)?;

    // Export tokenizer
    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(
        temp_dir.path(),
        &output_path,
        100, // bos_token_id
        101, // eos_token_id
    );

    assert!(result.is_ok());

    // Verify output file exists
    let tokenizer_output = output_path.with_extension("tokenizer");
    assert!(tokenizer_output.exists());

    // Verify binary format
    let mut file = File::open(&tokenizer_output)?;

    // Read header
    let max_token_length = file.read_u32::<LittleEndian>()?;
    let bos_token_id = file.read_u32::<LittleEndian>()?;
    let eos_token_id = file.read_u32::<LittleEndian>()?;

    // Check header values
    assert_eq!(max_token_length, 15); // Length of "<|startoftext|>" is 15
    assert_eq!(bos_token_id, 100);
    assert_eq!(eos_token_id, 101);

    // Read and verify tokens (should be ordered by ID)
    let expected_tokens = vec![
        (1, "hello", DEFAULT_SCORE),             // No merge rank
        (2, "world", DEFAULT_SCORE),             // No merge rank
        (3, "!", DEFAULT_SCORE),                 // No merge rank
        (4, "h e", 0.0),                         // Merge rank 0: -ln(1) = 0
        (5, "l l", -((1 + 1) as f32).ln()),      // Merge rank 1: -ln(2) ≈ -0.693
        (6, "other", DEFAULT_SCORE),             // No merge rank
        (100, "<|endoftext|>", DEFAULT_SCORE),   // Special token, no merge rank
        (101, "<|startoftext|>", DEFAULT_SCORE), // Special token, no merge rank
    ];

    for (_expected_id, expected_token, expected_score) in expected_tokens {
        let score = file.read_f32::<LittleEndian>()?;
        let token_length = file.read_u32::<LittleEndian>()? as usize;

        let mut token_bytes = vec![0u8; token_length];
        file.read_exact(&mut token_bytes)?;

        // Verify the token can be converted back to string
        let token_str = String::from_utf8_lossy(&token_bytes);

        // Check score matches expected value
        if (expected_score - score).abs() < 0.001 {
            // Score matches (within floating point precision)
            assert!((expected_score - score).abs() < 0.001);
        } else {
            panic!(
                "Score mismatch for token '{expected_token}': expected {expected_score}, got {score}",
            );
        }

        // Token should match expected (via Unicode mapping)
        if expected_token == "!" {
            assert_eq!(token_bytes, vec![33]); // ASCII for '!'
        } else if expected_token.is_ascii() {
            // ASCII tokens should match exactly
            assert_eq!(token_str, expected_token);
        }

        assert_eq!(token_bytes.len(), expected_token.len());
    }

    Ok(())
}

/// Test binary format with empty vocabulary
#[test]
fn test_export_empty_vocabulary() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    let tokenizer_data = json!({
        "model": {
            "vocab": {}
        }
    });

    create_tokenizer_file(&temp_dir, &tokenizer_data)?;

    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 0);

    assert!(result.is_ok());

    // Verify output file exists and has correct header
    let tokenizer_output = output_path.with_extension("tokenizer");
    assert!(tokenizer_output.exists());

    let mut file = File::open(&tokenizer_output)?;
    let max_token_length = file.read_u32::<LittleEndian>()?;
    let bos_token_id = file.read_u32::<LittleEndian>()?;
    let eos_token_id = file.read_u32::<LittleEndian>()?;

    assert_eq!(max_token_length, 0);
    assert_eq!(bos_token_id, 0);
    assert_eq!(eos_token_id, 0);

    // Should be no more data (empty vocabulary)
    let mut buffer = [0u8; 1];
    assert_eq!(file.read(&mut buffer)?, 0);

    Ok(())
}

/// Test Unicode character handling in binary format
#[test]
fn test_unicode_character_export() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    let tokenizer_data = json!({
        "model": {
            "vocab": {
                "café": 1,
                "naïve": 2,
                "🚀": 3,
                "你好": 4
            }
        }
    });

    create_tokenizer_file(&temp_dir, &tokenizer_data)?;

    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 0);

    assert!(result.is_ok());

    // Verify the binary file can be read correctly
    let tokenizer_output = output_path.with_extension("tokenizer");
    let mut file = File::open(&tokenizer_output)?;

    // Skip header
    file.read_u32::<LittleEndian>()?; // max_token_length
    file.read_u32::<LittleEndian>()?; // bos_token_id
    file.read_u32::<LittleEndian>()?; // eos_token_id

    // Read each token and verify it's properly encoded
    for _ in 0..4 {
        let _score = file.read_f32::<LittleEndian>()?;
        let token_length = file.read_u32::<LittleEndian>()? as usize;

        let mut token_bytes = vec![0u8; token_length];
        file.read_exact(&mut token_bytes)?;

        // Should be able to read the bytes without error
        assert!(!token_bytes.is_empty());
    }

    Ok(())
}

/// Test error handling for missing files
#[test]
fn test_export_missing_tokenizer_file() {
    let temp_dir = TempDir::new().unwrap();
    let exporter = TokenizerExporter::new();

    // Don't create tokenizer.json
    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 0);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("tokenizer.json not found")
    );
}

/// Test error handling for invalid JSON
#[test]
fn test_export_invalid_tokenizer_json() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    // Create invalid JSON file
    let tokenizer_path = temp_dir.path().join("tokenizer.json");
    let mut file = File::create(&tokenizer_path)?;
    write!(file, "{{invalid json content")?;

    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 0);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Failed to parse tokenizer.json")
    );

    Ok(())
}

/// Test error handling for missing vocabulary
#[test]
fn test_export_missing_vocabulary() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    let tokenizer_data = json!({
        "model": {
            "other_field": "value"
            // No vocab field
        }
    });

    create_tokenizer_file(&temp_dir, &tokenizer_data)?;

    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 0);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Could not find vocabulary")
    );

    Ok(())
}

/// Test alternative vocabulary format support
#[test]
fn test_alternative_vocab_format() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    // Use direct vocab format instead of model.vocab
    let tokenizer_data = json!({
        "vocab": {
            "token1": 1,
            "token2": 2
        }
    });

    create_tokenizer_file(&temp_dir, &tokenizer_data)?;

    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 0);

    assert!(result.is_ok());

    // Verify the tokens were processed correctly
    let tokenizer_output = output_path.with_extension("tokenizer");
    assert!(tokenizer_output.exists());

    let mut file = File::open(&tokenizer_output)?;

    // Skip header
    file.read_u32::<LittleEndian>()?;
    file.read_u32::<LittleEndian>()?;
    file.read_u32::<LittleEndian>()?;

    // Should have 2 tokens
    for _ in 0..2 {
        let _score = file.read_f32::<LittleEndian>()?;
        let token_length = file.read_u32::<LittleEndian>()? as usize;
        let mut token_bytes = vec![0u8; token_length];
        file.read_exact(&mut token_bytes)?;
        assert!(!token_bytes.is_empty());
    }

    Ok(())
}

/// Test merge ranks processing and score calculation
#[test]
fn test_merge_ranks_and_scores() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    let tokenizer_data = json!({
        "model": {
            "vocab": {
                "hello": 1,
                "world": 2,
                "unknown": 3
            },
            "merges": [
                "h e",  // rank 0, score = -ln(1) = 0
                "l l"   // rank 1, score = -ln(2) ≈ -0.693
            ]
        }
    });

    create_tokenizer_file(&temp_dir, &tokenizer_data)?;

    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 0);

    assert!(result.is_ok());

    let tokenizer_output = output_path.with_extension("tokenizer");
    let mut file = File::open(&tokenizer_output)?;

    // Skip header
    file.read_u32::<LittleEndian>()?;
    file.read_u32::<LittleEndian>()?;
    file.read_u32::<LittleEndian>()?;

    // Read tokens and check scores
    let mut scores = Vec::new();
    for _ in 0..3 {
        let score = file.read_f32::<LittleEndian>()?;
        scores.push(score);

        let token_length = file.read_u32::<LittleEndian>()? as usize;
        let mut token_bytes = vec![0u8; token_length];
        file.read_exact(&mut token_bytes)?;
    }

    // All tokens should have the default score since none of them appear in merges
    // (merges contain "h e" and "l l", not "hello", "world", "unknown")
    for score in scores {
        assert_eq!(score, DEFAULT_SCORE);
    }

    Ok(())
}

/// Test large vocabulary handling
#[test]
fn test_large_vocabulary() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    // Create a vocabulary with 1000 tokens
    let mut vocab = serde_json::Map::new();
    for i in 0..1000 {
        vocab.insert(format!("token_{i}"), json!(i));
    }

    let tokenizer_data = json!({
        "model": {
            "vocab": vocab
        }
    });

    create_tokenizer_file(&temp_dir, &tokenizer_data)?;

    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 999);

    assert!(result.is_ok());

    // Verify the output file exists and has reasonable size
    let tokenizer_output = output_path.with_extension("tokenizer");
    assert!(tokenizer_output.exists());

    let metadata = std::fs::metadata(&tokenizer_output)?;
    // Should be at least header (12 bytes) + 1000 tokens * (4 + 4 + token_length bytes)
    assert!(metadata.len() > 12 + 1000 * 8);

    Ok(())
}

/// Test special token handling edge cases
#[test]
fn test_special_token_edge_cases() -> std::io::Result<()> {
    let temp_dir = TempDir::new()?;
    let exporter = TokenizerExporter::new();

    let tokenizer_data = json!({
        "model": {
            "vocab": {
                "regular": 1
            }
        }
    });

    create_tokenizer_file(&temp_dir, &tokenizer_data)?;

    // Create config with malformed special tokens
    let config_data = json!({
        "added_tokens_decoder": {
            "not_a_number": {
                "content": "should_be_ignored"
            },
            "200": {
                // Missing content field
                "special": true
            },
            "201": {
                "content": "valid_special"
            }
        }
    });

    create_config_file(&temp_dir, &config_data)?;

    let output_path = temp_dir.path().join("output");
    let result = exporter.export_tokenizer(temp_dir.path(), &output_path, 0, 0);

    // Should succeed despite malformed entries
    assert!(result.is_ok());

    // Verify only valid tokens are included
    let tokenizer_output = output_path.with_extension("tokenizer");
    let mut file = File::open(&tokenizer_output)?;

    let max_token_length = file.read_u32::<LittleEndian>()?;

    // max_token_length should be 13 (length of "valid_special")
    assert_eq!(max_token_length, 13);

    Ok(())
}
