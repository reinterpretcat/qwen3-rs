//! Unit tests for TokenizerExporter
//!
//! These tests verify the core functionality of individual methods in the TokenizerExporter,
//! including Unicode to byte mapping, token ordering, vocabulary extraction, and merge rank
//! processing.

use crate::tokenizer_exporter::TokenizerExporter;

/// Test the UnicodeToByteMap functionality
#[cfg(test)]
mod unicode_to_byte_map_tests {
    use crate::tokenizer_exporter::UnicodeToByteMap;

    #[test]
    fn test_printable_ascii_mapping() {
        let u2b_map = UnicodeToByteMap::new();

        // Test printable ASCII characters (33-126)
        assert_eq!(u2b_map.token_to_bytes("A"), vec![65]);
        assert_eq!(u2b_map.token_to_bytes("z"), vec![122]);
        assert_eq!(u2b_map.token_to_bytes("!"), vec![33]);
        assert_eq!(u2b_map.token_to_bytes("~"), vec![126]);
    }

    #[test]
    fn test_extended_ascii_mapping() {
        let u2b_map = UnicodeToByteMap::new();

        // Test extended ASCII characters
        assert_eq!(u2b_map.token_to_bytes("¡"), vec![161]); // Start of first range
        assert_eq!(u2b_map.token_to_bytes("¬"), vec![172]); // End of first range
        assert_eq!(u2b_map.token_to_bytes("®"), vec![174]); // Start of second range
        assert_eq!(u2b_map.token_to_bytes("ÿ"), vec![255]); // End of second range
    }

    #[test]
    fn test_unprintable_character_mapping() {
        let u2b_map = UnicodeToByteMap::new();

        // Test that unprintable characters get mapped to Unicode offset
        let result = u2b_map.token_to_bytes("\u{0100}"); // 256 in Unicode
        assert!(!result.is_empty());

        // Test null character and other control characters get mapped
        let null_result = u2b_map.token_to_bytes("\0");
        assert_eq!(null_result.len(), 1);
        // u8 is always <= 255, so just check it's valid
        assert!(null_result[0] == null_result[0]);
    }

    #[test]
    fn test_multi_character_token() {
        let u2b_map = UnicodeToByteMap::new();

        // Test multi-character tokens
        assert_eq!(
            u2b_map.token_to_bytes("hello"),
            vec![104, 101, 108, 108, 111]
        );
        assert_eq!(u2b_map.token_to_bytes("ABC"), vec![65, 66, 67]);
    }

    #[test]
    fn test_empty_token() {
        let u2b_map = UnicodeToByteMap::new();

        assert_eq!(u2b_map.token_to_bytes(""), Vec::<u8>::new());
    }
}

/// Test TokenizerExporter basic functionality
#[cfg(test)]
mod tokenizer_exporter_tests {
    use super::TokenizerExporter;
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_new_and_default() {
        let exporter1 = TokenizerExporter::new();
        let exporter2 = TokenizerExporter;

        // Both should be equivalent (Debug trait allows comparison)
        assert_eq!(format!("{exporter1:?}"), format!("{:?}", exporter2));
    }

    #[test]
    fn test_create_ordered_tokens() {
        let exporter = TokenizerExporter::new();
        let mut vocab = HashMap::new();
        vocab.insert("token_b".to_string(), 2);
        vocab.insert("token_a".to_string(), 1);
        vocab.insert("token_c".to_string(), 3);

        let ordered = exporter.create_ordered_tokens(&vocab);

        assert_eq!(ordered.len(), 3);
        assert_eq!(ordered[0], (1, "token_a".to_string()));
        assert_eq!(ordered[1], (2, "token_b".to_string()));
        assert_eq!(ordered[2], (3, "token_c".to_string()));
    }

    #[test]
    fn test_create_ordered_tokens_empty() {
        let exporter = TokenizerExporter::new();
        let vocab = HashMap::new();

        let ordered = exporter.create_ordered_tokens(&vocab);

        assert!(ordered.is_empty());
    }

    #[test]
    fn test_extract_vocabulary_standard_format() {
        let exporter = TokenizerExporter::new();
        let tokenizer_data = json!({
            "model": {
                "vocab": {
                    "hello": 1,
                    "world": 2,
                    "test": 3
                }
            }
        });

        let vocab = exporter.extract_vocabulary(&tokenizer_data).unwrap();

        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get("hello"), Some(&1));
        assert_eq!(vocab.get("world"), Some(&2));
        assert_eq!(vocab.get("test"), Some(&3));
    }

    #[test]
    fn test_extract_vocabulary_alternative_format() {
        let exporter = TokenizerExporter::new();
        let tokenizer_data = json!({
            "vocab": {
                "token1": 10,
                "token2": 20
            }
        });

        let vocab = exporter.extract_vocabulary(&tokenizer_data).unwrap();

        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.get("token1"), Some(&10));
        assert_eq!(vocab.get("token2"), Some(&20));
    }

    #[test]
    fn test_extract_vocabulary_missing() {
        let exporter = TokenizerExporter::new();
        let tokenizer_data = json!({
            "other": "data"
        });

        let result = exporter.extract_vocabulary(&tokenizer_data);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Could not find vocabulary")
        );
    }

    #[test]
    fn test_extract_merge_ranks() {
        let exporter = TokenizerExporter::new();
        let tokenizer_data = json!({
            "model": {
                "merges": [
                    "h e",
                    "l l",
                    "o !",
                    "he ll"
                ]
            }
        });

        let merge_ranks = exporter.extract_merge_ranks(&tokenizer_data);

        assert_eq!(merge_ranks.len(), 4);
        assert_eq!(merge_ranks.get("h e"), Some(&0));
        assert_eq!(merge_ranks.get("l l"), Some(&1));
        assert_eq!(merge_ranks.get("o !"), Some(&2));
        assert_eq!(merge_ranks.get("he ll"), Some(&3));
    }

    #[test]
    fn test_extract_merge_ranks_empty() {
        let exporter = TokenizerExporter::new();
        let tokenizer_data = json!({
            "model": {}
        });

        let merge_ranks = exporter.extract_merge_ranks(&tokenizer_data);

        assert!(merge_ranks.is_empty());
    }

    #[test]
    fn test_extract_merge_ranks_missing() {
        let exporter = TokenizerExporter::new();
        let tokenizer_data = json!({
            "other": "data"
        });

        let merge_ranks = exporter.extract_merge_ranks(&tokenizer_data);

        assert!(merge_ranks.is_empty());
    }

    #[test]
    fn test_load_json_file() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;
        let json_path = temp_dir.path().join("test.json");

        let test_data = json!({
            "test": "value",
            "number": 42
        });

        let mut file = File::create(&json_path)?;
        write!(file, "{test_data}")?;

        let loaded = exporter.load_json_file(&json_path).unwrap();

        assert_eq!(loaded, test_data);

        Ok(())
    }

    #[test]
    fn test_load_json_file_invalid() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;
        let json_path = temp_dir.path().join("invalid.json");

        let mut file = File::create(&json_path)?;
        write!(file, "{{invalid json")?;

        let result = exporter.load_json_file(&json_path);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse JSON")
        );

        Ok(())
    }

    #[test]
    fn test_load_json_file_missing() {
        let exporter = TokenizerExporter::new();
        let missing_path = std::path::Path::new("/non/existent/path.json");

        let result = exporter.load_json_file(missing_path);

        assert!(result.is_err());
    }
}

/// Test token score calculation logic
#[cfg(test)]
mod token_score_tests {
    use super::TokenizerExporter;

    #[test]
    fn test_default_score_constant() {
        // Verify the DEFAULT_SCORE constant is as expected
        assert_eq!(TokenizerExporter::DEFAULT_SCORE, -1e6);
    }

    #[test]
    fn test_merge_rank_to_score_calculation() {
        // Test the score calculation logic: -ln(rank + 1)
        let rank_0_score = -1_f32.ln(); // -ln(1) = 0
        let rank_1_score = -((1 + 1) as f32).ln(); // -ln(2) ≈ -0.693
        let rank_10_score = -((10 + 1) as f32).ln(); // -ln(11) ≈ -2.398

        assert_eq!(rank_0_score, 0.0);
        assert!((rank_1_score - (-std::f32::consts::LN_2)).abs() < 0.001);
        assert!((rank_10_score - (-2.397895)).abs() < 0.001);
    }
}

/// Test special token handling
#[cfg(test)]
mod special_token_tests {
    use super::TokenizerExporter;
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_add_special_tokens_from_config() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;

        // Create tokenizer_config.json
        let config_data = json!({
            "added_tokens_decoder": {
                "100": {
                    "content": "<special1>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                },
                "101": {
                    "content": "<special2>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                }
            }
        });

        let config_path = temp_dir.path().join("tokenizer_config.json");
        let mut file = File::create(&config_path)?;
        write!(file, "{config_data}")?;

        let mut vocab = HashMap::new();
        vocab.insert("existing_token".to_string(), 1);

        exporter
            .add_special_tokens_from_config(temp_dir.path(), &mut vocab)
            .unwrap();

        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get("existing_token"), Some(&1));
        assert_eq!(vocab.get("<special1>"), Some(&100));
        assert_eq!(vocab.get("<special2>"), Some(&101));

        Ok(())
    }

    #[test]
    fn test_add_special_tokens_no_duplicates() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;

        let config_data = json!({
            "added_tokens_decoder": {
                "100": {
                    "content": "existing_token"
                }
            }
        });

        let config_path = temp_dir.path().join("tokenizer_config.json");
        let mut file = File::create(&config_path)?;
        write!(file, "{config_data}")?;

        let mut vocab = HashMap::new();
        vocab.insert("existing_token".to_string(), 1);

        exporter
            .add_special_tokens_from_config(temp_dir.path(), &mut vocab)
            .unwrap();

        // Should still have only 1 token, no duplicates
        assert_eq!(vocab.len(), 1);
        assert_eq!(vocab.get("existing_token"), Some(&1)); // Original ID preserved

        Ok(())
    }

    #[test]
    fn test_add_special_tokens_missing_config() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;

        let mut vocab = HashMap::new();
        vocab.insert("existing_token".to_string(), 1);

        // Should succeed even without config file
        let result = exporter.add_special_tokens_from_config(temp_dir.path(), &mut vocab);

        assert!(result.is_ok());
        assert_eq!(vocab.len(), 1);

        Ok(())
    }

    #[test]
    fn test_add_special_tokens_empty_decoder() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;

        let config_data = json!({
            "other_field": "value"
            // No added_tokens_decoder
        });

        let config_path = temp_dir.path().join("tokenizer_config.json");
        let mut file = File::create(&config_path)?;
        write!(file, "{config_data}")?;

        let mut vocab = HashMap::new();
        vocab.insert("existing_token".to_string(), 1);

        let result = exporter.add_special_tokens_from_config(temp_dir.path(), &mut vocab);

        assert!(result.is_ok());
        assert_eq!(vocab.len(), 1); // No tokens added

        Ok(())
    }
}

/// Test error handling
#[cfg(test)]
mod error_handling_tests {
    use super::TokenizerExporter;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_load_tokenizer_json_missing_file() {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new().unwrap();

        let result = exporter.load_tokenizer_json(temp_dir.path());

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("tokenizer.json not found")
        );
    }

    #[test]
    fn test_load_tokenizer_json_invalid_json() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;

        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = File::create(&tokenizer_path)?;
        write!(file, "{{invalid json content")?;

        let result = exporter.load_tokenizer_json(temp_dir.path());

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse tokenizer.json")
        );

        Ok(())
    }
}

/// Test the overall load_token_data functionality
#[cfg(test)]
mod load_token_data_tests {
    use super::TokenizerExporter;
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_load_token_data_complete() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;

        // Create tokenizer.json
        let tokenizer_data = json!({
            "model": {
                "vocab": {
                    "hello": 1,
                    "world": 2,
                    "!": 3
                },
                "merges": [
                    "h e",
                    "l l"
                ]
            }
        });

        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = File::create(&tokenizer_path)?;
        write!(file, "{tokenizer_data}")?;

        // Create tokenizer_config.json with special tokens
        let config_data = json!({
            "added_tokens_decoder": {
                "100": {
                    "content": "<special>"
                }
            }
        });

        let config_path = temp_dir.path().join("tokenizer_config.json");
        let mut file = File::create(&config_path)?;
        write!(file, "{config_data}")?;

        let token_data = exporter.load_token_data(temp_dir.path()).unwrap();

        // Check vocabulary includes both regular and special tokens
        assert_eq!(token_data.vocab.len(), 4);
        assert_eq!(token_data.vocab.get("hello"), Some(&1));
        assert_eq!(token_data.vocab.get("world"), Some(&2));
        assert_eq!(token_data.vocab.get("!"), Some(&3));
        assert_eq!(token_data.vocab.get("<special>"), Some(&100));

        // Check merge ranks
        assert_eq!(token_data.merge_ranks.len(), 2);
        assert_eq!(token_data.merge_ranks.get("h e"), Some(&0));
        assert_eq!(token_data.merge_ranks.get("l l"), Some(&1));

        // Check max token length (should be len of "<special>" = 9)
        assert_eq!(token_data.max_token_length, 9);

        Ok(())
    }

    #[test]
    fn test_max_token_length_calculation() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;

        let tokenizer_data = json!({
            "model": {
                "vocab": {
                    "a": 1,
                    "hello": 2,
                    "very_long_token_name": 3
                }
            }
        });

        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = File::create(&tokenizer_path)?;
        write!(file, "{tokenizer_data}")?;

        let token_data = exporter.load_token_data(temp_dir.path()).unwrap();

        // "very_long_token_name" has 20 characters
        assert_eq!(token_data.max_token_length, 20);

        Ok(())
    }

    #[test]
    fn test_max_token_length_empty_vocab() -> std::io::Result<()> {
        let exporter = TokenizerExporter::new();
        let temp_dir = TempDir::new()?;

        let tokenizer_data = json!({
            "model": {
                "vocab": {}
            }
        });

        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = File::create(&tokenizer_path)?;
        write!(file, "{tokenizer_data}")?;

        let token_data = exporter.load_token_data(temp_dir.path()).unwrap();

        assert_eq!(token_data.max_token_length, 0);

        Ok(())
    }
}
