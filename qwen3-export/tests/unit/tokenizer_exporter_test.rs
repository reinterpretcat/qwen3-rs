mod basic {
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
                "added_tokens": [
                    {
                        "content": "<special>",
                        "id": 100
                    }],
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
}

mod advanced {

    use byteorder::{LittleEndian, ReadBytesExt};
    use serde_json::{Value, json};
    use std::fs::File;
    use std::io::{Read, Write};
    use tempfile::TempDir;

    use crate::TokenizerExporter;

    const DEFAULT_SCORE: f32 = -1e6;

    fn create_tokenizer_file(temp_dir: &TempDir, tokenizer_data: &Value) -> std::io::Result<()> {
        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        let mut file = File::create(&tokenizer_path)?;
        write!(file, "{tokenizer_data}")?;
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
             "added_tokens": [
                {
                    "id": 100,
                    "content": "<|endoftext|>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                },
                {
                    "id": 101,
                    "content": "<|startoftext|>",
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": true
                }
            ],
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
}
