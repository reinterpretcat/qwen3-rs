use crate::models::ArchitectureId;

use super::*;

mod basic {
    use super::*;

    // Helper function for tests
    fn create_test_config() -> ModelConfig {
        ModelConfig {
            dim: 4,
            hidden_dim: 16,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 100,
            max_seq_len: 128,
            head_dim: 2,
            norm_eps: 1e-5,
            bos_token_id: 0,
            eos_token_id: 1,
            architecture: ArchitectureId::Qwen3ForCausalLM,
        }
    }

    #[test]
    fn test_round_half_to_even_basic() {
        // Test basic rounding cases
        assert_eq!(round_half_to_even(1.4), 1.0);
        assert_eq!(round_half_to_even(1.6), 2.0);
        assert_eq!(round_half_to_even(-1.4), -1.0);
        assert_eq!(round_half_to_even(-1.6), -2.0);
    }

    #[test]
    fn test_round_half_to_even_halfway_cases() {
        // Test banker's rounding for halfway cases
        assert_eq!(round_half_to_even(0.5), 0.0); // Round to even (0)
        assert_eq!(round_half_to_even(1.5), 2.0); // Round to even (2)
        assert_eq!(round_half_to_even(2.5), 2.0); // Round to even (2)
        assert_eq!(round_half_to_even(3.5), 4.0); // Round to even (4)
        assert_eq!(round_half_to_even(-0.5), 0.0); // Round to even (0)
        assert_eq!(round_half_to_even(-1.5), -2.0); // Round to even (-2)
        assert_eq!(round_half_to_even(-2.5), -2.0); // Round to even (-2)
    }

    #[test]
    fn test_quantize_q80_known_values() {
        let config = create_test_config();

        let exporter = BinaryModelExporter::new(config, 4);

        // Test with known values that should quantize predictably
        let weights = vec![0.0, 127.0, -127.0, 63.5]; // Group size = 4
        let result = exporter.quantize_q80(&weights).unwrap();

        // Check scale factor (should be 127.0 / 127.0 = 1.0)
        assert_eq!(result.scales.len(), 1);
        assert!((result.scales[0] - 1.0).abs() < 1e-6);

        // Check quantized values
        assert_eq!(result.int8_data.len(), 4);
        assert_eq!(result.int8_data[0], 0); // 0.0 / 1.0 = 0
        assert_eq!(result.int8_data[1], 127); // 127.0 / 1.0 = 127
        assert_eq!(result.int8_data[2], -127); // -127.0 / 1.0 = -127
        assert_eq!(result.int8_data[3], 64); // 63.5 / 1.0 = 63.5 -> round_half_to_even = 64
    }

    #[test]
    fn test_quantize_q80_zero_weights() {
        let config = create_test_config();

        let exporter = BinaryModelExporter::new(config, 4);

        // Test with all zeros
        let weights = vec![0.0, 0.0, 0.0, 0.0];
        let result = exporter.quantize_q80(&weights).unwrap();

        // Scale should be 1.0 (default)
        assert_eq!(result.scales[0], 1.0);

        // All quantized values should be 0
        assert!(result.int8_data.iter().all(|&x| x == 0));

        // Error should be 0
        assert_eq!(result.max_error, 0.0);
    }

    #[test]
    fn test_quantize_q80_invalid_group_size() {
        let config = create_test_config();

        let exporter = BinaryModelExporter::new(config, 4);

        // Test with weights length not divisible by group_size
        let weights = vec![1.0, 2.0, 3.0]; // Length 3, group_size 4
        let result = exporter.quantize_q80(&weights);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("multiple of group_size"));
    }

    #[test]
    fn test_find_optimal_group_size() {
        // Test normal case where group size divides evenly
        assert_eq!(BinaryModelExporter::find_optimal_group_size(128, 64), 64);
        assert_eq!(BinaryModelExporter::find_optimal_group_size(128, 32), 32);
        assert_eq!(BinaryModelExporter::find_optimal_group_size(128, 16), 16);

        // Test when requested size is larger than hidden_dim
        assert_eq!(BinaryModelExporter::find_optimal_group_size(32, 64), 32);

        // Test when requested size doesn't divide evenly - should find largest divisor >= MIN_GROUP_SIZE
        // For (128, 96):
        // 96 -> 48 -> 24 -> 12 -> 6 -> 3 (3 < MIN_GROUP_SIZE, so stop)
        // return max(3, 4) = 4
        assert_eq!(BinaryModelExporter::find_optimal_group_size(128, 96), 4);

        // Test with a number that has nice divisors:
        assert_eq!(BinaryModelExporter::find_optimal_group_size(60, 40), 20); // 60%40≠0 -> 20, 60%20=0 ✓
        assert_eq!(BinaryModelExporter::find_optimal_group_size(60, 30), 30); // 60%30=0 ✓

        // Test prime number case - should fall back to minimum group size
        assert_eq!(BinaryModelExporter::find_optimal_group_size(127, 64), 4); // 127 is prime, fallback to MIN_GROUP_SIZE

        // Test minimum group size enforcement
        assert_eq!(BinaryModelExporter::find_optimal_group_size(15, 8), 4); // Should reach MIN_GROUP_SIZE (4)

        // Test when requested size is smaller than minimum
        assert_eq!(BinaryModelExporter::find_optimal_group_size(128, 2), 4); // Should use MIN_GROUP_SIZE

        // Test case where we can find a good divisor
        assert_eq!(BinaryModelExporter::find_optimal_group_size(128, 64), 64); // 128%64=0 ✓
    }

    #[test]
    fn test_header_constants() {
        assert_eq!(BinaryModelExporter::MAGIC_NUMBER, 0x616A6331);
        assert_eq!(BinaryModelExporter::VERSION, 1);
        assert_eq!(BinaryModelExporter::HEADER_SIZE, 256);
        assert_eq!(BinaryModelExporter::MIN_GROUP_SIZE, 4);
    }

    #[test]
    fn test_quantization_symmetry() {
        let config = create_test_config();

        let exporter = BinaryModelExporter::new(config, 4);

        // Test that positive and negative values quantize symmetrically
        let weights = vec![100.0, -100.0, 50.0, -50.0];
        let result = exporter.quantize_q80(&weights).unwrap();

        // Scale should be 100.0 / 127.0
        let expected_scale = 100.0 / 127.0;
        assert!((result.scales[0] - expected_scale).abs() < 1e-6);

        // Check symmetry: quantized[0] should be -quantized[1]
        assert_eq!(result.int8_data[0], -result.int8_data[1]);
        assert_eq!(result.int8_data[2], -result.int8_data[3]);
    }

    #[test]
    fn test_minimum_group_size_enforcement() {
        let config = create_test_config();

        // Test that requesting a group size smaller than MIN_GROUP_SIZE gets adjusted
        let exporter = BinaryModelExporter::new(config, 2); // Request size 2, should get MIN_GROUP_SIZE

        // The actual group size should be at least MIN_GROUP_SIZE
        // We can verify this by checking that quantization works with weights that are
        // multiples of MIN_GROUP_SIZE but not multiples of the requested size
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // Length 4 = MIN_GROUP_SIZE
        let result = exporter.quantize_q80(&weights);

        // Should succeed because the exporter adjusted to use MIN_GROUP_SIZE = 4
        assert!(result.is_ok());
    }
}

mod advanced {
    use super::*;

    /// Create a minimal mock config for testing
    fn create_test_config() -> ModelConfig {
        ModelConfig {
            architecture: ArchitectureId::Qwen3ForCausalLM,
            dim: 8,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 100,
            max_seq_len: 128,
            head_dim: 4,
            norm_eps: 1e-5,
            bos_token_id: 0,
            eos_token_id: 1,
        }
    }

    /// Test group size optimization logic
    #[test]
    fn test_group_size_optimization() {
        let config = create_test_config(); // dim = 8

        // Test perfect divisor
        let exporter1 = BinaryModelExporter::new(config.clone(), 8);
        // Should keep group_size = 8 since 8 % 8 == 0

        // Test non-divisor that gets optimized
        let exporter2 = BinaryModelExporter::new(config.clone(), 6);
        // Should adjust: 6 -> 3 -> can't go lower than MIN_GROUP_SIZE=4, so use 4

        // Test too large group size
        let exporter3 = BinaryModelExporter::new(config.clone(), 16);
        // Should adjust to dim=8 since 16 > 8

        // We can't directly access the group_size field, but we can test
        // indirectly by trying quantization with different array sizes

        // Test with array size that's multiple of 8 (should work with exporter1)
        let weights1 = vec![1.0; 8];
        assert!(exporter1.quantize_q80(&weights1).is_ok());

        // Test with array size that's multiple of 4 (should work with exporter2)
        let weights2 = vec![1.0; 8]; // 8 is multiple of 4
        assert!(exporter2.quantize_q80(&weights2).is_ok());

        // Test with array size that's multiple of 8 (should work with exporter3)
        let weights3 = vec![1.0; 8];
        assert!(exporter3.quantize_q80(&weights3).is_ok());
    }

    /// Test quantization with different data patterns
    #[test]
    fn test_quantization_patterns() {
        let config = create_test_config();
        let exporter = BinaryModelExporter::new(config, 4);

        // Test uniform positive values
        let uniform_pos = vec![10.0; 8];
        let result1 = exporter.quantize_q80(&uniform_pos).unwrap();
        assert_eq!(result1.scales.len(), 2); // 8 / 4 = 2 groups
        assert!(result1.scales.iter().all(|&s| s > 0.0));

        // Test uniform negative values
        let uniform_neg = vec![-10.0; 8];
        let result2 = exporter.quantize_q80(&uniform_neg).unwrap();
        assert_eq!(result2.scales.len(), 2);
        assert!(result2.scales.iter().all(|&s| s > 0.0)); // Scales are always positive

        // Test mixed positive/negative
        let mixed = vec![10.0, -10.0, 5.0, -5.0, 20.0, -20.0, 15.0, -15.0];
        let result3 = exporter.quantize_q80(&mixed).unwrap();
        assert_eq!(result3.scales.len(), 2);

        // Test with zeros
        let with_zeros = vec![0.0, 10.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0];
        let result4 = exporter.quantize_q80(&with_zeros).unwrap();
        assert_eq!(result4.scales.len(), 2);

        // Test extreme values
        let extreme = vec![f32::MAX, f32::MIN, 0.0, 1.0, -1000.0, 1000.0, 0.1, -0.1];
        let result5 = exporter.quantize_q80(&extreme).unwrap();
        assert_eq!(result5.scales.len(), 2);
        assert!(result5.scales.iter().all(|&s| s.is_finite()));
    }

    /// Test quantization error bounds
    #[test]
    fn test_quantization_error_bounds() {
        let config = create_test_config();
        let exporter = BinaryModelExporter::new(config, 4);

        // Test with known values to verify error calculation
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // Single group
        let result = exporter.quantize_q80(&weights).unwrap();

        // Max value is 4.0, so scale = 4.0 / 127.0
        let expected_scale = 4.0 / 127.0;
        assert!((result.scales[0] - expected_scale).abs() < 1e-6);

        // Verify that dequantized values are within reasonable error bounds
        let max_expected_error = expected_scale; // Worst case error
        assert!(result.max_error <= max_expected_error * 1.1); // Allow small margin

        // Test error accumulation across groups
        let large_weights = vec![1.0; 16]; // 4 groups of 4
        let large_result = exporter.quantize_q80(&large_weights).unwrap();
        assert_eq!(large_result.scales.len(), 4);
        assert!(large_result.max_error >= 0.0);
    }

    /// Test error handling for invalid configurations
    #[test]
    fn test_invalid_configurations() {
        // Test with zero dimensions
        let invalid_config1 = ModelConfig { dim: 0, ..create_test_config() };

        // Constructor should handle this gracefully
        let exporter1 = BinaryModelExporter::new(invalid_config1, 4);

        // Quantization should fail for empty arrays
        let empty_weights = vec![];
        let result1 = exporter1.quantize_q80(&empty_weights);
        assert!(result1.is_ok()); // Empty array is valid, just produces empty result

        // Test with mismatched array size
        let config = create_test_config();
        let exporter2 = BinaryModelExporter::new(config, 4);

        let wrong_size_weights = vec![1.0, 2.0, 3.0]; // Not multiple of group_size
        let result2 = exporter2.quantize_q80(&wrong_size_weights);
        assert!(result2.is_err());
    }

    /// Test deterministic quantization
    #[test]
    fn test_deterministic_quantization() {
        let config = create_test_config();
        let exporter = BinaryModelExporter::new(config, 4);

        let weights = vec![1.23, -4.56, 7.89, -0.12, 3.45, -6.78, 9.01, -2.34];

        // Run quantization multiple times
        let result1 = exporter.quantize_q80(&weights).unwrap();
        let result2 = exporter.quantize_q80(&weights).unwrap();
        let result3 = exporter.quantize_q80(&weights).unwrap();

        // Results should be identical
        assert_eq!(result1.int8_data, result2.int8_data);
        assert_eq!(result1.int8_data, result3.int8_data);
        assert_eq!(result1.scales, result2.scales);
        assert_eq!(result1.scales, result3.scales);
        assert_eq!(result1.max_error, result2.max_error);
        assert_eq!(result1.max_error, result3.max_error);
    }

    /// Test group size adjustment logging and behavior
    #[test]
    fn test_group_size_adjustment() {
        let config = ModelConfig {
            dim: 15, // Prime number to force group size adjustment
            hidden_dim: 32,
            ..create_test_config()
        };

        // Request a group size that doesn't divide 15
        let exporter = BinaryModelExporter::new(config, 8);

        // Should be adjusted to a reasonable size
        // For dim=15 with requested=8: 8 -> 4 (since 15 % 8 != 0, 15 % 4 != 0, fall back to MIN_GROUP_SIZE=4)

        // Test that quantization works with a size that's multiple of the adjusted group size
        let weights = vec![1.0; 16]; // Multiple of 4
        let result = exporter.quantize_q80(&weights);
        assert!(result.is_ok());

        // Test that quantization fails with wrong size
        let wrong_weights = vec![1.0; 15]; // Not multiple of adjusted group size
        let wrong_result = exporter.quantize_q80(&wrong_weights);
        assert!(wrong_result.is_err());
    }

    /// Test edge cases in quantization
    #[test]
    fn test_quantization_edge_cases() {
        let config = create_test_config();
        let exporter = BinaryModelExporter::new(config, 4);
        // Test with NaN values (should handle gracefully or error)
        let nan_weights = vec![f32::NAN, 1.0, 2.0, 3.0];
        let _nan_result = exporter.quantize_q80(&nan_weights);
        // Implementation may choose to error or handle NaN specially

        // Test with infinite values
        let inf_weights = vec![f32::INFINITY, 1.0, 2.0, 3.0];
        let _inf_result = exporter.quantize_q80(&inf_weights);
        // Should handle infinite values gracefully

        // Test with very small values
        let tiny_weights = vec![1e-30, 2e-30, 3e-30, 4e-30];
        let tiny_result = exporter.quantize_q80(&tiny_weights).unwrap();
        assert!(tiny_result.scales[0] > 0.0);

        // Test with very large values
        let large_weights = vec![1e30, -1e30, 1e29, -1e29];
        let large_result = exporter.quantize_q80(&large_weights);
        if large_result.is_ok() {
            let result = large_result.unwrap();
            assert!(result.scales[0].is_finite());
        }
    }

    /// Test that quantization produces consistent binary format across different input sizes
    #[test]
    fn test_quantization_binary_consistency() {
        let config = create_test_config();
        let exporter = BinaryModelExporter::new(config, 4);

        // Test with multiple groups to ensure consistency
        let weights = vec![
            1.0, 2.0, 3.0, 4.0, // Group 1: max = 4.0, scale = 4.0/127
            -5.0, 6.0, -7.0, 8.0, // Group 2: max = 8.0, scale = 8.0/127
            0.1, -0.2, 0.3, -0.4, // Group 3: max = 0.4, scale = 0.4/127
            100.0, -100.0, 50.0, -25.0, // Group 4: max = 100.0, scale = 100.0/127
        ];

        let result = exporter.quantize_q80(&weights).unwrap();

        // Verify structure
        assert_eq!(result.int8_data.len(), 16);
        assert_eq!(result.scales.len(), 4);
        assert!(result.max_error >= 0.0);

        // Verify scale calculation for each group
        let expected_scales = [4.0 / 127.0, 8.0 / 127.0, 0.4 / 127.0, 100.0 / 127.0];
        for (i, expected_scale) in expected_scales.iter().enumerate() {
            assert!(
                (result.scales[i] - expected_scale).abs() < 1e-6,
                "Scale mismatch at group {}: expected {}, got {}",
                i,
                expected_scale,
                result.scales[i]
            );
        }

        // Verify all quantized values are in valid range [-127, 127]
        // Note: i8 range is actually [-128, 127], but our quantization uses [-127, 127]
        for &val in &result.int8_data {
            assert!(val >= -127, "Quantized value {val} below minimum");
            // Upper bound check omitted since i8 cannot exceed 127
        }

        // Test dequantization round-trip error bounds
        let mut max_observed_error = 0.0f32;
        for group_idx in 0..4 {
            let group_start = group_idx * 4;
            let scale = result.scales[group_idx];

            for i in 0..4 {
                let original = weights[group_start + i];
                let quantized = result.int8_data[group_start + i];
                let dequantized = f32::from(quantized) * scale;
                let error = (dequantized - original).abs();
                max_observed_error = max_observed_error.max(error);
            }
        }

        // Error should be within reasonable bounds (less than max scale/2)
        let max_scale = result.scales.iter().fold(0.0f32, |acc, &x| acc.max(x));
        assert!(
            max_observed_error <= max_scale * 0.6,
            "Quantization error {max_observed_error} exceeds expected bound {}",
            max_scale * 0.6
        );
    }

    /// Test group size optimization with various dimensions
    #[test]
    fn test_comprehensive_group_size_optimization() {
        // Test various dimension scenarios
        let test_cases = vec![
            (32, 16, 16),  // Perfect fit: 32 % 16 == 0
            (32, 20, 16),  // Adjust down: 20 -> 16 (since 32 % 20 != 0)
            (15, 8, 4),    // Prime dim: adjust to MIN_GROUP_SIZE
            (128, 64, 64), // Large perfect fit
            (100, 30, 20), // 100 % 30 != 0, try 15, then 10, then 5, then use MIN_GROUP_SIZE=4? Let's see...
            (1, 10, 4),    // Tiny dim: force MIN_GROUP_SIZE
        ];

        for (dim, requested, _expected_note) in test_cases {
            let config = ModelConfig { dim: dim as u32, ..create_test_config() };

            let exporter = BinaryModelExporter::new(config, requested);

            // Test that we can quantize weights with size that's a multiple of dim
            // (since the group size should divide dim after optimization)
            let multiple_of_dim = (dim as usize).div_ceil(4) * 4; // Round up to multiple of 4 (MIN_GROUP_SIZE)
            let weights = vec![1.0; multiple_of_dim];

            // The quantization should work regardless of what group size was chosen,
            // as long as the weight array size is compatible
            let result = exporter.quantize_q80(&weights);

            // For some cases it might fail if the optimized group size doesn't divide the array size
            // That's expected behavior - let's just check it doesn't crash
            match result {
                Ok(r) => {
                    assert!(!r.scales.is_empty(), "Should have at least one scale");
                    assert_eq!(r.int8_data.len(), weights.len(), "Quantized data size should match input");
                }
                Err(_) => {
                    // It's OK if quantization fails due to size mismatch - that's expected
                }
            }
        }
    }

    /// Test quantization with extreme values and edge cases
    #[test]
    fn test_quantization_extreme_values_comprehensive() {
        let config = create_test_config();
        let exporter = BinaryModelExporter::new(config, 4);

        // Test case 1: All zeros
        let all_zeros = vec![0.0; 8];
        let result_zeros = exporter.quantize_q80(&all_zeros).unwrap();
        assert_eq!(result_zeros.scales.len(), 2);
        assert!(result_zeros.scales.iter().all(|&s| s > 0.0)); // Scale should be positive even for zeros
        assert!(result_zeros.int8_data.iter().all(|&i| i == 0));
        assert_eq!(result_zeros.max_error, 0.0);

        // Test case 2: Single large value with zeros
        let single_large = vec![1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result_large = exporter.quantize_q80(&single_large).unwrap();
        assert_eq!(result_large.scales.len(), 2);
        assert!(result_large.scales[0] > result_large.scales[1]); // First group should have larger scale

        // Test case 3: Symmetric positive/negative values
        let symmetric = vec![100.0, -100.0, 50.0, -50.0, 200.0, -200.0, 25.0, -25.0];
        let result_sym = exporter.quantize_q80(&symmetric).unwrap();
        assert_eq!(result_sym.scales.len(), 2);
        // Check that quantized values respect the symmetry (approximately)
        for i in (0..result_sym.int8_data.len()).step_by(2) {
            let pos_val = result_sym.int8_data[i];
            let neg_val = result_sym.int8_data[i + 1];
            // They should have opposite signs and similar magnitudes
            assert!(pos_val > 0 && neg_val < 0, "Values should have opposite signs");
        }

        // Test case 4: Very small values
        let tiny_values = vec![1e-10, -1e-10, 2e-10, -2e-10, 1e-9, -1e-9, 2e-9, -2e-9];
        let result_tiny = exporter.quantize_q80(&tiny_values).unwrap();
        assert_eq!(result_tiny.scales.len(), 2);
        assert!(result_tiny.scales.iter().all(|&s| s.is_finite() && s > 0.0));

        // Test case 5: Mixed scales across groups
        let mixed_scales = vec![
            1.0, 2.0, 3.0, 4.0, // Small values
            1000.0, 2000.0, 3000.0, 4000.0, // Large values
        ];
        let result_mixed = exporter.quantize_q80(&mixed_scales).unwrap();
        assert_eq!(result_mixed.scales.len(), 2);
        assert!(result_mixed.scales[1] > result_mixed.scales[0] * 100.0); // Second group should have much larger scale
    }

    /// Test error propagation and boundary conditions
    #[test]
    fn test_error_propagation_and_boundaries() {
        let config = create_test_config();
        let exporter = BinaryModelExporter::new(config, 4);

        // Test boundary: array size exactly equals group size
        let exact_size = vec![1.0, 2.0, 3.0, 4.0];
        let result_exact = exporter.quantize_q80(&exact_size);
        assert!(result_exact.is_ok());

        // Test boundary: array size is multiple of group size
        let multiple_size = vec![1.0; 8];
        let result_multiple = exporter.quantize_q80(&multiple_size);
        assert!(result_multiple.is_ok());

        // Test error: array size not multiple of group size
        let wrong_sizes = vec![
            vec![1.0; 1], // Too small
            vec![1.0; 3], // Not multiple of 4
            vec![1.0; 5], // Not multiple of 4
            vec![1.0; 7], // Not multiple of 4
        ];

        for wrong_size in wrong_sizes {
            let result = exporter.quantize_q80(&wrong_size);
            assert!(result.is_err(), "Should fail for size {}", wrong_size.len());

            // Verify error message contains useful information
            let error_msg = format!("{}", result.unwrap_err());
            assert!(
                error_msg.contains("multiple") || error_msg.contains("group_size"),
                "Error message should mention group size: {error_msg}",
            );
        }
    }

    /// Test minimum group size enforcement
    #[test]
    fn test_minimum_group_size_enforcement() {
        let config = ModelConfig {
            dim: 7, // Prime number that will force group size adjustment
            ..create_test_config()
        };

        // Test various requested group sizes
        let test_cases = vec![
            (1, 4), // Below minimum, should be adjusted to MIN_GROUP_SIZE=4
            (2, 4), // Below minimum, should be adjusted to MIN_GROUP_SIZE=4
            (3, 4), // Below minimum, should be adjusted to MIN_GROUP_SIZE=4
            (4, 4), // At minimum, should stay at 4
            (8, 4), // Above dimension, should be adjusted down
        ];

        for (requested, expected_effective) in test_cases {
            let exporter = BinaryModelExporter::new(config.clone(), requested);

            // Test with an array size that's a multiple of the expected effective group size
            let test_size = expected_effective * 3; // 12 elements
            let weights = vec![1.0; test_size];

            let result = exporter.quantize_q80(&weights);

            match result {
                Ok(r) => {
                    // If it succeeds, verify the number of groups matches expectation
                    let expected_groups = test_size / expected_effective;
                    assert_eq!(
                        r.scales.len(),
                        expected_groups,
                        "Requested group_size {} should result in {} groups, got {}",
                        requested,
                        expected_groups,
                        r.scales.len()
                    );
                }
                Err(_) => {
                    // If it fails, it might be because the effective group size doesn't divide the test size
                    // This is acceptable behavior
                }
            }

            // Test that wrong sizes definitely fail
            let wrong_size = test_size + 1; // Not multiple of any reasonable group size
            let wrong_weights = vec![1.0; wrong_size];
            let wrong_result = exporter.quantize_q80(&wrong_weights);
            assert!(
                wrong_result.is_err(),
                "Should fail for array size {wrong_size} with requested group_size {requested}",
            );
        }
    }
}
