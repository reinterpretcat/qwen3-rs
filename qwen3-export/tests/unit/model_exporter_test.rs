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
        architectures: vec!["Qwen3ForCausalLM".to_string()],
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
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("multiple of group_size")
    );
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
