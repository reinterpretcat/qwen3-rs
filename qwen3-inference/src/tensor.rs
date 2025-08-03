use rayon::prelude::*;
use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub q: Cow<'static, [i8]>,
    pub s: Cow<'static, [f32]>,
}

impl QuantizedTensor {
    // Create with owned data (for temporary/working tensors)
    pub fn new(size: usize, group_size: usize) -> Self {
        let scale_size = size / group_size;
        Self { q: Cow::Owned(vec![0; size]), s: Cow::Owned(vec![0.0; scale_size]) }
    }

    // Create from borrowed slices (for memory-mapped data)
    pub fn from_slices(q: &'static [i8], s: &'static [f32]) -> Self {
        Self { q: Cow::Borrowed(q), s: Cow::Borrowed(s) }
    }
}

pub fn matmul(xout: &mut [f32], x: &QuantizedTensor, w: &QuantizedTensor, n: usize, d: usize, group_size: usize) {
    assert!(xout.len() >= d, "Output slice length must be at least d parameter: {} >= {}", xout.len(), d);

    xout.par_iter_mut().enumerate().take(d).for_each(|(i, out_val)| {
        compute_matmul_row(out_val, x, w, i, n, group_size);
    });
}

#[inline]
fn compute_matmul_row(
    out_val: &mut f32,
    x: &QuantizedTensor,
    w: &QuantizedTensor,
    row_idx: usize,
    n: usize,
    group_size: usize,
) {
    debug_assert_eq!(n % group_size, 0, "n must be divisible by group_size");

    let weight_row_offset = row_idx * n;
    let num_groups = n / group_size;

    *out_val = (0..num_groups)
        .map(|group_idx| {
            let group_start = group_idx * group_size;
            let weight_group_offset = weight_row_offset + group_start;

            let quantized_dot_product: i32 = x.q[group_start..group_start + group_size]
                .iter()
                .zip(&w.q[weight_group_offset..weight_group_offset + group_size])
                .map(|(&x_quant, &w_quant)| x_quant as i32 * w_quant as i32)
                .sum();

            let weight_scale = w.s[weight_group_offset / group_size];
            let input_scale = x.s[group_idx];

            quantized_dot_product as f32 * weight_scale * input_scale
        })
        .sum();
}

/// Dequantizes a quantized tensor into a float buffer.
///
/// For each group of quantized values, multiplies by the corresponding scale factor.
///
/// # Arguments
/// * `qx` - The quantized tensor (with quantized values and scale factors)
/// * `x` - Output buffer for dequantized values (must be at least as large as `qx.q`)
/// * `group_size` - Number of elements per quantization group
pub fn dequantize(qx: &QuantizedTensor, x: &mut [f32], group_size: usize) {
    debug_assert_eq!(x.len(), qx.q.len(), "Output buffer size must match quantized tensor size");
    debug_assert_eq!(qx.s.len(), x.len() / group_size);

    for (i, &q_val) in qx.q.iter().enumerate() {
        let scale = qx.s[i / group_size];
        x[i] = q_val as f32 * scale;
    }
}

/// Quantizes a float buffer into a quantized tensor using per-group scaling.
///
/// For each group, finds the max absolute value, computes a scale, and quantizes values to i8.
///
/// # Arguments
/// * `qx` - The quantized tensor to write into (must have preallocated `q` and `s`)
/// * `x` - Input float buffer to quantize
/// * `size` - Number of elements to quantize (should be <= x.len())
/// * `group_size` - Number of elements per quantization group
pub fn quantize(qx: &mut QuantizedTensor, x: &[f32], size: usize, group_size: usize) {
    debug_assert_eq!(x.len(), size);
    debug_assert!(qx.q.len() >= size, "Quantized buffer too small: {} < {}", qx.q.len(), size);
    debug_assert!(qx.s.len() >= size / group_size, "Scale buffer too small: {} < {}", qx.s.len(), size / group_size);

    const Q_MAX: f32 = 127.0;
    let num_groups = size / group_size;

    // Get separate mutable references to avoid borrowing conflicts
    let q_data = qx.q.to_mut();
    let s_data = qx.s.to_mut();

    for group in 0..num_groups {
        let group_start = group * group_size;
        let group_end = group_start + group_size;

        // Find the maximum absolute value in the group
        let wmax = x[group_start..group_end].iter().fold(0.0f32, |acc, &val| acc.max(val.abs()));

        let scale = wmax / Q_MAX;
        s_data[group] = scale;

        // Quantize the group
        for (i, &val) in x[group_start..group_end].iter().enumerate() {
            let quant_value = if scale != 0.0 { val / scale } else { 0.0 };
            q_data[group_start + i] = quant_value.round() as i8;
        }
    }
}
