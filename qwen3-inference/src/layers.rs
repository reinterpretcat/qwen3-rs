use crate::configuration::ModelConfig;
use crate::tensor::{QuantizedTensor, quantize};
use anyhow::Result;
use rayon::prelude::*;

/// Epsilon value for numerical stability in normalization
const EPSILON: f32 = 1e-6;

/// Base frequency for RoPE (Rotary Position Embedding)
const ROPE_BASE_FREQ: f32 = 1e6;

/// Token embedding layer - converts token IDs to dense vectors
///
/// **Purpose**: Maps discrete vocabulary tokens to continuous vector space
/// **Shape**: [vocab_size, embedding_dim]
/// **Note**: Often shared with output projection weights (weight tying)
pub struct TokenEmbedding {
    pub embedding_table: Vec<f32>,
    pub dim: usize,
}

impl TokenEmbedding {
    pub fn new(embedding_table: Vec<f32>, dim: usize) -> Self {
        Self { embedding_table, dim }
    }

    pub fn forward(&self, token: usize, output: &mut [f32]) {
        let start_idx = token * self.dim;
        let end_idx = start_idx + self.dim;
        output[..self.dim].copy_from_slice(&self.embedding_table[start_idx..end_idx]);
    }
}

impl std::fmt::Debug for TokenEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenEmbedding")
            .field("dim", &self.dim)
            .field("vocab_size", &(self.embedding_table.len() / self.dim))
            .finish()
    }
}

/// RMS Layer Normalization - alternative to LayerNorm used in modern LLMs
///
/// **Mathematical Formula**:
/// ```text
/// RMSNorm(x) = x / RMS(x) * γ
/// where RMS(x) = sqrt(mean(x²) + ε)
/// ```
///
/// **Advantages over LayerNorm**:
/// - Computationally simpler (no mean subtraction)
/// - Better numerical stability
/// - Widely adopted in modern architectures (LLaMA, PaLM, etc.)
pub struct RMSNorm {
    pub weight: Vec<f32>,
}

impl RMSNorm {
    pub fn new(weight: Vec<f32>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, output: &mut [f32], input: &[f32]) {
        debug_assert_eq!(output.len(), input.len());
        debug_assert_eq!(input.len(), self.weight.len());

        let sum_of_squares = input.iter().map(|&x| x * x).sum::<f32>();
        let rms_norm_factor = 1.0f32 / ((sum_of_squares / input.len() as f32) + EPSILON).sqrt();

        output.iter_mut().zip(input.iter()).zip(self.weight.iter()).for_each(|((out, &inp), &w)| {
            *out = w * (rms_norm_factor * inp);
        });
    }

    pub fn forward_inplace(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.weight.len());

        let sum_of_squares = x.iter().map(|&val| val * val).sum::<f32>();
        let rms_norm_factor = 1.0f32 / ((sum_of_squares / x.len() as f32) + EPSILON).sqrt();

        x.iter_mut().zip(self.weight.iter()).for_each(|(val, &w)| {
            *val = w * (rms_norm_factor * *val);
        });
    }
}

impl std::fmt::Debug for RMSNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RMSNorm").field("dim", &self.weight.len()).finish()
    }
}

/// Rotary Position Embedding (RoPE) - relative position encoding mechanism
///
/// **Purpose**: Encodes positional information directly into query/key vectors
/// **Advantages over absolute position embeddings**:
/// - Naturally handles variable sequence lengths
/// - Provides relative position awareness
/// - Better extrapolation to longer sequences
/// - No learnable parameters required
///
/// **Mathematical Foundation**:
/// - Rotates pairs of dimensions in Q/K vectors based on position
/// - Frequency decreases with dimension index for multi-scale position encoding
/// - Enables attention to naturally focus on relative distances
pub struct RoPE {
    pub head_dim: usize,
}

impl RoPE {
    pub fn new(head_dim: usize) -> Self {
        Self { head_dim }
    }

    pub fn compute_freqs(&self, pos: usize) -> Vec<(f32, f32)> {
        let head_dim_half = self.head_dim / 2;

        (0..head_dim_half)
            .map(|dim_idx| {
                let freq = ROPE_BASE_FREQ.powf(-(dim_idx as f32) / head_dim_half as f32);
                let angle = pos as f32 * freq;
                (angle.cos(), angle.sin())
            })
            .collect()
    }

    pub fn apply(&self, slice: &mut [f32], freqs: &[(f32, f32)]) {
        let head_dim_half = slice.len() / 2;
        let (first_half, second_half) = slice.split_at_mut(head_dim_half);

        first_half.iter_mut().zip(second_half.iter_mut()).zip(freqs.iter()).for_each(
            |((x, y), &(cos_freq, sin_freq))| {
                let x_val = *x;
                let y_val = *y;
                *x = x_val * cos_freq - y_val * sin_freq;
                *y = x_val * sin_freq + y_val * cos_freq;
            },
        );
    }
}

impl std::fmt::Debug for RoPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoPE").field("head_dim", &self.head_dim).finish()
    }
}

/// Linear layer with quantized weights for memory-efficient inference
///
/// **Quantization Strategy**:
/// - Weights stored in reduced precision (INT8)
/// - Dynamic dequantization during computation
/// - Significant memory savings with minimal accuracy loss
pub struct Linear {
    pub weight: QuantizedTensor,
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: usize,
}

impl Linear {
    pub fn new(weight: QuantizedTensor, in_features: usize, out_features: usize, group_size: usize) -> Self {
        Self { weight, in_features, out_features, group_size }
    }

    pub fn forward(&self, output: &mut [f32], input: &QuantizedTensor) {
        crate::tensor::matmul(output, input, &self.weight, self.in_features, self.out_features, self.group_size);
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("group_size", &self.group_size)
            .finish()
    }
}

/// Multi-Head Attention with Grouped Query Attention (GQA) optimization
///
/// **Architecture Details**:
/// - **Standard MHA**: n_heads query heads, n_heads key heads, n_heads value heads
/// - **GQA**: n_heads query heads, n_kv_heads key/value heads (n_kv_heads < n_heads)
/// - **Memory Efficiency**: Reduces KV cache size by sharing key/value heads
/// - **Performance**: Maintains quality while reducing memory bandwidth
///
/// **Components**:
/// - **Q, K, V Projections**: Linear transformations to query, key, value spaces
/// - **QK-RMSNorm**: Qwen3-style normalization applied to queries and keys
/// - **RoPE**: Rotary position embedding for relative position encoding
/// - **Scaled Dot-Product Attention**: Core attention mechanism with softmax
/// - **Output Projection**: Final linear transformation
///
/// **Attention Formula**:
/// ```text
/// Attention(Q,K,V) = softmax(QK^T / √d_k)V
/// ```
pub struct MultiHeadAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub rope: RoPE,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_mul: usize,
    pub seq_len: usize,
}

impl MultiHeadAttention {
    pub fn new(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        config: &ModelConfig,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            rope: RoPE::new(config.head_dim),
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            kv_mul: config.n_heads / config.n_kv_heads,
            seq_len: config.seq_len,
        }
    }

    pub fn forward(&self, pos: usize, layer_idx: usize, state: &mut RunState) {
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_cache_offset = layer_idx * self.seq_len * kv_dim;
        let current_pos_offset = kv_cache_offset + pos * kv_dim;

        // Compute Q, K, V projections
        self.wq.forward(&mut state.q, &state.xq);
        self.wk.forward(&mut state.key_cache[current_pos_offset..], &state.xq);
        self.wv.forward(&mut state.value_cache[current_pos_offset..], &state.xq);

        // Apply normalization and RoPE
        let rope_freqs = self.rope.compute_freqs(pos);
        self.apply_qk_normalization_and_rope(current_pos_offset, &rope_freqs, state);

        // Compute attention
        self.compute_attention(pos, kv_cache_offset, state);
    }

    fn apply_qk_normalization_and_rope(
        &self,
        current_pos_offset: usize,
        rope_freqs: &[(f32, f32)],
        state: &mut RunState,
    ) {
        // Process Query heads
        for head_idx in 0..self.n_heads {
            let q_range = head_idx * self.head_dim..(head_idx + 1) * self.head_dim;
            let q_slice = &mut state.q[q_range.clone()];

            state.temp_workspace[..self.head_dim].copy_from_slice(q_slice);
            self.q_norm.forward(q_slice, &state.temp_workspace[..self.head_dim]);
            self.rope.apply(q_slice, rope_freqs);
        }

        // Process Key heads
        for head_idx in 0..self.n_kv_heads {
            let k_range =
                current_pos_offset + head_idx * self.head_dim..current_pos_offset + (head_idx + 1) * self.head_dim;
            let k_slice = &mut state.key_cache[k_range];

            state.temp_workspace[..self.head_dim].copy_from_slice(k_slice);
            self.k_norm.forward(k_slice, &state.temp_workspace[..self.head_dim]);
            self.rope.apply(k_slice, rope_freqs);
        }
    }

    fn compute_attention(&self, pos: usize, kv_cache_offset: usize, state: &mut RunState) {
        let attention_scale = (self.head_dim as f32).sqrt().recip();
        let kv_dim = self.n_kv_heads * self.head_dim;

        state
            .att
            .par_chunks_mut(self.seq_len)
            .zip(state.xb.par_chunks_mut(self.head_dim))
            .zip((0..self.n_heads).into_par_iter())
            .for_each(|((att_slice, xb_slice), head_idx)| {
                let q_range = head_idx * self.head_dim..(head_idx + 1) * self.head_dim;
                let kv_head_idx = head_idx / self.kv_mul;

                // Compute attention scores for this head
                let att_head = &mut att_slice[0..=pos];

                // Vectorized dot product computation
                att_head.iter_mut().enumerate().for_each(|(time_step, att_score)| {
                    let k_cache_start = kv_cache_offset + time_step * kv_dim + kv_head_idx * self.head_dim;
                    let k_cache_end = k_cache_start + self.head_dim;

                    *att_score = state.q[q_range.clone()]
                        .iter()
                        .zip(&state.key_cache[k_cache_start..k_cache_end])
                        .map(|(&q, &k)| q * k)
                        .sum::<f32>()
                        * attention_scale;
                });

                // Apply softmax
                softmax(att_head);

                // Compute weighted sum of values
                xb_slice.fill(0.0);
                for time_step in 0..=pos {
                    let v_cache_start = kv_cache_offset + time_step * kv_dim + kv_head_idx * self.head_dim;
                    let v_cache_end = v_cache_start + self.head_dim;
                    let attention_weight = att_head[time_step];

                    xb_slice
                        .iter_mut()
                        .zip(&state.value_cache[v_cache_start..v_cache_end])
                        .for_each(|(out, &value)| *out += attention_weight * value);
                }
            });
    }
}

impl std::fmt::Debug for MultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiHeadAttention")
            .field("n_heads", &self.n_heads)
            .field("n_kv_heads", &self.n_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("wq", &self.wq)
            .field("wk", &self.wk)
            .field("wv", &self.wv)
            .field("wo", &self.wo)
            .field("q_norm", &self.q_norm)
            .field("k_norm", &self.k_norm)
            .finish()
    }
}

/// Feed-Forward Network with SwiGLU activation
///
/// **Architecture**: Two-layer MLP with gated activation
/// **Activation Function**: SwiGLU (Swish-Gated Linear Unit)
/// - Combines Swish activation with gating mechanism
/// - Formula: SwiGLU(x) = Swish(W1·x) ⊙ (W3·x)
/// - Where Swish(x) = x · sigmoid(x)
///
/// **Purpose**:
/// - Provides non-linear transformation and feature mixing
/// - Increases model capacity and expressiveness
/// - Typically expands dimension by 4x (hidden_dim = 4 * model_dim)
///
/// **Components**:
/// - **Gate Projection (W1)**: Projects to expanded dimension with gating
/// - **Up Projection (W3)**: Projects to expanded dimension for multiplication
/// - **Down Projection (W2)**: Projects back to original dimension
pub struct FeedForward {
    pub w1: Linear, // Gate projection
    pub w2: Linear, // Down projection
    pub w3: Linear, // Up projection
}

impl FeedForward {
    pub fn new(w1: Linear, w2: Linear, w3: Linear) -> Self {
        Self { w1, w2, w3 }
    }

    pub fn forward(&self, state: &mut RunState) {
        // Gate and up projections
        self.w1.forward(&mut state.hb, &state.xq);
        self.w3.forward(&mut state.hb2, &state.xq);

        // Apply SwiGLU activation
        state.hb.iter_mut().zip(state.hb2.iter()).for_each(|(gate_val, &linear_val)| {
            let swish_output = *gate_val * (1.0f32 + (-*gate_val).exp()).recip();
            *gate_val = swish_output * linear_val;
        });

        // Down projection
        quantize(&mut state.hq, &state.hb, state.hb.len(), self.w2.group_size);
        self.w2.forward(&mut state.xb, &state.hq);
    }
}

impl std::fmt::Debug for FeedForward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedForward")
            .field("hidden_dim", &self.w1.out_features)
            .field("w1", &self.w1)
            .field("w2", &self.w2)
            .field("w3", &self.w3)
            .finish()
    }
}

// Applies softmax normalization to a slice in-place.
pub(crate) fn softmax(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |acc, &val| acc.max(val));
    let sum = x
        .iter_mut()
        .map(|val| {
            *val = (*val - max_val).exp();
            *val
        })
        .sum::<f32>();
    let inv_sum = sum.recip();
    x.iter_mut().for_each(|val| *val *= inv_sum);
}

/// Runtime state for transformer inference.
///
/// This structure contains all the temporary buffers and caches needed
/// during model execution. Buffers are pre-allocated to avoid allocation
/// overhead during inference.
#[derive(Debug)]
pub struct RunState {
    /// Primary activation buffer for current layer input
    /// Shape: [dim]
    pub x: Vec<f32>,

    /// Secondary activation buffer for attention computations
    /// Shape: [n_heads * head_dim]
    pub xb: Vec<f32>,

    /// Tertiary activation buffer for residual connections
    /// Shape: [dim]
    pub xb2: Vec<f32>,

    /// Hidden state buffer for feed-forward computations
    /// Shape: [hidden_dim]
    pub hb: Vec<f32>,

    /// Secondary hidden buffer for FFN gate operations
    /// Shape: [hidden_dim]
    pub hb2: Vec<f32>,

    /// Quantized activation buffer for efficient computation
    /// Max shape: [n_heads * head_dim]
    pub xq: QuantizedTensor,

    /// Quantized hidden buffer for FFN operations
    /// Max shape: [hidden_dim]
    pub hq: QuantizedTensor,

    /// Query buffer for attention computation
    /// Shape: [n_heads * head_dim]
    pub q: Vec<f32>,

    /// Attention weights buffer
    /// Shape: [n_heads, seq_len]
    pub att: Vec<f32>,

    /// Final output logits over vocabulary
    /// Shape: [vocab_size]
    pub logits: Vec<f32>,

    /// Key-Value cache for efficient autoregressive generation
    /// Keys: [n_layers, seq_len, n_kv_heads * head_dim]
    pub key_cache: Vec<f32>,
    /// Values: [n_layers, seq_len, n_kv_heads * head_dim]
    pub value_cache: Vec<f32>,

    /// Temporary workspace to avoid allocations in hot paths.
    /// Used for intermediate computations
    pub temp_workspace: Vec<f32>,
}

impl RunState {
    /// Creates a new runtime state with pre-allocated buffers based on model configuration.
    pub fn new(config: &ModelConfig) -> Result<Self> {
        let ModelConfig {
            group_size,
            n_heads,
            head_dim,
            n_kv_heads,
            dim,
            hidden_dim,
            vocab_size,
            seq_len,
            n_layers,
            ..
        } = *config;

        let all_heads_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        Ok(Self {
            // Core activation buffers
            x: vec![0.0; dim],
            xb: vec![0.0; all_heads_dim],
            xb2: vec![0.0; dim],

            // FFN buffers
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],

            // Quantized buffers for efficient computation
            xq: QuantizedTensor::new(all_heads_dim, group_size),
            hq: QuantizedTensor::new(hidden_dim, group_size),

            // Attention-specific buffers
            q: vec![0.0; all_heads_dim],
            att: vec![0.0; n_heads * seq_len],

            // Output buffer
            logits: vec![0.0; vocab_size],

            // KV cache for autoregressive generation
            key_cache: vec![0.0; n_layers * seq_len * kv_dim],
            value_cache: vec![0.0; n_layers * seq_len * kv_dim],

            // Temporary workspace for computations
            temp_workspace: vec![0.0; head_dim],
        })
    }
}
