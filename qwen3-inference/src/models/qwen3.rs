use crate::{configuration::ModelConfig, layers::*, tensor::*, utils::MemoryMapper};
use anyhow::{Context, Result};

/// Main Transformer model implementing a decoder-only architecture.
pub struct Qwen3Transformer {
    config: ModelConfig,
    token_embedding: TokenEmbedding,
    blocks: Vec<Qwen3TransformerBlock>,
    final_norm: RMSNorm,
    lm_head: Linear,
    state: RunState,
    _mapper: MemoryMapper,
}

impl Qwen3Transformer {
    pub(crate) fn new(config: ModelConfig, mut mapper: MemoryMapper) -> Result<Self> {
        let weights = load_weights(&mut mapper, &config)?;

        // Initialize runtime state
        let state = RunState::new(&config)?;

        // Create transformer blocks
        let mut blocks = Vec::new();
        for layer_idx in 0..config.n_layers {
            let block = create_transformer_block(&config, layer_idx, &weights)?;
            blocks.push(block);
        }

        // Create final normalization
        let final_norm = RMSNorm::new(weights.rms_final_weight[..config.dim].to_vec());

        // Create language model head
        let lm_head = Linear::new(weights.wcls, config.dim, config.vocab_size, config.group_size);

        // Create token embedding
        let token_embedding = TokenEmbedding::new(weights.token_embedding_table, config.dim);

        Ok(Self {
            config,
            token_embedding,
            blocks,
            final_norm,
            lm_head,
            state,
            _mapper: mapper, // Keep the mapper alive for the lifetime of the transformer
        })
    }

    /// Forward pass through the transformer for autoregressive generation
    ///
    /// **Process Flow:**
    /// 1. **Token Embedding**: Convert token ID to dense vector representation
    /// 2. **Transformer Blocks**: Process through N decoder layers with self-attention and FFN
    /// 3. **Final Normalization**: Apply RMSNorm to output hidden states
    /// 4. **Classification Head**: Project to vocabulary space for next-token prediction
    ///
    /// **Arguments:**
    /// - `token`: Current input token ID
    /// - `pos`: Current position in sequence (for RoPE and KV cache indexing)
    ///
    /// **Returns:**
    /// - Probability distribution over vocabulary (logits) for next token prediction
    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        // Token embedding
        self.token_embedding.forward(token, &mut self.state.x);

        // Process through transformer blocks
        for block in &self.blocks {
            block.forward(pos, &mut self.state);
        }

        // Final normalization
        self.final_norm.forward_inplace(&mut self.state.x);

        // Classification head
        quantize(&mut self.state.xq, &self.state.x, self.state.x.len(), self.lm_head.group_size);
        self.lm_head.forward(&mut self.state.logits, &self.state.xq);

        &self.state.logits
    }

    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }
}

impl std::fmt::Debug for Qwen3Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct BlocksSummary<'a, T>(&'a [T]);

        impl<'a, T: std::fmt::Debug> std::fmt::Debug for BlocksSummary<'a, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_list()
                    .entries(self.0.iter().take(1))
                    .entry(&format_args!("... and {} more", self.0.len().saturating_sub(1)))
                    .finish()
            }
        }

        f.debug_struct("Qwen3Transformer")
            .field("config", &self.config)
            .field("token_embedding", &self.token_embedding)
            .field("blocks", &BlocksSummary(&self.blocks))
            .field("final_norm", &self.final_norm)
            .field("lm_head", &self.lm_head)
            .finish()
    }
}

/// Transformer Block - Core decoder layer combining self-attention and feed-forward
pub struct Qwen3TransformerBlock {
    pub attn_norm: RMSNorm,
    pub attention: MultiHeadAttention,
    pub ffn_norm: RMSNorm,
    pub feed_forward: FeedForward,
    pub layer_idx: usize,
}

impl Qwen3TransformerBlock {
    pub fn new(
        attn_norm: RMSNorm,
        attention: MultiHeadAttention,
        ffn_norm: RMSNorm,
        feed_forward: FeedForward,
        layer_idx: usize,
    ) -> Self {
        Self { attn_norm, attention, ffn_norm, feed_forward, layer_idx }
    }

    fn forward(&self, pos: usize, state: &mut RunState) {
        // Attention block with residual connection
        let dim = state.x.len();
        self.attn_norm.forward(&mut state.xb[..dim], &state.x);

        quantize(&mut state.xq, &state.xb[..dim], dim, self.attention.wq.group_size);

        self.attention.forward(pos, self.layer_idx, state);

        quantize(&mut state.xq, &state.xb, state.xb.len(), self.attention.wo.group_size);
        self.attention.wo.forward(&mut state.xb2, &state.xq);

        // Residual connection
        state.x.iter_mut().zip(state.xb2.iter()).for_each(|(x_val, &delta)| *x_val += delta);

        // Feed-forward block with residual connection
        self.ffn_norm.forward(&mut state.xb[..dim], &state.x);

        quantize(&mut state.xq, &state.xb[..dim], dim, self.feed_forward.w1.group_size);

        self.feed_forward.forward(state);

        // Residual connection
        state.x.iter_mut().zip(state.xb.iter()).for_each(|(x_val, &delta)| *x_val += delta);
    }
}

impl std::fmt::Debug for Qwen3TransformerBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerBlock")
            .field("layer_idx", &self.layer_idx)
            .field("attn_norm", &self.attn_norm)
            .field("attention", &self.attention)
            .field("ffn_norm", &self.ffn_norm)
            .field("feed_forward", &self.feed_forward)
            .finish()
    }
}

/// Loads all model weights from the checkpoint data.
///
/// This function reads weights in the order they appear in the checkpoint:
/// 1. Normalization weights (f32)
/// 2. Token embeddings (quantized)
/// 3. Attention weights (quantized)
/// 4. Feed-forward weights (quantized)
/// 5. Classification weights (quantized, may be shared)
fn load_weights(mapper: &mut MemoryMapper, config: &ModelConfig) -> Result<Qwen3TransformerWeights> {
    let ModelConfig {
        group_size,
        dim,
        n_layers,
        head_dim,
        vocab_size,
        hidden_dim,
        n_heads,
        n_kv_heads,
        shared_classifier,
        ..
    } = *config;

    let all_heads_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // Helper macro for reading f32 arrays with context
    macro_rules! read_f32_weights {
        ($size:expr, $field:literal) => {
            // SAFETY: we keep the mmap alive for the lifetime of the transformer
            unsafe {
                std::mem::transmute::<&[f32], &[f32]>(
                    mapper.get_f32_slice($size).with_context(|| format!("Failed to read {}", $field))?,
                )
            }
        };
    }

    let rms_att_weight = read_f32_weights!(n_layers * dim, "attention normalization weights");
    let rms_ffn_weight = read_f32_weights!(n_layers * dim, "FFN normalization weights");
    let rms_final_weight = read_f32_weights!(dim, "final normalization weights");
    let q_ln_weights = read_f32_weights!(n_layers * head_dim, "query layer norm weights");
    let k_ln_weights = read_f32_weights!(n_layers * head_dim, "key layer norm weights");

    // Read quantized tensors
    let q_tokens = create_quantized_tensors(mapper, 1, vocab_size * dim, group_size)?
        .into_iter()
        .next()
        .expect("Expected exactly one token embedding tensor");

    // Dequantize token embeddings (we need owned data for this)
    let mut token_embedding_table = vec![0.0; vocab_size * dim];
    dequantize(&q_tokens, &mut token_embedding_table, group_size);

    let wq = create_quantized_tensors(mapper, n_layers, dim * all_heads_dim, group_size)?;
    let wk = create_quantized_tensors(mapper, n_layers, dim * kv_dim, group_size)?;
    let wv = create_quantized_tensors(mapper, n_layers, dim * kv_dim, group_size)?;
    let wo = create_quantized_tensors(mapper, n_layers, all_heads_dim * dim, group_size)?;
    let w1 = create_quantized_tensors(mapper, n_layers, dim * hidden_dim, group_size)?;
    let w2 = create_quantized_tensors(mapper, n_layers, hidden_dim * dim, group_size)?;
    let w3 = create_quantized_tensors(mapper, n_layers, dim * hidden_dim, group_size)?;

    let wcls = if shared_classifier {
        q_tokens.clone()
    } else {
        create_quantized_tensors(mapper, 1, dim * vocab_size, group_size)?
            .into_iter()
            .next()
            .expect("Expected exactly one classification tensor")
    };

    Ok(Qwen3TransformerWeights {
        token_embedding_table,
        rms_att_weight,
        rms_ffn_weight,
        wq,
        wk,
        wv,
        wo,
        q_ln_weights,
        k_ln_weights,
        w1,
        w2,
        w3,
        rms_final_weight,
        wcls,
    })
}

/// Reads multiple quantized tensors from memory mapper.
///
/// Each quantized tensor consists of:
/// 1. Quantized weights (i8 values)
/// 2. Scale factors (f32 values)
///
/// The scale factors are grouped according to the quantization group size.
fn create_quantized_tensors(
    mapper: &mut MemoryMapper,
    n_tensors: usize,
    size_each: usize,
    group_size: usize,
) -> Result<Vec<QuantizedTensor>> {
    (0..n_tensors)
        .map(|i| {
            // Read quantized values
            let q_bytes =
                mapper.get_bytes(size_each).with_context(|| format!("Failed to read quantized tensor {i} data"))?;

            // Convert bytes to i8 (avoiding copy by using unsafe)
            let q_slice = unsafe { std::slice::from_raw_parts(q_bytes.as_ptr() as *const i8, size_each) };

            // Calculate and read scale factors
            let s_len = size_each / group_size;
            let s_slice =
                mapper.get_f32_slice(s_len).with_context(|| format!("Failed to read scale factors for tensor {i}"))?;

            // Convert to 'static lifetime using unsafe transmute
            let q_static = unsafe { std::mem::transmute::<&[i8], &'static [i8]>(q_slice) };
            let s_static = unsafe { std::mem::transmute::<&[f32], &'static [f32]>(s_slice) };

            Ok(QuantizedTensor::from_slices(q_static, s_static))
        })
        .collect()
}

fn create_transformer_block(
    model_config: &ModelConfig,
    layer_idx: usize,
    weights: &Qwen3TransformerWeights,
) -> Result<Qwen3TransformerBlock> {
    let dim = model_config.dim;
    let head_dim = model_config.head_dim;
    let all_heads_dim = model_config.n_heads * head_dim;
    let kv_dim = model_config.n_kv_heads * head_dim;
    let hidden_dim = model_config.hidden_dim;
    let group_size = model_config.group_size;

    // Attention normalization
    let attn_norm_start = layer_idx * dim;
    let attn_norm = RMSNorm::new(weights.rms_att_weight[attn_norm_start..attn_norm_start + dim].to_vec());

    // Query/Key normalization
    let qk_norm_start = layer_idx * head_dim;
    let q_norm = RMSNorm::new(weights.q_ln_weights[qk_norm_start..qk_norm_start + head_dim].to_vec());
    let k_norm = RMSNorm::new(weights.k_ln_weights[qk_norm_start..qk_norm_start + head_dim].to_vec());

    // Attention projections
    let wq = Linear::new(weights.wq[layer_idx].clone(), dim, all_heads_dim, group_size);
    let wk = Linear::new(weights.wk[layer_idx].clone(), dim, kv_dim, group_size);
    let wv = Linear::new(weights.wv[layer_idx].clone(), dim, kv_dim, group_size);
    let wo = Linear::new(weights.wo[layer_idx].clone(), all_heads_dim, dim, group_size);

    let attention = MultiHeadAttention::new(wq, wk, wv, wo, q_norm, k_norm, model_config);

    // FFN normalization
    let ffn_norm_start = layer_idx * dim;
    let ffn_norm = RMSNorm::new(weights.rms_ffn_weight[ffn_norm_start..ffn_norm_start + dim].to_vec());

    // Feed-forward projections
    let w1 = Linear::new(weights.w1[layer_idx].clone(), dim, hidden_dim, group_size);
    let w2 = Linear::new(weights.w2[layer_idx].clone(), hidden_dim, dim, group_size);
    let w3 = Linear::new(weights.w3[layer_idx].clone(), dim, hidden_dim, group_size);

    let feed_forward = FeedForward::new(w1, w2, w3);

    Ok(Qwen3TransformerBlock::new(attn_norm, attention, ffn_norm, feed_forward, layer_idx))
}

/// Contains all the learned parameters for the transformer model.
///
/// This structure holds both quantized weights (for memory efficiency) and
/// pre-computed values like the dequantized token embedding table.
#[derive(Debug)]
struct Qwen3TransformerWeights {
    /// Pre-dequantized token embedding table for fast lookup during inference
    /// Shape: [vocab_size, dim]
    pub token_embedding_table: Vec<f32>,

    /// RMS normalization weights for attention layers
    /// Shape: [n_layers, dim] (flattened)
    pub rms_att_weight: &'static [f32],

    /// RMS normalization weights for feed-forward layers
    /// Shape: [n_layers, dim] (flattened)
    pub rms_ffn_weight: &'static [f32],

    /// Attention projection weights (quantized for memory efficiency)
    /// Query projections: [n_layers] × [dim, n_heads * head_dim]
    pub wq: Vec<QuantizedTensor>,
    /// Key projections: [n_layers] × [dim, n_kv_heads * head_dim]
    pub wk: Vec<QuantizedTensor>,
    /// Value projections: [n_layers] × [dim, n_kv_heads * head_dim]
    pub wv: Vec<QuantizedTensor>,
    /// Output projections: [n_layers] × [n_heads * head_dim, dim]
    pub wo: Vec<QuantizedTensor>,

    /// QK-RMSNorm weights for Qwen3 architecture
    /// Query layer norm: [n_layers, head_dim] (flattened)
    pub q_ln_weights: &'static [f32],
    /// Key layer norm: [n_layers, head_dim] (flattened)
    pub k_ln_weights: &'static [f32],

    /// Feed-forward network weights (quantized)
    /// Gate projection: [n_layers] × [dim, hidden_dim]
    pub w1: Vec<QuantizedTensor>,
    /// Down projection: [n_layers] × [hidden_dim, dim]
    pub w2: Vec<QuantizedTensor>,
    /// Up projection: [n_layers] × [dim, hidden_dim]
    pub w3: Vec<QuantizedTensor>,

    /// Final RMS normalization weight before classification
    /// Shape: [dim]
    pub rms_final_weight: &'static [f32],

    /// Classification head weights (may be shared with token embeddings)
    /// Shape: [dim, vocab_size]
    pub wcls: QuantizedTensor,
}
