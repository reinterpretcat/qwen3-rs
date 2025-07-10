## 🔬 Model Architecture Summary

The project uses a **decoder-only Transformer** model with optimizations such as **Grouped Query Attention (GQA)**, **Rotary Position Embedding (RoPE)**, **INT8 Quantization**, and **RMSNorm**. The architecture closely follows Qwen3 and is designed for efficient autoregressive language modeling.

---

### 🧠 Layer Order in Forward Pass

1. **Token Embedding**
   - Converts discrete token IDs into continuous vector representations.
   - Shape: `[vocab_size, dim]`
   - Shared weights with `lm_head` if enabled.

2. **Transformer Blocks (N times)**
   Each block contains:
   - **Pre-Attention RMSNorm**: Normalizes input before attention.
     $$
     \text{RMSNorm}(x) = x \cdot \frac{\gamma}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}}
     $$
   - **Multi-Head Self-Attention with GQA and RoPE**
     $$
     \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
     $$
     - Queries are projected using `wq`, keys with `wk`, values with `wv`.
     - Keys/Queries are normalized via `QK-RMSNorm` and rotated via RoPE.
     - Grouped heads reduce memory usage.
   - **Residual Connection** after attention.
   - **Pre-FFN RMSNorm**: Same formula as above.
   - **SwiGLU Feed-Forward Network**
     $$
     \text{SwiGLU}(x) = \sigma(xW_1) \odot (xW_3) W_2
     $$
     where:
     - $ W_1 $: gate projection
     - $ W_3 $: up-projection
     - $ W_2 $: down-projection
     - $ \sigma $: sigmoid function
   - **Residual Connection** after FFN.

3. **Final RMSNorm**
   - Applied to the final hidden state before output.
     $$
     \text{RMSNorm}(x) = x \cdot \frac{\gamma}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}}
     $$

4. **Language Model Head (`lm_head`)**
   - Linear projection from hidden dimension to vocabulary size.
   - Optionally shares weights with `token_embedding`.

---

### 📐 Key Components & Formulas

#### 1. **RMSNorm (Root Mean Square Layer Normalization)**
$$
\text{RMSNorm}(x) = x \cdot \frac{\gamma}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}}
$$
- No mean subtraction, only variance normalization.
- Used before attention and FFN blocks.

#### 2. **Rotary Position Embedding (RoPE)**
- Encodes position information directly into query and key vectors.
- For each head and dimension pair:
  $$
  \text{freq} = \theta^{-2i/d}, \quad i \in [0, d/2)
  $$
  $$
  \text{angle} = pos \cdot freq
  $$
  $$
  \begin{bmatrix}
  x' \\
  y'
  \end{bmatrix}
  =
  \begin{bmatrix}
  \cos(\text{angle}) & -\sin(\text{angle}) \\
  \sin(\text{angle}) & \cos(\text{angle})
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y
  \end{bmatrix}
  $$

#### 3. **Multi-Head Attention (MHA)**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
- With **Grouped Query Attention (GQA)**:
  $$
  n_{kv\_heads} < n_{heads}
  $$
  Multiple query heads share the same key/value heads, reducing KV cache size.

#### 4. **SwiGLU Activation Function**
$$
\text{SwiGLU}(x) = \sigma(xW_1) \odot (xW_3) W_2
$$
- Combines gated activation with linear transformation.
- Enhances model expressiveness.

#### 5. **Quantized Linear Layers**
- Weights stored in INT8 format:
  $$
  w_{quantized} = \text{round}(w / s), \quad w = s \cdot w_{quantized}
  $$
- Dynamic dequantization during forward pass improves efficiency without much accuracy loss.

---

### 📦 PyTorch-like Module Hierarchy

```python
class Transformer(nn.Module):
    def __init__(self):
        self.token_embedding = TokenEmbedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, head_dim, hidden_dim, group_size)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = Linear(dim, vocab_size)

    def forward(self, token_id, pos):
        x = self.token_embedding(token_id)
        for block in self.blocks:
            x = block(x, pos)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
```

Each `TransformerBlock` includes:

```python
class TransformerBlock(nn.Module):
    def __init__():
        self.attn_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(...)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = FeedForward(...)  # SwiGLU-based
```

---

### ⚙️ Inference Flow (Mathematically)

```plaintext
Input Token ID → TokenEmbedding → N × TransformerBlock → Final RMSNorm → lm_head → Logits
```

Where each `TransformerBlock` does:

1. **Attention Path**
   - $ x_{norm} = \text{RMSNorm}(x) $
   - $ Q, K, V = x_{norm}W_q, x_{norm}W_k, x_{norm}W_v $
   - Apply RoPE to $ Q, K $
   - Compute $ A = \text{softmax}(QK^T/\sqrt{d_k})V $
   - $ x = x + A $

2. **Feed-Forward Path**
   - $ x_{norm} = \text{RMSNorm}(x) $
   - $ G = \sigma(x_{norm}W_1) $
   - $ U = x_{norm}W_3 $
   - $ F = (G \odot U)W_2 $
   - $ x = x + F $

---

### ✅ Summary of Optimizations

| Feature | Description | Benefit |
|--------|-------------|---------|
| **Grouped Query Attention (GQA)** | Reduces number of KV heads | Lowers memory usage |
| **Rotary Position Embedding (RoPE)** | Relative positional encoding | Better extrapolation |
| **RMSNorm** | Simplified normalization | Faster and more stable |
| **SwiGLU** | Gated non-linearity | Increased model capacity |
| **INT8 Quantization** | Stores weights in 8-bit integers | Saves memory, faster inference |

---

### Educational Insights

1. **Why RMSNorm?**
   - Removes mean-centering for faster computation
   - Works well when combined with residual connections
   - Original paper: https://arxiv.org/abs/1910.07467

2. **Why Rotary Embeddings?**
   - Relative positions handled naturally via rotation
   - No position embedding learned parameters
   - Original paper: https://arxiv.org/abs/2104.09864

3. **GQA Tradeoffs**
   - Memory: Reduces KV cache by `n_heads/n_kv_heads`
   - Quality: Minimal impact when ratio ≤ 8:1
   - Paper: https://arxiv.org/abs/2305.13245

4. **SwiGLU Benefits**
   - More parameters than standard FFN (W1,W3 vs single W1)
   - Better modeling of complex interactions
   - From PaLM paper: https://arxiv.org/abs/2204.02311

This architecture represents modern best practices for efficient LLM design, combining memory optimizations (GQA, quantization) with high-performance components (RoPE, SwiGLU).