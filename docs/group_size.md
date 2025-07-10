Group size is a critical hyperparameter in quantization that affects both accuracy and performance. Let's explain how it's typically determined.

## **Common Group Size Values**

```rust
// Typical group sizes in practice:
let common_group_sizes = [32, 64, 128, 256, 512, 1024];

// Most popular choices:
let group_size = 128;  // Good balance for most models
let group_size = 256;  // Your code's likely default
```

---

## **Factors That Determine Group Size**

### **1. Model Architecture Constraints**

In the code we have this consideration:

```rust
pub fn new(config: ModelConfig, mut group_size: usize) -> Self {
    // Adjust group size to fit hidden_dim
    while config.dim % group_size as i32 != 0 {
        group_size /= 2;  // Keep halving until it divides evenly
    }
    Self { config, group_size }
}
```

**Why this matters:**
```rust
// Example: LLaMA model
let hidden_dim = 4096;  // Must divide evenly

// Good group sizes: 32, 64, 128, 256, 512, 1024, 2048, 4096
// Bad group sizes:  100, 300, 500 (don't divide evenly)

// The code automatically fixes this:
let mut group_size = 300;  // Bad choice
// After adjustment: 300 → 150 → 75 → 37 → 18 → 9 → 4
// Final group_size = 4 (algorithm can be improved)
```

### **2. Accuracy vs Compression Trade-off**

```rust
// Smaller groups = Better accuracy, More overhead
let small_groups = 32;   // High precision, more scales to store

// Larger groups = Less accuracy, Less overhead
let large_groups = 1024; // Lower precision, fewer scales
```

**Example with real numbers:**

```rust
// Tensor: [4096] weights
// Group size 32:  4096/32 = 128 groups → 128 scales (512 bytes overhead)
// Group size 256: 4096/256 = 16 groups → 16 scales (64 bytes overhead)
// Group size 1024: 4096/1024 = 4 groups → 4 scales (16 bytes overhead)
```

### **3. Hardware Optimization**

```rust
// Modern CPUs prefer certain sizes for vectorization
let cpu_friendly = [32, 64, 128, 256];  // Align with SIMD instructions

// GPU memory coalescing (if using GPU)
let gpu_friendly = [128, 256, 512];     // Align with warp/wavefront sizes
```

---

## **How Group Size Affects Quality**

Let's show with a concrete example:

```rust
// Example weight tensor with mixed scales
let weights = [
    // Large values section
    5.0, -4.2, 3.8, -3.1, 4.5, -2.9, 3.2, -4.1,
    // Small values section
    0.01, -0.008, 0.012, -0.015, 0.009, -0.011, 0.007, -0.013
];
```

### **Large Group Size (group_size = 16, one group):**
```rust
let max_abs = 5.0;  // Dominated by large values
let scale = 5.0 / 127.0 = 0.0394;

// Large values quantize well:
5.0 / 0.0394 = 127 ✓

// Small values lose precision:
0.01 / 0.0394 = 0.25 → 0  // Becomes zero! ❌
```

### **Small Group Size (group_size = 8, two groups):**
```rust
// Group 1: [5.0, -4.2, 3.8, -3.1, 4.5, -2.9, 3.2, -4.1]
let scale1 = 5.0 / 127.0 = 0.0394;

// Group 2: [0.01, -0.008, 0.012, -0.015, 0.009, -0.011, 0.007, -0.013]
let scale2 = 0.015 / 127.0 = 0.000118;

// Now small values preserve precision:
0.01 / 0.000118 = 85 ✓  // Good quantization!
```

---

## **Different Model Sizes**
```rust
// Small models (7B parameters)
let group_size = 64;    // Can afford smaller groups

// Medium models (13B-30B parameters)
let group_size = 128;   // Standard choice

// Large models (70B+ parameters)
let group_size = 256;   // Reduce memory overhead
```
