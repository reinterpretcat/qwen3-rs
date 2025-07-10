# Temperature sampling

The **theoretical range for temperature is [0, ∞)**, but in practice, most useful values fall within a much smaller range.

Here's what's happening with different temperature values:

## Temperature Ranges and Effects

**Temperature = 0**: Greedy sampling (deterministic, always picks highest probability token)

**Temperature ∈ (0, 1)**:
- Makes the distribution more "sharp" (concentrates probability on high-scoring tokens)
- Values like 0.1-0.8 are common for more focused/conservative generation

**Temperature = 1**:
- No scaling applied (uses raw softmax probabilities)
- This is often considered the "neutral" baseline

**Temperature > 1**:
- Makes distribution more "flat" (spreads probability more evenly)
- Values like 1.2-2.0 can work for more creative/diverse generation
- Very high values (>3.0) tend to produce mostly random text

## Why Very High Temperatures Cause Issues

When temperature gets very large (say, >5.0), the logits become very small after division, and the softmax essentially becomes uniform. This means you're sampling almost randomly from your vocabulary, which produces nonsensical text.

## Practical Recommendations

Most practitioners use temperature values in these ranges:
- **Conservative/Focused**: 0.1 - 0.7
- **Balanced**: 0.7 - 1.2
- **Creative/Diverse**: 1.2 - 2.0
- **Experimental**: 2.0 - 3.0
