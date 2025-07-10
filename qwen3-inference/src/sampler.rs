use crate::transformer::softmax;

/// Stores a probability and its associated index (token id).
#[derive(Clone, Debug)]
pub struct ProbIndex {
    pub prob: f32,
    pub index: usize,
}

/// Top-p/temperature sampler for language model logits.
///
/// This struct implements temperature scaling, top-p (nucleus) sampling,
/// and multinomial sampling, using a simple xorshift RNG for reproducibility.
#[derive(Debug)]
pub struct Sampler {
    pub probindex: Vec<ProbIndex>,
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
}

impl Sampler {
    /// Creates a new sampler with the given vocabulary size, temperature, top-p, and RNG seed.
    ///
    /// # Arguments
    /// * `vocab_size` - Size of the vocabulary
    /// * `temperature` - Temperature for sampling (typical range: 0.1-2.0, 0.0 for greedy)
    /// * `topp` - Top-p threshold (0.0-1.0, 1.0 disables top-p)
    /// * `rng_seed` - Random seed for reproducibility
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        assert!(vocab_size > 0, "Vocab size must be positive");
        assert!(temperature >= 0.0, "Temperature must be non-negative");
        assert!(
            (0.0..=1.0).contains(&topp),
            "Top-p must be between 0.0 and 1.0"
        );

        Self {
            probindex: vec![
                ProbIndex {
                    prob: 0.0,
                    index: 0
                };
                vocab_size
            ],
            temperature,
            topp: topp.clamp(0.0, 1.0),
            rng_state: rng_seed,
        }
    }

    /// Xorshift-based random number generator.
    fn random_u32(&mut self) -> u32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        ((self.rng_state.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
    }

    /// Returns a random float in [0, 1).
    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16777216.0
    }

    /// Returns the index of the maximum logit (greedy decoding).
    fn sample_argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or_default()
    }

    /// Multinomial sampling from a probability distribution.
    fn sample_mult(logits: &[f32], coin: f32) -> usize {
        let mut cdf = 0.0;
        for (i, &prob) in logits.iter().enumerate() {
            cdf += prob;
            if coin < cdf {
                return i;
            }
        }
        logits.len().saturating_sub(1)
    }

    /// Top-p (nucleus) sampling: sample from the smallest set of tokens whose cumulative probability exceeds `topp`.
    fn sample_topp(&mut self, logits: &[f32], coin: f32) -> usize {
        let cutoff = (1.0 - self.topp) / (logits.len().saturating_sub(1).max(1)) as f32;
        let mut n0 = 0;

        // Collect candidates above cutoff
        for (i, &prob) in logits.iter().enumerate() {
            if prob >= cutoff {
                self.probindex[n0] = ProbIndex { prob, index: i };
                n0 += 1;
            }
        }

        // Sort by probability (descending)
        self.probindex[..n0].sort_unstable_by(|a, b| b.prob.total_cmp(&a.prob));

        // Find truncation point
        let mut cumulative_prob = 0.0;
        let mut last_idx = n0.saturating_sub(1);
        for i in 0..n0 {
            cumulative_prob += self.probindex[i].prob;
            if cumulative_prob > self.topp {
                last_idx = i;
                break;
            }
        }

        // Sample from truncated list
        let r = coin * cumulative_prob;
        let mut cdf = 0.0;
        for i in 0..=last_idx {
            cdf += self.probindex[i].prob;
            if r < cdf {
                return self.probindex[i].index;
            }
        }
        self.probindex[last_idx].index
    }

    /// Samples a token index from logits using temperature and top-p.
    ///
    /// - If temperature is 0, returns the argmax (greedy).
    /// - Otherwise, applies temperature scaling, softmax, and top-p or multinomial sampling.
    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            Self::sample_argmax(logits)
        } else {
            // Apply temperature
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }

            // Apply softmax
            softmax(logits);

            let coin = self.random_f32();

            if self.topp <= 0.0 || self.topp >= 1.0 {
                Self::sample_mult(logits, coin)
            } else {
                self.sample_topp(logits, coin)
            }
        }
    }
}
