//! Tokenizer for BPE-based language models.
//!
//! This module provides a simple byte-level BPE tokenizer that matches the behavior of C reference implementations.
//!
//! - Loads vocabulary and merge scores from a binary file.
//! - Encodes text into token IDs using special token and character lookup, then applies BPE merges.
//! - Decodes token IDs back to strings, handling both valid and invalid UTF-8.

use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use std::borrow::Cow;
use std::fs::File;
use std::io::Read;

/// Tokenizer for byte-level BPE models.
///
/// Holds the vocabulary, merge scores, and prompt templates.
/// Provides encode/decode methods for text and token IDs.
pub struct Tokenizer {
    /// Vocabulary: each token is a byte sequence (not necessarily valid UTF-8)
    pub vocab: Vec<Vec<u8>>, // Store raw bytes instead of Strings
    /// Merge scores for BPE merges (higher is better)
    pub merge_scores: Vec<f32>,
    /// Number of tokens in the vocabulary
    pub vocab_size: usize,
    /// Maximum token length (in bytes)
    pub max_token_length: u32,
    /// Beginning-of-sequence token ID
    pub bos_token_id: u32,
    /// End-of-sequence token ID
    pub eos_token_id: u32,
    /// Prompt template for user prompts
    pub prompt_template: String,
    /// Prompt template for system prompts
    pub system_prompt_template: String,
}

impl Tokenizer {
    /// Loads a tokenizer from a checkpoint path and vocabulary size.
    ///
    /// Reads the vocabulary, merge scores, and prompt templates from disk.
    pub fn new(checkpoint_path: &str, vocab_size: usize, enable_thinking: bool) -> Result<Self> {
        let tokenizer_path = format!("{checkpoint_path}.tokenizer");
        let file = File::open(&tokenizer_path)?;
        let mut reader = std::io::BufReader::new(file);

        // Read header: max token length, BOS/EOS token IDs
        let max_token_length = reader.read_u32::<LittleEndian>()?;
        let bos_token_id = reader.read_u32::<LittleEndian>()?;
        let eos_token_id = reader.read_u32::<LittleEndian>()?;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut merge_scores = Vec::with_capacity(vocab_size);

        // Read vocabulary: (score, length, bytes) for each token
        for _i in 0..vocab_size {
            // Read score
            let score = match reader.read_f32::<LittleEndian>() {
                Ok(s) => s,
                Err(_) => {
                    // If reading fails, push empty token and zero score
                    vocab.push(Vec::new());
                    merge_scores.push(0.0);
                    continue;
                }
            };
            merge_scores.push(score);

            // Read token length
            let len = match reader.read_u32::<LittleEndian>() {
                Ok(l) => l as usize,
                Err(_) => {
                    vocab.push(Vec::new());
                    continue;
                }
            };

            // Read token bytes
            let mut token_bytes = vec![0u8; len];
            match reader.read_exact(&mut token_bytes) {
                Ok(_) => vocab.push(token_bytes),
                Err(_) => vocab.push(Vec::new()),
            }
        }

        // Load prompt templates (for chat/instruction mode)
        let prompt_template = Self::load_prompt_template(checkpoint_path, false, enable_thinking)?;
        let system_prompt_template = Self::load_prompt_template(checkpoint_path, true, enable_thinking)?;

        Ok(Self {
            vocab,
            merge_scores,
            vocab_size,
            max_token_length,
            bos_token_id,
            eos_token_id,
            prompt_template,
            system_prompt_template,
        })
    }

    /// Loads a prompt template from disk, with support for system and "thinking" variants.
    fn load_prompt_template(checkpoint_path: &str, with_system: bool, enable_thinking: bool) -> Result<String> {
        let suffix = match (with_system, enable_thinking) {
            (true, true) => ".template.with-system-and-thinking",
            (true, false) => ".template.with-system",
            (false, true) => ".template.with-thinking",
            (false, false) => ".template",
        };

        let template_path = format!("{checkpoint_path}{suffix}");

        match std::fs::read_to_string(&template_path) {
            Ok(content) => Ok(content),
            Err(_) => {
                eprintln!("Warning: Could not load prompt template {template_path}");
                Ok(String::new())
            }
        }
    }

    /// Decodes a token ID to a string (may be invalid UTF-8).
    ///
    /// Returns a borrowed str if valid UTF-8, otherwise an owned String.
    pub fn decode(&self, token: usize) -> Cow<str> {
        if token < self.vocab.len() {
            // Try to interpret as valid UTF-8 first (no allocation needed)
            match std::str::from_utf8(&self.vocab[token]) {
                Ok(valid_str) => Cow::Borrowed(valid_str),
                Err(_) => {
                    // SAFETY: For incomplete UTF-8 sequences (like partial emoji bytes),
                    // we need to preserve the exact bytes. Use unsafe since we know
                    // these bytes come from a trusted tokenizer file and will be
                    // combined with other tokens to form valid UTF-8 during generation.
                    let string = unsafe { String::from_utf8_unchecked(self.vocab[token].clone()) };
                    Cow::Owned(string)
                }
            }
        } else {
            Cow::Borrowed("")
        }
    }

    /// Looks up a string in the vocabulary and returns its token ID, if present.
    fn str_lookup(&self, s: &str) -> Option<usize> {
        // Validate vocab_size matches actual vocab length (safety check)
        debug_assert_eq!(self.vocab.len(), self.vocab_size, "Vocab size mismatch");
        // Convert string to bytes and compare with vocab bytes
        let s_bytes = s.as_bytes();
        self.vocab.iter().position(|token| token.as_slice() == s_bytes)
    }

    /// Encodes a string into a sequence of token IDs using BPE.
    ///
    /// 1. Looks up special tokens (e.g., <bos>, <eos>) and single characters.
    /// 2. Applies BPE merges: repeatedly merges the pair of tokens with the highest merge score,
    ///    replacing them with the merged token, until no more merges are possible.
    /// 3. Returns the resulting token ID sequence.
    ///
    /// # Arguments
    /// * `text` - The input string to encode.
    ///
    /// # Returns
    /// A vector of token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut found_special = false;

            // Check for special tokens (use max_token_length for buffer bounds)
            if chars[i] == '<' {
                let mut end_pos = None;
                let search_limit = chars.len().min(i + self.max_token_length as usize);
                for j in i + 1..search_limit {
                    if chars[j] == '>' {
                        end_pos = Some(j);
                        break;
                    }
                }

                if let Some(end) = end_pos {
                    let special_token: String = chars[i..=end].iter().collect();
                    if let Some(token_id) = self.str_lookup(&special_token) {
                        tokens.push(token_id);
                        i = end + 1;
                        found_special = true;
                    }
                }
            }

            if !found_special {
                let char_str = chars[i].to_string();
                if let Some(token_id) = self.str_lookup(&char_str) {
                    tokens.push(token_id);
                } else {
                    // Print a warning for unknown characters (not present in vocab)
                    println!("Warning: unknown character '{}' in input, skipping.", chars[i]);
                }
                i += 1;
            }
        }

        // Merge tokens using BPE (Byte Pair Encoding)
        // Repeatedly merge the pair with the highest merge score until no merges remain.
        loop {
            let mut best_score = -1e10;
            let mut best_id = None;
            let mut best_idx = None;

            for i in 0..tokens.len().saturating_sub(1) {
                // Concatenate the raw bytes of the two tokens
                let mut merged_bytes = self.vocab[tokens[i]].clone();
                merged_bytes.extend_from_slice(&self.vocab[tokens[i + 1]]);

                if let Some(id) = self.vocab.iter().position(|token| token.as_slice() == merged_bytes.as_slice()) {
                    if self.merge_scores[id] > best_score {
                        best_score = self.merge_scores[id];
                        best_id = Some(id);
                        best_idx = Some(i);
                    }
                }
            }

            let (id, idx) = match (best_id, best_idx) {
                (Some(id), Some(idx)) => (id, idx),
                _ => break,
            };

            tokens[idx] = id;
            tokens.remove(idx + 1);
        }

        tokens
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bos_token = (self.bos_token_id, self.decode(self.bos_token_id as usize));
        let eos_token = (self.eos_token_id, self.decode(self.eos_token_id as usize));

        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size)
            .field("max_token_length", &self.max_token_length)
            .field("bos_token_id", &bos_token)
            .field("eos_token_id", &eos_token)
            .field("prompt_template", &self.prompt_template)
            .field("system_prompt_template", &self.system_prompt_template)
            .finish_non_exhaustive()
    }
}
