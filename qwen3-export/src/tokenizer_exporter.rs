#[cfg(test)]
#[path = "../tests/unit/tokenizer_exporter_test.rs"]
mod tests;

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use log::info;
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Read, Write},
    path::Path,
};

/// Tokenizer exporter for converting HuggingFace tokenizers to binary format
#[derive(Debug)]
pub struct TokenizerExporter;

#[derive(Debug)]
struct TokenData {
    vocab: HashMap<String, u32>,
    merge_ranks: HashMap<String, usize>,
    max_token_length: u32,
}

/// GPT-2 style Unicode to byte mapping
#[derive(Debug)]
struct UnicodeToByteMap {
    mapping: HashMap<char, u8>,
}

impl UnicodeToByteMap {
    const PRINTABLE_ASCII_START: u8 = 33;
    const PRINTABLE_ASCII_END: u8 = 126;
    const EXTENDED_ASCII_START1: u8 = 161;
    const EXTENDED_ASCII_END1: u8 = 172;
    const EXTENDED_ASCII_START2: u8 = 174;
    const EXTENDED_ASCII_END2: u8 = 255;
    const UNICODE_OFFSET: u32 = 256;

    fn new() -> Self {
        let mut mapping = HashMap::new();

        // Printable ASCII characters
        for b in Self::PRINTABLE_ASCII_START..=Self::PRINTABLE_ASCII_END {
            mapping.insert(b as char, b);
        }

        // Extended ASCII characters
        for b in Self::EXTENDED_ASCII_START1..=Self::EXTENDED_ASCII_END1 {
            mapping.insert(b as char, b);
        }

        for b in Self::EXTENDED_ASCII_START2..=Self::EXTENDED_ASCII_END2 {
            mapping.insert(b as char, b);
        }

        // Special mappings for unprintable characters
        let mut n = 0u8;
        for b in 0..=255u8 {
            if !mapping.values().any(|&v| v == b) {
                mapping.insert(char::from_u32(Self::UNICODE_OFFSET + n as u32).unwrap(), b);
                n += 1;
            }
        }

        Self { mapping }
    }

    /// Convert token string to bytes using GPT-2 style mapping
    fn token_to_bytes(&self, token_str: &str) -> Vec<u8> {
        token_str
            .chars()
            .flat_map(|ch| {
                self.mapping.get(&ch).map(|&b| vec![b]).unwrap_or_else(|| ch.to_string().as_bytes().to_vec())
            })
            .collect()
    }
}

impl TokenizerExporter {
    const TOKENIZER_FILE_NAME: &'static str = "tokenizer.json";
    const DEFAULT_SCORE: f32 = -1e6;

    /// Create a new TokenizerExporter
    pub const fn new() -> Self {
        Self
    }

    /// Export tokenizer to binary format
    pub fn export_tokenizer(
        &self,
        model_path: &Path,
        output_path: &Path,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> Result<()> {
        let token_data = self.load_token_data(model_path)?;
        let tokens_by_id = self.create_ordered_tokens(&token_data.vocab);
        let u2b_map = UnicodeToByteMap::new();

        self.write_tokenizer_file(output_path, &token_data, &tokens_by_id, &u2b_map, bos_token_id, eos_token_id)
    }

    /// Load and process all token data
    fn load_token_data(&self, model_path: &Path) -> Result<TokenData> {
        let tokenizer_data = self.load_tokenizer_json(model_path)?;
        let vocab = self.extract_vocabulary(&tokenizer_data)?;

        let merge_ranks = self.extract_merge_ranks(&tokenizer_data);
        let max_token_length = vocab.keys().map(|token| token.len()).max().unwrap_or(0) as u32;

        info!("📊 Found {} tokens in vocabulary", vocab.len());

        Ok(TokenData { vocab, merge_ranks, max_token_length })
    }

    /// Load tokenizer.json file
    fn load_tokenizer_json(&self, model_path: &Path) -> Result<Value> {
        let tokenizer_path = model_path.join(Self::TOKENIZER_FILE_NAME);

        if !tokenizer_path.exists() {
            anyhow::bail!("tokenizer.json not found in model directory: {}", model_path.display());
        }

        let mut file = File::open(&tokenizer_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse tokenizer.json from {}", tokenizer_path.display()))
    }

    /// Create ordered list of tokens by ID
    fn create_ordered_tokens(&self, vocab: &HashMap<String, u32>) -> Vec<(u32, String)> {
        let mut tokens_by_id: Vec<(u32, String)> = vocab.iter().map(|(token, &id)| (id, token.clone())).collect();
        tokens_by_id.sort_by_key(|&(id, _)| id);
        tokens_by_id
    }

    /// Write tokenizer binary file
    fn write_tokenizer_file(
        &self,
        output_path: &Path,
        token_data: &TokenData,
        tokens_by_id: &[(u32, String)],
        u2b_map: &UnicodeToByteMap,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> Result<()> {
        let tokenizer_output = format!("{}.tokenizer", output_path.display());
        let file = File::create(&tokenizer_output)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_u32::<LittleEndian>(token_data.max_token_length)?;
        writer.write_u32::<LittleEndian>(bos_token_id)?;
        writer.write_u32::<LittleEndian>(eos_token_id)?;

        // Write tokens
        for (_, token) in tokens_by_id {
            self.write_token(&mut writer, token, &token_data.merge_ranks, u2b_map)?;
        }

        writer.flush()?;
        info!("💾 Written tokenizer model to {tokenizer_output}");
        Ok(())
    }

    /// Write a single token to the binary file
    fn write_token<W: Write>(
        &self,
        writer: &mut W,
        token: &str,
        merge_ranks: &HashMap<String, usize>,
        u2b_map: &UnicodeToByteMap,
    ) -> Result<()> {
        // Calculate pseudo-score
        let score = merge_ranks.get(token).map(|&rank| -((rank + 1) as f32).ln()).unwrap_or(Self::DEFAULT_SCORE);

        writer.write_f32::<LittleEndian>(score)?;

        // Convert token to bytes using GPT-2 style mapping
        let token_bytes = u2b_map.token_to_bytes(token);
        writer.write_u32::<LittleEndian>(token_bytes.len() as u32)?;
        writer.write_all(&token_bytes)?;

        Ok(())
    }

    /// Extract vocabulary from tokenizer data
    fn extract_vocabulary(&self, tokenizer_data: &Value) -> Result<HashMap<String, u32>> {
        // Extract vocabulary from model/vocab
        let mut vocab: HashMap<String, u32> =
            if let Some(vocab_obj) = tokenizer_data.pointer("/model/vocab").and_then(|v| v.as_object()) {
                vocab_obj.iter().filter_map(|(token, id)| id.as_u64().map(|id| (token.clone(), id as u32))).collect()
            } else {
                anyhow::bail!("Could not find vocabulary in tokenizer.json")
            };

        info!("📚 Found {} tokens in model/vocab", vocab.len());

        // Add tokens from added_tokens array
        if let Some(added_tokens) = tokenizer_data.pointer("/added_tokens").and_then(|v| v.as_array()) {
            for token_obj in added_tokens {
                if let (Some(content), Some(id)) = (
                    token_obj.pointer("/content").and_then(|v| v.as_str()),
                    token_obj.pointer("/id").and_then(|v| v.as_u64()),
                ) {
                    vocab.insert(content.to_string(), id as u32);
                }
            }

            info!("📝 Added {} tokens from added_tokens", added_tokens.len());
        }

        info!("📖 Total vocabulary size: {}", vocab.len());

        Ok(vocab)
    }

    /// Extract merge ranks from tokenizer data
    fn extract_merge_ranks(&self, tokenizer_data: &Value) -> HashMap<String, usize> {
        tokenizer_data
            .pointer("/model/merges")
            .and_then(|m| m.as_array())
            .map(|merges| {
                merges
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, merge)| merge.as_str().map(|merge_str| (merge_str.to_string(), rank)))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for TokenizerExporter {
    fn default() -> Self {
        Self::new()
    }
}
