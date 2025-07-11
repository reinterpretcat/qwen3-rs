#[cfg(test)]
#[path = "../tests/unit/tokenizer_exporter_test.rs"]
mod tests;

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use log::{info, warn};
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
                self.mapping
                    .get(&ch)
                    .map(|&b| vec![b])
                    .unwrap_or_else(|| ch.to_string().as_bytes().to_vec())
            })
            .collect()
    }
}

impl TokenizerExporter {
    const TOKENIZER_FILE_NAME: &'static str = "tokenizer.json";
    const TOKENIZER_CONFIG_FILE_NAME: &'static str = "tokenizer_config.json";
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
        let token_data = self.load_token_data(model_path, bos_token_id, eos_token_id)?;
        let tokens_by_id = self.create_ordered_tokens(&token_data.vocab);
        let u2b_map = UnicodeToByteMap::new();

        self.write_tokenizer_file(
            output_path,
            &token_data,
            &tokens_by_id,
            &u2b_map,
            bos_token_id,
            eos_token_id,
        )
    }

    /// Load and process all token data
    fn load_token_data(
        &self,
        model_path: &Path,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> Result<TokenData> {
        let tokenizer_data = self.load_tokenizer_json(model_path)?;
        let mut vocab = self.extract_vocabulary(&tokenizer_data)?;

        self.add_special_tokens_from_config(model_path, &mut vocab, bos_token_id, eos_token_id)?;

        let merge_ranks = self.extract_merge_ranks(&tokenizer_data);
        let max_token_length = vocab.keys().map(|token| token.len()).max().unwrap_or(0) as u32;

        info!("📊 Found {} tokens in vocabulary", vocab.len());

        Ok(TokenData {
            vocab,
            merge_ranks,
            max_token_length,
        })
    }

    /// Load tokenizer.json file
    fn load_tokenizer_json(&self, model_path: &Path) -> Result<Value> {
        let tokenizer_path = model_path.join(Self::TOKENIZER_FILE_NAME);

        if !tokenizer_path.exists() {
            anyhow::bail!(
                "tokenizer.json not found in model directory: {}",
                model_path.display()
            );
        }

        let mut file = File::open(&tokenizer_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        serde_json::from_str(&contents).with_context(|| {
            format!(
                "Failed to parse tokenizer.json from {}",
                tokenizer_path.display()
            )
        })
    }

    /// Create ordered list of tokens by ID
    fn create_ordered_tokens(&self, vocab: &HashMap<String, u32>) -> Vec<(u32, String)> {
        let mut tokens_by_id: Vec<(u32, String)> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();
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
        let score = merge_ranks
            .get(token)
            .map(|&rank| -((rank + 1) as f32).ln())
            .unwrap_or(Self::DEFAULT_SCORE);

        writer.write_f32::<LittleEndian>(score)?;

        // Convert token to bytes using GPT-2 style mapping
        let token_bytes = u2b_map.token_to_bytes(token);
        writer.write_u32::<LittleEndian>(token_bytes.len() as u32)?;
        writer.write_all(&token_bytes)?;

        Ok(())
    }

    /// Extract vocabulary from tokenizer data
    fn extract_vocabulary(&self, tokenizer_data: &Value) -> Result<HashMap<String, u32>> {
        // Try standard format first: model.vocab
        if let Some(vocab_obj) = tokenizer_data
            .pointer("/model/vocab")
            .and_then(|v| v.as_object())
        {
            return Ok(vocab_obj
                .iter()
                .filter_map(|(token, id)| id.as_u64().map(|id| (token.clone(), id as u32)))
                .collect());
        }

        // Try alternative format: direct vocab
        if let Some(vocab_obj) = tokenizer_data.pointer("/vocab").and_then(|v| v.as_object()) {
            return Ok(vocab_obj
                .iter()
                .filter_map(|(token, id)| id.as_u64().map(|id| (token.clone(), id as u32)))
                .collect());
        }

        anyhow::bail!("Could not find vocabulary in tokenizer.json")
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
                    .filter_map(|(rank, merge)| {
                        merge
                            .as_str()
                            .map(|merge_str| (merge_str.to_string(), rank))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Add special tokens from tokenizer_config.json that transformers library includes
    fn add_special_tokens_from_config(
        &self,
        model_path: &Path,
        vocab: &mut HashMap<String, u32>,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> Result<()> {
        let config_path = model_path.join(Self::TOKENIZER_CONFIG_FILE_NAME);

        if !config_path.exists() {
            warn!("tokenizer_config.json not found, skipping special tokens");
            return Ok(());
        }

        info!("Analyzing special tokens from tokenizer_config.json");

        let config_data = self.load_json_file(&config_path)?;

        // Add BOS and EOS tokens
        self.extract_and_add_token(&config_data, "bos_token", bos_token_id, "BOS", vocab);
        self.extract_and_add_token(&config_data, "eos_token", eos_token_id, "EOS", vocab);

        // Add other special tokens from added_tokens_decoder
        let added_count = config_data
            .pointer("/added_tokens_decoder")
            .and_then(|obj| obj.as_object())
            .map(|added_tokens_obj| {
                added_tokens_obj
                    .iter()
                    .filter_map(|(id_str, token_info)| {
                        let id = id_str.parse::<u32>().ok()?;
                        let content = token_info.pointer("/content")?.as_str()?;

                        // Only add if not already in vocab (avoid duplicates)
                        if !vocab.contains_key(content) {
                            vocab.insert(content.to_string(), id);
                            Some(())
                        } else {
                            None
                        }
                    })
                    .count()
            })
            .unwrap_or(0);

        if added_count > 0 {
            info!("🎯 Added {added_count} additional special tokens from added_tokens_decoder");
        }

        Ok(())
    }

    /// Extract token content and add it to vocabulary if not already present
    fn extract_and_add_token(
        &self,
        config_data: &Value,
        token_field: &str,
        token_id: u32,
        token_type: &str,
        vocab: &mut HashMap<String, u32>,
    ) {
        if token_id != 0 {
            info!("⚠️ {token_type} token ID is not 0 ({token_id}), skipping");
            return;
        }

        let token_path = format!("/{}", token_field);

        if let Some(token_value) = config_data.pointer(&token_path) {
            let token_content = token_value.as_str().map(str::to_owned).or_else(|| {
                token_value
                    .pointer("/content")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
            });

            if let Some(token_content) = token_content {
                match vocab.entry(token_content.clone()) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(token_id);
                        info!(
                            "🎯 Added {} token '{}' with ID {}",
                            token_type, token_content, token_id
                        );
                    }
                    std::collections::hash_map::Entry::Occupied(_) => {
                        info!(
                            "{} token '{}' already exists in vocabulary",
                            token_type, token_content
                        );
                    }
                }
            }
        }
    }

    /// Load and parse a JSON file
    fn load_json_file(&self, path: &Path) -> Result<Value> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse JSON from {}", path.display()))
    }
}

impl Default for TokenizerExporter {
    fn default() -> Self {
        Self::new()
    }
}
