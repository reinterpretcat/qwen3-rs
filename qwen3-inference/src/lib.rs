//! Stub crate for Qwen3 inference code.
//!
//! This crate will provide inference functionality for Qwen3 models in the future.

mod configuration;
mod generation;
mod sampler;
mod tensor;
mod tokenizer;
mod transformer;
mod utils;

use anyhow::Result;
use log::debug;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::generation::{chat, generate};
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::transformer::TransformerBuilder;

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub checkpoint_path: String,
    pub temperature: f32,
    pub topp: f32,
    pub ctx_length: Option<usize>,
    pub mode: String,
    pub prompt: Option<String>,
    pub system_prompt: Option<String>,
    pub enable_thinking: bool,
    pub seed: u64,
}

impl InferenceConfig {
    pub fn builder() -> InferenceConfigBuilder {
        InferenceConfigBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct InferenceConfigBuilder {
    checkpoint_path: Option<String>,
    temperature: Option<f32>,
    topp: Option<f32>,
    ctx_length: Option<usize>,
    mode: Option<String>,
    prompt: Option<String>,
    system_prompt: Option<String>,
    enable_thinking: Option<bool>,
    seed: Option<u64>,
}

impl InferenceConfigBuilder {
    pub fn checkpoint_path(mut self, path: Option<&String>) -> Self {
        self.checkpoint_path = path.cloned();
        self
    }
    pub fn temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }
    pub fn topp(mut self, topp: Option<f32>) -> Self {
        self.topp = topp;
        self
    }
    pub fn ctx_length(mut self, ctx_length: Option<usize>) -> Self {
        self.ctx_length = ctx_length;
        self
    }
    pub fn mode(mut self, mode: Option<&String>) -> Self {
        self.mode = mode.cloned();
        self
    }
    pub fn prompt(mut self, prompt: Option<&String>) -> Self {
        self.prompt = prompt.cloned();
        self
    }
    pub fn system_prompt(mut self, system_prompt: Option<&String>) -> Self {
        self.system_prompt = system_prompt.cloned();
        self
    }
    pub fn enable_thinking(mut self, enable: Option<bool>) -> Self {
        self.enable_thinking = enable;
        self
    }
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
    pub fn build(self) -> Result<InferenceConfig, String> {
        Ok(InferenceConfig {
            checkpoint_path: self.checkpoint_path.ok_or("checkpoint_path is required")?,
            temperature: self.temperature.unwrap_or(1.0),
            topp: self.topp.unwrap_or(0.9),
            ctx_length: self.ctx_length,
            mode: self.mode.unwrap_or_else(|| "chat".to_string()),
            prompt: self.prompt,
            system_prompt: self.system_prompt,
            enable_thinking: self.enable_thinking.unwrap_or(false),
            seed: self.seed.unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            }),
        })
    }
}

/// Runs inference.
pub fn run_inference(inference_config: InferenceConfig) -> Result<()> {
    debug!("{inference_config:#?}");

    let mut transformer = TransformerBuilder::new(&inference_config.checkpoint_path)
        .with_ctx_length(inference_config.ctx_length)
        .build()?;

    debug!("{transformer:#?}");

    let transformer_config = transformer.get_config();

    let tokenizer = Tokenizer::new(
        &inference_config.checkpoint_path,
        transformer_config.vocab_size,
        inference_config.enable_thinking,
    )?;

    debug!("{tokenizer:#?}");

    let mut sampler = Sampler::new(
        transformer_config.vocab_size,
        inference_config.temperature,
        inference_config.topp,
        inference_config.seed,
    );

    let prompt = inference_config.prompt.as_deref();
    let system_prompt = inference_config.system_prompt.as_deref();

    // Run
    match inference_config.mode.as_str() {
        "generate" => generate(&mut transformer, &tokenizer, &mut sampler, prompt),
        "chat" => chat(
            &mut transformer,
            &tokenizer,
            &mut sampler,
            prompt,
            system_prompt,
        ),
        _ => anyhow::bail!("Unknown mode: {inference_config:?}"),
    }
}
