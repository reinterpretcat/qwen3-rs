use anyhow::Result;

use crate::{ModelInfo, models::qwen3::Qwen3, tensor_reader::TensorReader};

mod qwen3;

/// Architecture ID for binary format identification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ArchitectureId {
    Qwen3ForCausalLM = 1,
    LlamaForCausalLM = 2,
}

impl TryFrom<&str> for ArchitectureId {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "Qwen3ForCausalLM" => Ok(Self::Qwen3ForCausalLM),
            "LlamaForCausalLM" => Ok(Self::LlamaForCausalLM),
            _ => anyhow::bail!("Unknown ArchitectureId: {value}"),
        }
    }
}

impl TryFrom<u32> for ArchitectureId {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::Qwen3ForCausalLM),
            2 => Ok(Self::LlamaForCausalLM),
            _ => anyhow::bail!("Unknown ArchitectureId: {value}"),
        }
    }
}

/// Header information structure (lightweight)
#[derive(Debug, Clone)]
pub struct HeaderInfo {
    pub architecture_id: u32,
    pub shared_classifier: bool,
}

/// Represents normalization layer.
pub struct NormWeightLayer<'a> {
    /// Name of the layer
    pub name: &'a str,
    /// If set to true, name is a pattern parametrized with layer index
    pub layered: bool,
    /// If true, error will be returned if the layer not found
    /// Otherwise, default(1.0) value will be set.
    pub is_required: bool,
}

impl<'a> NormWeightLayer<'a> {
    pub const fn new(pattern: &'a str, layered: bool, is_required: bool) -> Self {
        Self { name: pattern, layered, is_required }
    }
}

pub struct WeightLayer<'a> {
    pub tensor_name: String,
    pub component: &'a str,
    pub layer_idx: u32,
}

impl<'a> WeightLayer<'a> {
    pub fn new(tensor_name: String, component: &'a str, layer_idx: u32) -> Self {
        Self { tensor_name, component, layer_idx }
    }
}

pub trait Architecture {
    fn id(&self) -> ArchitectureId;

    fn name(&self) -> &'static str;

    fn header(&self) -> Result<HeaderInfo>;

    fn norm_weight_layers(&self) -> &[NormWeightLayer<'_>];

    fn embed_tokens_layer(&self) -> &'static str;

    fn lm_head_layer(&self) -> &'static str;

    fn weight_layers(&self) -> &[WeightLayer<'_>];
}

pub fn create_architecture<'a>(model_info: &ModelInfo, tensor_reader: &'a TensorReader) -> Box<dyn Architecture + 'a> {
    match model_info.config.architecture {
        ArchitectureId::Qwen3ForCausalLM => Box::new(Qwen3::new(model_info, tensor_reader)),
        ArchitectureId::LlamaForCausalLM => todo!("LlamaForCausalLM not yet implemented"),
    }
}
