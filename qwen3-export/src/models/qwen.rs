use crate::tensor_reader::TensorReader;

use super::*;

pub struct Qwen<'a> {
    weight_layers: Vec<WeightLayer<'a>>,
    tensor_reader: &'a TensorReader,
}

impl<'a> Qwen<'a> {
    const ARCH_NAME: &'static str = "Qwen3ForCausalLM";
    const EMBED_TOKENS_KEY: &'static str = "model.embed_tokens.weight";
    const LM_HEAD_KEY: &'static str = "lm_head.weight";

    #[rustfmt::skip]
    const NORM_WEIGHTS_LAYERS: &'static [NormWeightLayer<'static>] = &[
        NormWeightLayer::new("model.layers.{}.input_layernorm.weight", true, true),
        NormWeightLayer::new("model.layers.{}.post_attention_layernorm.weight", true, true),
        NormWeightLayer::new("model.norm.weight", false, true),
        NormWeightLayer::new("model.layers.{}.self_attn.q_norm.weight", true, false),
        NormWeightLayer::new("model.layers.{}.self_attn.k_norm.weight", true, false),
    ];

    // Qwen3 model layer weight component names (without .weight suffix)
    const QWEN3_LAYER_COMPONENTS: &'static [&'static str] = &[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
    ];

    pub fn new(model_info: &ModelInfo, tensor_reader: &'a TensorReader) -> Self {
        let weight_layers = Self::QWEN3_LAYER_COMPONENTS
            .iter()
            .flat_map(|&component| {
                (0..model_info.config.n_layers).map(move |layer_idx| {
                    let tensor_name = format!("model.layers.{}.{}.weight", layer_idx, component);
                    WeightLayer::new(tensor_name, component, layer_idx)
                })
            })
            .collect();

        Self {
            weight_layers,
            tensor_reader,
        }
    }
}

impl<'a> Architecture for Qwen<'a> {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::Qwen3ForCausalLM
    }

    fn name(&self) -> &'static str {
        Self::ARCH_NAME
    }

    fn header(&self) -> Result<HeaderInfo> {
        let shared_classifier = match (
            self.tensor_reader.load_tensor(Self::LM_HEAD_KEY)?,
            self.tensor_reader.load_tensor(Self::EMBED_TOKENS_KEY)?,
        ) {
            (Some(lm_head_weights), Some(embed_weights)) => {
                // Compare tensor values to determine if they're identical
                lm_head_weights.len() == embed_weights.len()
                    && lm_head_weights
                        .iter()
                        .zip(embed_weights.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-6)
            }
            (None, Some(_)) => true, // No lm_head means shared
            _ => false,              // Missing embed_tokens is an error, but we'll handle it later
        };

        Ok(HeaderInfo {
            architecture_id: self.id() as u32,
            shared_classifier,
        })
    }

    fn norm_weight_layers(&self) -> &[NormWeightLayer<'_>] {
        &Self::NORM_WEIGHTS_LAYERS
    }

    fn embed_tokens_layer(&self) -> &'static str {
        Self::EMBED_TOKENS_KEY
    }

    fn lm_head_layer(&self) -> &'static str {
        Self::LM_HEAD_KEY
    }

    fn weight_layers(&self) -> &[WeightLayer<'_>] {
        &self.weight_layers
    }
}
