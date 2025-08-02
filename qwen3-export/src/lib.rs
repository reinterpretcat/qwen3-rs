//! # qwen3-export
//!
//! A Rust library for exporting Qwen3 models from HuggingFace format to binary format.
//!
//! ## Examples
//!
//! ### Exporting a model
//!
//! ```rust,no_run
//! use qwen3_export::export_model;
//!
//! # fn main() -> anyhow::Result<()> {
//! let model_path = "path/to/huggingface/model";
//! let output_path = "output/model";
//!
//!
//! // Export the model
//! export_model(model_path, output_path, 32)?;
//! # Ok(())
//! # }
//! ```

// Public modules and re-exports from the former export module
pub mod chat_template_exporter;
pub mod config_loader;
pub mod lora_merger;
pub mod model_exporter;
pub mod models;
pub mod tensor_reader;
pub mod tokenizer_exporter;
mod utils;

// Re-export main types for easy access
pub use chat_template_exporter::ChatTemplateExporter;
pub use config_loader::{LoRAConfig, ModelConfig, ModelInfo, ModelType, load_model_info};
pub use model_exporter::{BinaryModelExporter, QuantizedWeight};
pub use tokenizer_exporter::TokenizerExporter;

use anyhow::Result;
use log::info;
use std::path::Path;

/// Export the model weights in Q8_0 into .bin file to be used by later within inference implementation.
/// That is:
/// - quantize all weights to symmetric int8, in range [-127, 127]
/// - all other tensors (the rmsnorm params) are kept and exported in fp32
/// - quantization is done in groups of group_size to reduce the effects of any outliers
///
/// Automatically detects model type and loads appropriate configs.
pub fn export_model(model_path: &str, output_path: &str, group_size: usize) -> Result<()> {
    info!("🚀 Starting automatic model export with type detection...");
    info!("");

    // Auto-detect model type and load configs
    let model_info = load_model_info(model_path)?;

    let model_path = Path::new(model_path);
    let output_path = Path::new(output_path);

    info!("🧮 Exporting quantized binary model...");
    BinaryModelExporter::new_from_model_info(&model_info, group_size).export_binary_model(
        model_path,
        output_path,
        &model_info,
    )?;
    info!("");

    info!("🔤 Exporting tokenizer...");
    TokenizerExporter::new().export_tokenizer(
        model_path,
        output_path,
        model_info.config.bos_token_id,
        model_info.config.eos_token_id,
    )?;
    info!("");

    info!("💬 Exporting chat templates...");
    ChatTemplateExporter::new().export_templates(model_path, output_path)?;

    info!("");
    info!("✅ Complete export finished successfully!");
    Ok(())
}
