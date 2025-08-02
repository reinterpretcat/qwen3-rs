use anyhow::{Context, Result};
use log::info;
use serde_json::Value;
use std::path::Path;

/// Chat template exporter for model chat templates
///
/// This module creates chat templates based on the model's Jinja2 template
/// from tokenizer_config.json. It generates templates that match the exact
/// output of the Python reference implementation.
///
/// Supported Templates:
/// - Qwen3: Uses `<|im_start|>role\ncontent<|im_end|>\n` format with thinking support
/// - DeepSeek-R1: Uses `<｜User｜>content<｜Assistant｜>` format without thinking
///
/// Generated Templates:
/// - `.template`: Single user message with thinking disabled
/// - `.template.with-thinking`: Single user message with thinking enabled (if supported)
/// - `.template.with-system`: System + user messages with thinking disabled
/// - `.template.with-system-and-thinking`: System + user messages with thinking enabled (if supported)
#[derive(Debug)]
pub struct ChatTemplateExporter;

/// Template configuration for different chat formats
#[derive(Debug, Clone)]
pub struct TemplateConfig {
    pub suffix: &'static str,
    pub description: &'static str,
    pub enable_thinking: bool,
    pub has_system: bool,
}

/// Template capabilities detected from the Jinja2 template
#[derive(Debug)]
struct TemplateCapabilities {
    supports_thinking: bool,
    supports_system: bool,
    template_type: TemplateType,
}

/// Different template types we support
#[derive(Debug, Clone)]
enum TemplateType {
    Qwen3,
    DeepSeek,
    Unknown,
}

impl ChatTemplateExporter {
    // Template suffixes
    const BASIC_SUFFIX: &'static str = ".template";
    const WITH_THINKING_SUFFIX: &'static str = ".template.with-thinking";
    const WITH_SYSTEM_SUFFIX: &'static str = ".template.with-system";
    const WITH_SYSTEM_THINKING_SUFFIX: &'static str = ".template.with-system-and-thinking";

    // Template constants
    const TEMPLATE_EXTENSION: &'static str = ".template";

    /// Create a new ChatTemplateExporter
    pub fn new() -> Self {
        Self
    }

    /// Export chat templates to the specified output path
    pub fn export_templates(&self, model_path: &Path, output_path: &Path) -> Result<()> {
        // Load chat template from tokenizer config - this is now required
        let chat_template = self
            .load_chat_template_from_model(model_path)
            .with_context(|| format!("Failed to load chat template from model at {}", model_path.display()))?
            .ok_or_else(|| {
                anyhow::anyhow!("No chat template found in tokenizer_config.json at {}", model_path.display())
            })?;

        // Analyze template capabilities
        let capabilities = self.analyze_template_capabilities(&chat_template);
        info!("Template type: {:?}", capabilities.template_type);
        info!("Template capabilities:");
        info!("   Supports thinking: {}", capabilities.supports_thinking);
        info!("   Supports system: {}", capabilities.supports_system);

        self.export_dynamic_templates(output_path, &chat_template, &capabilities)
            .with_context(|| format!("Failed to export templates for model at {}", model_path.display()))
    }

    /// Analyze template to determine what capabilities it supports
    /// A very basic analysis to detect template type and capabilities
    fn analyze_template_capabilities(&self, template: &str) -> TemplateCapabilities {
        let template_type = if template.contains("<|im_start|>") && template.contains("<|im_end|>") {
            TemplateType::Qwen3
        } else if template.contains("<｜User｜>") && template.contains("<｜Assistant｜>") {
            TemplateType::DeepSeek
        } else {
            TemplateType::Unknown
        };

        let (supports_thinking, supports_system) = match template_type {
            TemplateType::Qwen3 => (
                template.contains("enable_thinking"),
                template.contains("system") && template.contains("messages[0].role"),
            ),
            TemplateType::DeepSeek => (template.contains("think"), template.contains("system_prompt")),
            TemplateType::Unknown => (false, false),
        };

        TemplateCapabilities { supports_thinking, supports_system, template_type }
    }

    /// Get template configurations based on detected capabilities
    fn get_template_configs(&self, capabilities: &TemplateCapabilities) -> Vec<TemplateConfig> {
        // Maximum possible templates: basic + thinking + system + system_thinking = 4
        let mut configs = Vec::with_capacity(4);

        // Always generate basic user template
        configs.push(TemplateConfig {
            suffix: Self::BASIC_SUFFIX,
            description: "basic",
            enable_thinking: false,
            has_system: false,
        });

        // Add thinking variant if supported
        if capabilities.supports_thinking {
            configs.push(TemplateConfig {
                suffix: Self::WITH_THINKING_SUFFIX,
                description: "with thinking",
                enable_thinking: true,
                has_system: false,
            });
        }

        // Add system variants if supported
        if capabilities.supports_system {
            configs.push(TemplateConfig {
                suffix: Self::WITH_SYSTEM_SUFFIX,
                description: "with system",
                enable_thinking: false,
                has_system: true,
            });

            // Only add system + thinking if both are supported
            if capabilities.supports_thinking {
                configs.push(TemplateConfig {
                    suffix: Self::WITH_SYSTEM_THINKING_SUFFIX,
                    description: "with system and thinking",
                    enable_thinking: true,
                    has_system: true,
                });
            }
        }

        configs
    }

    /// Load chat template from model's tokenizer_config.json
    fn load_chat_template_from_model(&self, model_path: &Path) -> Result<Option<String>> {
        let tokenizer_config_path = model_path.join("tokenizer_config.json");

        if !tokenizer_config_path.exists() {
            return Ok(None);
        }

        let config_content = std::fs::read_to_string(&tokenizer_config_path)
            .with_context(|| format!("Failed to read tokenizer config from {}", tokenizer_config_path.display()))?;

        let config: Value =
            serde_json::from_str(&config_content).with_context(|| "Failed to parse tokenizer config JSON")?;

        // Extract chat_template if it exists
        Ok(config.get("chat_template").and_then(|v| v.as_str()).map(|s| s.to_string()))
    }

    /// Export templates using the dynamic chat template from the model
    fn export_dynamic_templates(
        &self,
        output_path: &Path,
        chat_template: &str,
        capabilities: &TemplateCapabilities,
    ) -> Result<()> {
        info!("Using model's chat template for template generation");

        // Create template variants based on detected capabilities
        let template_configs = self.get_template_configs(capabilities);

        info!("Generating {} template variants:", template_configs.len());
        for config in &template_configs {
            info!("   - {}", config.description);
        }

        template_configs.iter().try_for_each(|config| {
            let template_content = self.render_chat_template(chat_template, config, capabilities)?;
            let template_path = format!("{}{}", output_path.display(), config.suffix);

            std::fs::write(&template_path, template_content)
                .with_context(|| format!("Failed to write template to {template_path}"))?;

            info!("📝 Written {} template: {template_path}", config.description);
            Ok::<(), anyhow::Error>(())
        })?;

        info!("💬 All prompt templates written to {}{}.*", output_path.display(), Self::TEMPLATE_EXTENSION);
        Ok(())
    }

    /// Render chat template for specific configuration
    /// This is a simplified Jinja2 template renderer for different template types
    fn render_chat_template(
        &self,
        _template: &str,
        config: &TemplateConfig,
        capabilities: &TemplateCapabilities,
    ) -> Result<String> {
        match capabilities.template_type {
            TemplateType::Qwen3 => {
                if config.has_system {
                    self.render_qwen3_system_message_template(config.enable_thinking)
                } else {
                    self.render_qwen3_single_message_template(config.enable_thinking)
                }
            }
            TemplateType::DeepSeek => {
                if config.has_system {
                    self.render_deepseek_system_message_template(config.enable_thinking)
                } else {
                    self.render_deepseek_single_message_template(config.enable_thinking)
                }
            }
            TemplateType::Unknown => Err(anyhow::anyhow!("Unknown template type, cannot render templates")),
        }
    }

    /// Render Qwen3 template for single user message
    fn render_qwen3_single_message_template(&self, enable_thinking: bool) -> Result<String> {
        if enable_thinking {
            Ok("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n".to_string())
        } else {
            Ok("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_string())
        }
    }

    /// Render Qwen3 template for system + user messages
    fn render_qwen3_system_message_template(&self, enable_thinking: bool) -> Result<String> {
        if enable_thinking {
            Ok("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n".to_string())
        } else {
            Ok("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_string())
        }
    }

    /// Render DeepSeek template for single user message
    fn render_deepseek_single_message_template(&self, enable_thinking: bool) -> Result<String> {
        if enable_thinking {
            Ok("<｜User｜>%s<｜Assistant｜>".to_string())
        } else {
            Ok("<｜User｜>%s<｜Assistant｜><think>\n</think>".to_string())
        }
    }

    /// Render DeepSeek template for system + user messages
    fn render_deepseek_system_message_template(&self, enable_thinking: bool) -> Result<String> {
        if enable_thinking {
            Ok("%s<｜User｜>%s<｜Assistant｜>".to_string())
        } else {
            Ok("%s<｜User｜>%s<｜Assistant｜><think>\n</think>".to_string())
        }
    }
}

impl Default for ChatTemplateExporter {
    fn default() -> Self {
        Self::new()
    }
}
