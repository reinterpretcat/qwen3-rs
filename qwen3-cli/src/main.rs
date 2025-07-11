use std::path::Path;

use anyhow::Result;
use clap::{Arg, ArgMatches, Command};
use log::{error, info};
use qwen3_export::{export_model, load_hf_config};
use qwen3_inference::{InferenceConfigBuilder, run_inference};

/// Define the export subcommand.
fn export_subcommand() -> Command {
    Command::new("export")
        .about("Export Qwen3 model from HuggingFace format to custom binary format")
        .arg(Arg::new("MODEL_PATH")
            .help("Path to the HuggingFace model directory (containing config.json, *.safetensors, tokenizer.json)")
            .required(true)
            .index(1))
        .arg(Arg::new("OUTPUT_PATH")
            .help("Output path for the binary model file (without extension)")
            .required(true)
            .index(2))
        .arg(Arg::new("group-size")
            .long("group-size")
            .short('g')
            .help("Quantization group size")
            .value_name("SIZE")
            .default_value("64"))
}

/// Define the inference subcommand.
fn inference_subcommand() -> Command {
    Command::new("inference")
        .about("Qwen3 inference in Rust")
        .arg(
            Arg::new("checkpoint")
                .help("Model checkpoint file")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("temperature")
                .short('t')
                .long("temperature")
                .value_name("FLOAT")
                .help("Temperature for sampling in [0, inf], default 1.0")
                .default_value("1.0")
                .value_parser(clap::value_parser!(f32)),
        )
        .arg(
            Arg::new("topp")
                .short('p')
                .long("topp")
                .value_name("FLOAT")
                .help("Top-p for nucleus sampling in [0,1], default 0.9")
                .default_value("0.9")
                .value_parser(clap::value_parser!(f32)),
        )
        .arg(
            Arg::new("seed")
                .short('s')
                .long("seed")
                .value_name("INT")
                .help("Random seed")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("context")
                .short('c')
                .long("context")
                .value_name("INT")
                .help("Context window size, (default) = max_seq_len")
                .value_parser(clap::value_parser!(u32)),
        )
        .arg(
            Arg::new("mode")
                .short('m')
                .long("mode")
                .value_name("STRING")
                .help("Mode: generate|chat [default: chat]")
                .default_value("chat"),
        )
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("STRING")
                .help("Input prompt"),
        )
        .arg(
            Arg::new("system")
                .short('y')
                .long("system")
                .value_name("STRING")
                .help("System prompt in chat mode"),
        )
        .arg(
            Arg::new("reasoning")
                .short('r')
                .long("reasoning")
                .value_name("INT")
                .help("Reasoning mode: 0=no thinking, 1=thinking [default: 0]")
                .default_value("0")
                .value_parser(clap::value_parser!(i32)),
        )
}

/// Run the export command with the provided arguments
fn run_export_command(matches: &ArgMatches) -> Result<()> {
    let model_path = matches.get_one::<String>("MODEL_PATH").unwrap();
    let output_path = matches.get_one::<String>("OUTPUT_PATH").unwrap();
    let group_size: usize = matches
        .get_one::<String>("group-size")
        .unwrap()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid group size"))?;

    // Validate input path
    let model_dir = Path::new(model_path);
    if !model_dir.exists() {
        anyhow::bail!("Model directory does not exist: {model_path}");
    }

    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        anyhow::bail!("config.json not found in model directory")
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("tokenizer.json not found in model directory");
    }

    // Check for safetensors files
    let has_safetensors = std::fs::read_dir(model_dir)?.any(|entry| {
        if let Ok(entry) = entry {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        } else {
            false
        }
    });

    if !has_safetensors {
        anyhow::bail!("No .safetensors files found in model directory");
    }

    info!("");
    info!("🚀 Qwen3 Model Exporter");
    info!("📁 Model path: {model_path}");
    info!("💾 Output path: {output_path}");
    info!("🔢 Group size: {group_size}\n");

    // Load model configuration
    info!("Loading model configuration...");
    let config = load_hf_config(model_path)?;

    // Create exporter and run the export
    export_model(model_path, output_path, config, group_size)?;

    Ok(())
}

/// Run the inference command with the provided arguments
fn run_inference_command(matches: &ArgMatches) -> Result<()> {
    let config = InferenceConfigBuilder::default()
        .checkpoint_path(matches.get_one::<String>("checkpoint"))
        .temperature(matches.get_one::<f32>("temperature").copied())
        .topp(matches.get_one::<f32>("topp").copied())
        .ctx_length(matches.get_one::<usize>("context").copied())
        .mode(matches.get_one::<String>("mode"))
        .prompt(matches.get_one::<String>("input"))
        .system_prompt(matches.get_one::<String>("system"))
        .enable_thinking(matches.get_one::<i32>("reasoning").map(|v| *v != 0))
        .seed(matches.get_one::<u64>("seed").copied())
        .build()
        .map_err(|e| anyhow::anyhow!(e))?;

    run_inference(config).map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;

    Ok(())
}

fn execute_commands() -> Result<()> {
    // Initialize logger with clean format (no timestamp/module prefix)
    env_logger::Builder::from_default_env()
        .format(|buf, record| {
            use std::io::Write;
            writeln!(buf, "{}", record.args())
        })
        .init();

    let matches = Command::new("qwen3")
        .about("Qwen3 CLI: an educational tool for exporting and running Qwen3 models")
        .subcommand(export_subcommand())
        .subcommand(inference_subcommand())
        .get_matches();

    match matches.subcommand() {
        Some(("export", matches)) => run_export_command(matches),
        Some(("inference", matches)) => run_inference_command(matches),
        _ => anyhow::bail!("No subcommand specified. Use -h to print help information."),
    }
}

fn main() {
    if let Err(e) = execute_commands() {
        error!("Error: {e}");
        std::process::exit(1);
    }
}
