# Description

**qwen3-rs** is an educational Rust project for exploring and running Qwen3 language family models. It is designed to be clear, modular, and approachable for learners, with minimal dependencies and many core algorithms reimplemented from scratch for transparency.

> **Note:** Parts of this codebase, including documentation and core algorithms, were generated or assisted by large language models (LLMs) to accelerate development and improve educational clarity. As a starting reference, the project [qwen3.c](https://github.com/adriancable/qwen3.c) was used for understanding model internals and file formats.


## Project Goals

- **Educational:** Learn how transformer architectures, quantization, and efficient inference work in Rust.
- **Minimal Dependencies:** Most algorithms (tokenization, quantization, sampling, etc.) are implemented from scratch—no heavy ML or Python bindings.
- **Modular:** Core library logic is separated from CLI tools for clarity and maintainability.
- **Efficiency:** Uses memory mapping and zero-copy techniques for handling large model files.

## Workspace Structure

```
qwen3-rs/
├── docs                # LLM generated docs for key components
├── Cargo.toml          # Workspace configuration
├── qwen3-cli/          # Command-line interface crate
├── qwen3-export/       # Model export crate
├── qwen3-inference/    # LLM inference crate
```

## How to Use

### 1. Get a HuggingFace Qwen3 model

```bash
git clone https://huggingface.co/Qwen/Qwen3-0.6B
# Or try larger/alternative models:
# git clone https://huggingface.co/Qwen/Qwen3-4B
# git clone https://huggingface.co/Qwen/Qwen3-8B
# git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

### 2. Build and run the exporter

```bash
cargo build --release -p qwen3-cli

# Export a HuggingFace model to quantized checkpoint format
cargo run --release -p qwen3-cli -- export /path/to/model /path/to/output.bin --group-size 64
```

### 3. Run inference

In chat mode with default parameters:

```bash
cargo run --release -p qwen3-cli -- inference /path/to/output.bin -m chat
```

## CLI Commands and Options

### `export`
Exports a HuggingFace Qwen3 model to a custom binary format for efficient Rust inference.

**Usage:**
```bash
qwen3 export <MODEL_PATH> <OUTPUT_PATH> [--group-size <SIZE>]
```
- `MODEL_PATH`: Path to HuggingFace model directory (must contain config.json, *.safetensors, tokenizer.json)
- `OUTPUT_PATH`: Output path for the binary model file
- `--group-size`, `-g`: Quantization group size (default: 64)

### `inference`
Runs inference on a binary Qwen3 model.

**Usage:**
```bash
qwen3 inference <checkpoint> [options]
```
**Options:**
- `--temperature`, `-t <FLOAT>`: Sampling temperature (default: 1.0)
- `--topp`, `-p <FLOAT>`: Top-p nucleus sampling (default: 0.9)
- `--seed`, `-s <INT>`: Random seed
- `--context`, `-c <INT>`: Context window size (default: max_seq_len)
- `--mode`, `-m <STRING>`: Mode: `generate` or `chat` (default: chat)
- `--input`, `-i <STRING>`: Input prompt
- `--system`, `-y <STRING>`: System prompt (for chat mode)
- `--reasoning`, `-r <INT>`: Reasoning mode: 0=no thinking, 1=thinking (default: 0)

