use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;
use anyhow::Result;
use std::io::{self, Write};
use std::time::Instant;

pub fn generate(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: Option<&str>,
) -> Result<()> {
    let prompt = prompt.unwrap_or("");
    let prompt_tokens = tokenizer.encode(prompt);

    if prompt_tokens.is_empty() {
        anyhow::bail!("Please provide a prompt");
    }

    let seq_len = transformer.config.seq_len as usize;
    let mut state = GenerationState::new(prompt_tokens[0]);

    while state.pos < seq_len {
        let next_token = if state.pos < prompt_tokens.len() - 1 {
            // Still processing prompt tokens
            prompt_tokens[state.pos + 1]
        } else {
            // Generate new tokens
            state.metrics.start_generation();
            let next = generate_next_token(transformer, sampler, state.token, state.pos)?;
            state.metrics.increment_token();

            if is_termination_token(next, tokenizer) {
                break;
            }
            next
        };

        output_token(tokenizer, state.token)?;
        state.advance(next_token);
    }

    state.metrics.report_and_reset();
    println!();
    Ok(())
}

pub fn chat(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    cli_user_prompt: Option<&str>,
    system_prompt: Option<&str>,
) -> Result<()> {
    let stdin = io::stdin();
    let seq_len = transformer.config.seq_len as usize;
    let mut state = GenerationState::new(0);
    let mut user_turn = true;
    let mut next_token = 0;

    loop {
        // Reset context if window exceeded
        if state.pos >= seq_len {
            state.reset(0);
            user_turn = true;
            println!();
        }

        if user_turn {
            state.metrics.report_and_reset();

            if !handle_user_turn(
                &stdin,
                transformer,
                tokenizer,
                sampler,
                &mut state,
                &mut next_token,
                cli_user_prompt,
                system_prompt,
            )? {
                break;
            }
            user_turn = false;
        } else {
            if handle_assistant_turn(
                transformer,
                tokenizer,
                sampler,
                &mut state,
                &mut next_token,
                &mut user_turn,
            )? {
                continue; // Turn ended, continue to next iteration
            }
        }
    }

    Ok(())
}

fn handle_user_turn(
    stdin: &io::Stdin,
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    state: &mut GenerationState,
    next_token: &mut usize,
    cli_user_prompt: Option<&str>,
    system_prompt: Option<&str>,
) -> Result<bool> {
    let user_prompt = get_user_input(stdin, state.pos, cli_user_prompt)?;

    // Check if we should exit
    if user_prompt.is_empty() && !(state.pos == 0 && cli_user_prompt.is_some()) {
        return Ok(false);
    }

    let rendered_prompt = render_prompt(state.pos, system_prompt, &user_prompt, tokenizer);
    let prompt_tokens = tokenizer.encode(&rendered_prompt);

    // Process prompt tokens
    for &token in &prompt_tokens {
        if state.pos >= transformer.config.seq_len as usize {
            break;
        }

        *next_token = generate_next_token(transformer, sampler, token, state.pos)?;
        state.advance(token);
    }

    Ok(true)
}

fn handle_assistant_turn(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    state: &mut GenerationState,
    next_token: &mut usize,
    user_turn: &mut bool,
) -> Result<bool> {
    if is_termination_token(*next_token, tokenizer) {
        state.metrics.report_and_reset();
        println!();
        *user_turn = true;
        return Ok(true);
    }

    state.metrics.start_generation();
    output_token(tokenizer, *next_token)?;

    *next_token = generate_next_token(transformer, sampler, *next_token, state.pos)?;
    state.metrics.increment_token();
    state.advance(*next_token);

    Ok(false)
}

fn generate_next_token(
    transformer: &mut Transformer,
    sampler: &mut Sampler,
    token: usize,
    pos: usize,
) -> Result<usize> {
    let logits = transformer.forward(token, pos);
    let mut logits_copy = logits.to_vec();
    Ok(sampler.sample(&mut logits_copy))
}

fn output_token(tokenizer: &Tokenizer, token: usize) -> Result<()> {
    print!("{}", tokenizer.decode(token));
    io::stdout().flush()?;
    Ok(())
}

fn is_termination_token(token: usize, tokenizer: &Tokenizer) -> bool {
    token == tokenizer.bos_token_id as usize || token == tokenizer.eos_token_id as usize
}

fn get_user_input(stdin: &io::Stdin, pos: usize, cli_user_prompt: Option<&str>) -> Result<String> {
    match (pos, cli_user_prompt) {
        (0, Some(prompt)) => Ok(prompt.to_string()),
        (_, Some(_)) => Ok(String::new()), // Signal to break
        _ => {
            print!("> ");
            io::stdout().flush()?;
            let mut input = String::new();
            stdin.read_line(&mut input)?;
            Ok(input.trim().to_string())
        }
    }
}

fn render_prompt(
    pos: usize,
    system_prompt: Option<&str>,
    user_prompt: &str,
    tokenizer: &Tokenizer,
) -> String {
    match (pos, system_prompt) {
        (0, Some(sys_prompt)) => tokenizer
            .system_prompt_template
            .replace("%s", &format!("{sys_prompt}\n{user_prompt}")),
        _ => tokenizer.prompt_template.replace("%s", user_prompt),
    }
}

/// Tracks token generation performance metrics
struct TokenMetrics {
    start_time: Option<Instant>,
    generated_count: usize,
}

impl TokenMetrics {
    fn new() -> Self {
        Self {
            start_time: None,
            generated_count: 0,
        }
    }

    fn start_generation(&mut self) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }
    }

    fn increment_token(&mut self) {
        self.generated_count += 1;
    }

    fn report_and_reset(&mut self) {
        if let Some(start_time) = self.start_time.take() {
            let duration = start_time.elapsed();
            if self.generated_count > 0 && duration.as_secs_f64() > 0.0 {
                let tps = self.generated_count as f64 / duration.as_secs_f64();
                println!(
                    "\n[Generated {} tokens in {:.2}s - {:.2} tokens/sec]",
                    self.generated_count,
                    duration.as_secs_f64(),
                    tps
                );
            }
        }
        self.generated_count = 0;
    }
}

/// Represents the current generation state
struct GenerationState {
    pos: usize,
    token: usize,
    metrics: TokenMetrics,
}

impl GenerationState {
    fn new(initial_token: usize) -> Self {
        Self {
            pos: 0,
            token: initial_token,
            metrics: TokenMetrics::new(),
        }
    }

    fn reset(&mut self, initial_token: usize) {
        self.metrics.report_and_reset();
        self.pos = 0;
        self.token = initial_token;
    }

    fn advance(&mut self, next_token: usize) {
        self.token = next_token;
        self.pos += 1;
    }
}
