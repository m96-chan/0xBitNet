//! Interactive chat CLI for oxbitnet.
//!
//! ```sh
//! cargo run --example chat --release
//! cargo run --example chat --release -- --url /path/to/model.gguf
//! cargo run --example chat --release -- --temperature 0.5 --max-tokens 1024
//! ```

use std::io::{self, BufRead, Write};
use std::time::Instant;

use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use oxbitnet::{BitNet, ChatMessage, GenerateOptions, LoadOptions, LoadProgress};

const DEFAULT_URL: &str =
    "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf";

struct Args {
    url: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    repeat_penalty: f32,
}

fn parse_args() -> Args {
    let mut args = Args {
        url: DEFAULT_URL.to_string(),
        max_tokens: 512,
        temperature: 0.7,
        top_k: 40,
        repeat_penalty: 1.1,
    };

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--url" => {
                i += 1;
                args.url = raw[i].clone();
            }
            "--max-tokens" => {
                i += 1;
                args.max_tokens = raw[i].parse().expect("invalid --max-tokens");
            }
            "--temperature" => {
                i += 1;
                args.temperature = raw[i].parse().expect("invalid --temperature");
            }
            "--top-k" => {
                i += 1;
                args.top_k = raw[i].parse().expect("invalid --top-k");
            }
            "--repeat-penalty" => {
                i += 1;
                args.repeat_penalty = raw[i].parse().expect("invalid --repeat-penalty");
            }
            "-h" | "--help" => {
                eprintln!(
                    "oxbitnet chat — interactive BitNet inference

Usage: cargo run --example chat --release -- [options]

Options:
  --url <url>              Model GGUF URL or path (default: BitNet 2B-4T)
  --max-tokens <n>         Max tokens to generate (default: 512)
  --temperature <f>        Sampling temperature (default: 0.7)
  --top-k <n>              Top-K sampling (default: 40)
  --repeat-penalty <f>     Repetition penalty (default: 1.1)
  -h, --help               Show this help"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    args
}

#[tokio::main]
async fn main() -> oxbitnet::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "oxbitnet=info".parse().unwrap()),
        )
        .with_writer(io::stderr)
        .init();

    let cli = parse_args();

    eprintln!("oxbitnet — Rust CLI");
    eprintln!("===================\n");
    eprintln!("Loading model from:\n  {}\n", cli.url);

    // Progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {msg:8} [{bar:30}] {pos}%")
            .unwrap()
            .progress_chars("##-"),
    );

    let options = LoadOptions {
        on_progress: Some(Box::new(move |p: LoadProgress| {
            let pct = (p.fraction * 100.0) as u64;
            let phase = match p.phase {
                oxbitnet::model::loader::LoadPhase::Download => "Download",
                oxbitnet::model::loader::LoadPhase::Parse => "Parse",
                oxbitnet::model::loader::LoadPhase::Upload => "Upload",
            };
            pb.set_message(phase.to_string());
            pb.set_position(pct);
            if p.fraction >= 1.0 {
                pb.finish_and_clear();
            }
        })),
        cache_dir: None,
    };

    let mut bitnet = BitNet::load(&cli.url, options).await?;

    eprintln!("\nModel loaded! Type your message (Ctrl+D to exit).\n");

    let stdin = io::stdin();
    let mut history: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".into(),
        content: "You are a helpful assistant.".into(),
    }];

    loop {
        eprint!("You: ");
        io::stderr().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }

        let text = line.trim().to_string();
        if text.is_empty() {
            continue;
        }

        history.push(ChatMessage {
            role: "user".into(),
            content: text,
        });

        print!("\nAssistant: ");
        io::stdout().flush().unwrap();

        let t0 = Instant::now();
        let mut token_count = 0usize;

        let options = GenerateOptions {
            max_tokens: cli.max_tokens,
            temperature: cli.temperature,
            top_k: cli.top_k,
            repeat_penalty: cli.repeat_penalty,
            ..Default::default()
        };

        let mut stream = bitnet.generate_chat(&history, options);
        let mut assistant_response = String::new();

        while let Some(token) = stream.next().await {
            print!("{token}");
            io::stdout().flush().unwrap();
            assistant_response.push_str(&token);
            token_count += 1;
        }
        drop(stream);

        let elapsed = t0.elapsed().as_secs_f64();
        let tok_sec = token_count as f64 / elapsed;

        println!(
            "\n\n  [{token_count} tokens in {elapsed:.1}s — {tok_sec:.1} tok/s]\n"
        );

        history.push(ChatMessage {
            role: "assistant".into(),
            content: assistant_response,
        });

        // Keep system + last 2 turns to avoid context overflow (4096 tokens)
        if history.len() > 5 {
            history.drain(1..history.len() - 2);
        }
    }

    eprintln!("\nBye!");
    bitnet.dispose();
    Ok(())
}
