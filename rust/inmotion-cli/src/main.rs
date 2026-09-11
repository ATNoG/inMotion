//! `inmotion-infer` — run an exported model without Python.
//!
//! Two modes:
//!
//!     inmotion-infer predict --artifact DIR --input window.txt
//!     inmotion-infer check   --artifact DIR --golden DIR
//!
//! `check` is the parity harness: it replays the golden vectors the Python
//! exporter wrote and reports the maximum deviation, so the port can be verified
//! on the machine that will run it.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use inmotion_core::{Artifact, LeJepa, Tensor};

fn usage() -> &'static str {
    "usage:\n  \
     inmotion-infer describe --artifact DIR\n  \
     inmotion-infer predict  --artifact DIR --input FILE\n  \
     inmotion-infer check    --artifact DIR --golden DIR"
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

/// Read a `seq_len x in_channels` window: whitespace- or comma-separated floats.
fn read_window(path: &Path, seq_len: usize, channels: usize) -> Result<Tensor, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f32> = text
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f32>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{}: not a float: {e}", path.display()))?;
    let expected = seq_len * channels;
    if values.len() != expected {
        return Err(format!(
            "{}: expected {expected} values ({seq_len} timesteps x {channels} channels), got {}",
            path.display(),
            values.len()
        ));
    }
    Ok(Tensor::new(values, vec![seq_len, channels]))
}

fn describe(artifact_dir: &Path) -> Result<(), String> {
    let artifact = Artifact::load(artifact_dir)?;
    let model = LeJepa::load(&artifact)?;
    println!("model        : {}", artifact.meta.model);
    println!("artifact     : {}", artifact_dir.display());
    println!("tensors      : {}", artifact.tensor_count());
    println!("seq_len      : {}", model.config.seq_len);
    println!("in_channels  : {}", model.config.in_channels);
    println!("d_model      : {}", model.config.d_model);
    println!("n_heads      : {}", model.config.n_heads);
    println!("num_layers   : {}", model.config.num_layers);
    println!("num_classes  : {}", model.config.num_classes);
    match &artifact.meta.quant {
        Some(q) => println!("quantization : {q}"),
        None => println!("quantization : none (fp32)"),
    }
    Ok(())
}

fn predict(artifact_dir: &Path, input: &Path) -> Result<(), String> {
    let artifact = Artifact::load(artifact_dir)?;
    let model = LeJepa::load(&artifact)?;
    let x = read_window(input, model.config.seq_len, model.config.in_channels)?;

    let logits = model.forward(&x);
    let probs = inmotion_core::ops::softmax_last(&Tensor::new(logits.data.clone(), vec![1, logits.len()]));
    let (best, &confidence) = probs
        .data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("logits     : {:?}", logits.data);
    println!("probabilities: {:?}", probs.data);
    println!("predicted  : class {best} (confidence {confidence:.6})");
    Ok(())
}

fn check(artifact_dir: &Path, golden_dir: &Path) -> Result<(), String> {
    let artifact = Artifact::load(artifact_dir)?;
    let model = LeJepa::load(&artifact)?;
    let text = fs::read_to_string(golden_dir.join("golden.json"))
        .map_err(|e| format!("golden.json: {e}"))?;
    let golden: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("golden.json: {e}"))?;

    let tol = golden["tol"].as_f64().unwrap_or(2e-4) as f32;
    let cases = golden["cases"].as_array().ok_or("golden.json: no cases")?;
    let mut worst = 0.0f32;
    let mut mismatches = 0usize;

    for (i, case) in cases.iter().enumerate() {
        let input: Vec<f32> = case["input"]
            .as_array()
            .ok_or("case has no input")?
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32))
            .collect();
        let expected: Vec<f32> = case["logits"]
            .as_array()
            .ok_or("case has no logits")?
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let x = Tensor::new(input, vec![model.config.seq_len, model.config.in_channels]);
        let got = model.forward(&x).data;
        let diff = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(diff);
        if diff > tol {
            mismatches += 1;
            println!("  case {i}: max|diff| = {diff:.6} exceeds tol {tol:.6}");
        }
    }

    println!("cases      : {}", cases.len());
    println!("max |diff| : {worst:.8}");
    println!("tolerance  : {tol:.6}");
    if mismatches == 0 {
        println!("PARITY OK");
        Ok(())
    } else {
        Err(format!("{mismatches} case(s) outside tolerance"))
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    let artifact_dir: PathBuf = match arg_value(&args, "--artifact") {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("--artifact is required\n{}", usage());
            return ExitCode::from(2);
        }
    };

    let result = match args[1].as_str() {
        "describe" => describe(&artifact_dir),
        "predict" => match arg_value(&args, "--input") {
            Some(p) => predict(&artifact_dir, &PathBuf::from(p)),
            None => Err("--input is required".into()),
        },
        "check" => match arg_value(&args, "--golden") {
            Some(p) => check(&artifact_dir, &PathBuf::from(p)),
            None => Err("--golden is required".into()),
        },
        other => Err(format!("unknown command {other:?}\n{}", usage())),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
