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

use inmotion_core::{Artifact, Model, Tensor};

fn usage() -> &'static str {
    "usage:\n  \
     inmotion-infer describe --artifact DIR\n  \
     inmotion-infer predict  --artifact DIR --input FILE\n  \
     inmotion-infer check    --artifact DIR --golden DIR"
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Read a `seq_len x in_channels` window: whitespace- or comma-separated floats.
fn read_window(path: &Path, seq_len: usize, channels: usize) -> Result<Tensor, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut values = Vec::new();
    for field in text.split(|c: char| c.is_whitespace() || c == ',') {
        if field.is_empty() {
            continue;
        }
        let value: f32 = field
            .parse()
            .map_err(|e| format!("{}: not a float: {e}", path.display()))?;
        // `nan` and `inf` parse happily and would then poison every comparison
        // downstream, so they are rejected at the trust boundary.
        if !value.is_finite() {
            return Err(format!(
                "{}: {field:?} is not a finite number",
                path.display()
            ));
        }
        values.push(value);
    }
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
    let model = Model::load(&artifact)?;
    println!("model        : {}", artifact.meta.model);
    println!("artifact     : {}", artifact_dir.display());
    println!("tensors      : {}", artifact.tensor_count());
    println!("family       : {}", model.family().as_str());
    println!("seq_len      : {}", model.seq_len());
    println!("in_channels  : {}", model.in_channels());
    println!("num_classes  : {}", model.num_classes());
    // Each family reports the hyperparameters that shape its own graph.
    for (name, value) in model.describe() {
        println!("{name:<13}: {value}");
    }
    match &artifact.meta.quant {
        Some(q) => println!("quantization : {q}"),
        None => println!("quantization : none (fp32)"),
    }
    Ok(())
}

fn predict(artifact_dir: &Path, input: &Path) -> Result<(), String> {
    let artifact = Artifact::load(artifact_dir)?;
    let model = Model::load(&artifact)?;
    let x = read_window(input, model.seq_len(), model.in_channels())?;

    let logits = model.forward(&x);
    let probs = inmotion_core::ops::softmax_last(&logits);
    // `total_cmp` is a total order, so a NaN cannot make the comparison lie.
    let (best, &confidence) = probs
        .data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .ok_or("model returned no logits")?;

    println!("logits     : {:?}", logits.data);
    println!("probabilities: {:?}", probs.data);
    println!("predicted  : class {best} (confidence {confidence:.6})");
    Ok(())
}

fn check(artifact_dir: &Path, golden_dir: &Path) -> Result<(), String> {
    let artifact = Artifact::load(artifact_dir)?;
    let model = Model::load(&artifact)?;
    let text = fs::read_to_string(golden_dir.join("golden.json"))
        .map_err(|e| format!("golden.json: {e}"))?;
    let golden: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("golden.json: {e}"))?;

    // A missing tolerance is an error, not a default: silently picking 2e-4
    // would let a golden file with the field deleted still print PARITY OK.
    let tol = golden["tol"]
        .as_f64()
        .ok_or("golden.json: no tol (the exporter writes one per artifact)")? as f32;
    let cases = golden["cases"].as_array().ok_or("golden.json: no cases")?;
    let mut worst = 0.0f32;
    let mut mismatches = 0usize;

    for (i, case) in cases.iter().enumerate() {
        let rows = case["input"]
            .as_array()
            .ok_or_else(|| format!("case {i}: no input"))?;
        let mut input = Vec::new();
        for row in rows {
            let row = row
                .as_array()
                .ok_or_else(|| format!("case {i}: input row is not an array"))?;
            for value in row {
                let value = value
                    .as_f64()
                    .ok_or_else(|| format!("case {i}: input is not a number"))?;
                input.push(value as f32);
            }
        }
        let expected: Vec<f32> = case["logits"]
            .as_array()
            .ok_or_else(|| format!("case {i}: no logits"))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|v| v as f32)
                    .ok_or_else(|| format!("case {i}: logit is not a number"))
            })
            .collect::<Result<_, _>>()?;

        // The same trust boundary as the logit check below: a malformed golden
        // fixture must name the case, not reach `Tensor::new`'s invariant assert.
        let expected_input = model.seq_len() * model.in_channels();
        if input.len() != expected_input {
            return Err(format!(
                "case {i}: input has {} values, expected {expected_input}",
                input.len()
            ));
        }

        if expected.len() != model.num_classes() {
            return Err(format!(
                "case {i}: golden has {} logits, but the model has {} classes",
                expected.len(),
                model.num_classes()
            ));
        }

        let x = Tensor::new(input, vec![model.seq_len(), model.in_channels()]);
        let got = model.forward(&x).data;
        if got.len() != expected.len() {
            return Err(format!(
                "case {i}: expected {} logits, got {}",
                expected.len(),
                got.len()
            ));
        }
        if got.len() != model.num_classes() {
            return Err(format!(
                "case {i}: model returned {} logits for {} classes",
                got.len(),
                model.num_classes()
            ));
        }
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

/// Time repeated inference and report latency statistics.
///
/// A single timed run on a quiet machine is noise, so this warms up first and
/// reports the median and the tail, which is what an embedded deployment is
/// actually judged on. `--json` emits the same numbers as one JSON object, so a
/// benchmark taken on the router lands in `results/perf/` without a parser.
fn bench(artifact_dir: &Path, iterations: usize, warmup: usize, json: bool) -> Result<(), String> {
    if iterations == 0 {
        return Err("--iterations must be > 0".into());
    }
    let artifact = Artifact::load(artifact_dir)?;
    let model = Model::load(&artifact)?;
    let channels = model.in_channels();
    let seq_len = model.seq_len();

    let rng = (0..seq_len * channels)
        .map(|i| ((i * 37) % 41) as f32 - 60.0)
        .collect();
    let x = Tensor::new(rng, vec![seq_len, channels]);

    for _ in 0..warmup {
        let _ = model.forward(&x);
    }

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let _ = model.forward(&x);
        samples.push(start.elapsed().as_secs_f64() * 1e3); // milliseconds
    }
    samples.sort_by(f64::total_cmp);

    let median = samples[samples.len() / 2];
    let p95 = samples[(samples.len() * 95) / 100];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;

    let throughput = 1000.0 / median;
    if json {
        // Schema matches `results/quant/edge_router_j3455.json`'s `benchmark`
        // block, so the two can be diffed field by field.
        println!(
            "{{\"model\":{},\"artifact\":{},\"iterations\":{iterations},\"warmup\":{warmup},\
             \"quantization\":{},\"min_ms\":{min:.4},\"median_ms\":{median:.4},\
             \"mean_ms\":{mean:.4},\"p95_ms\":{p95:.4},\"max_ms\":{max:.4},\
             \"throughput_per_s\":{throughput:.1}}}",
            serde_json::to_string(&artifact.meta.model)
                .map_err(|e| format!("model name is not serialisable: {e}"))?,
            serde_json::to_string(&artifact_dir.display().to_string())
                .map_err(|e| format!("artifact path is not serialisable: {e}"))?,
            match &artifact.meta.quant {
                Some(q) => serde_json::to_string(q)
                    .map_err(|e| format!("quantization spec is not serialisable: {e}"))?,
                None => "null".to_string(),
            },
        );
    } else {
        println!("model       : {}", artifact.meta.model);
        println!("artifact    : {}", artifact_dir.display());
        println!("iterations  : {iterations} (+{warmup} warmup)");
        println!("min         : {min:.4} ms");
        println!("median      : {median:.4} ms");
        println!("mean        : {mean:.4} ms");
        println!("p95         : {p95:.4} ms");
        println!("max         : {max:.4} ms");
        println!("throughput  : {throughput:.1} inferences/s");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    let artifact_dir: PathBuf = if let Some(p) = arg_value(&args, "--artifact") {
        PathBuf::from(p)
    } else {
        eprintln!("--artifact is required\n{}", usage());
        return ExitCode::from(2);
    };

    let result = match args[1].as_str() {
        "describe" => describe(&artifact_dir),
        "predict" => match arg_value(&args, "--input") {
            Some(p) => predict(&artifact_dir, &PathBuf::from(p)),
            None => Err("--input is required".into()),
        },
        "bench" => {
            let iterations = arg_value(&args, "--iterations")
                .and_then(|v| v.parse().ok())
                .unwrap_or(200);
            let warmup = arg_value(&args, "--warmup")
                .and_then(|v| v.parse().ok())
                .unwrap_or(20);
            bench(
                &artifact_dir,
                iterations,
                warmup,
                args.iter().any(|a| a == "--json"),
            )
        }
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
