//! `inmotion-infer` — run and measure an exported model without Python.
//!
//! Commands:
//!
//! ```text
//! inmotion-infer predict --artifact DIR --input window.txt
//! inmotion-infer check   --artifact DIR --golden DIR
//! inmotion-infer load    --artifact DIR [--cold] [--json]
//! inmotion-infer loadgen --artifact DIR --seconds S [--interval-ms N]
//!                        [--cpus LIST] [--json]
//! inmotion-infer bench   --artifact DIR [--iterations N] [--warmup N]
//!                        [--cpus LIST] [--cold] [--json]
//! inmotion-infer bench   --artifact DIR --concurrency N --seconds S
//!                        [--cpus LIST] [--cold] [--json]
//! inmotion-infer stress  --mode spin|mem|net|mixed --seconds S --workers N
//!                        [--mem-mb M] [--target IP] [--pps N] [--dry-run] [--json]
//! ```
//!
//! `check` is the parity harness: it replays the golden vectors the Python
//! exporter wrote and reports the maximum deviation, so the port can be verified
//! on the machine that will run it.
//!
//! The measurement commands emit schema 2 JSON: latency statistics plus the
//! self-statistics (CPU time, RSS, peak RSS, artifact bytes, affinity, governor,
//! frequency) the edge campaign compares across devices and precisions. The
//! previous fields are all still present, so an older consumer keeps working.
//!
//! `--cpus` is applied before anything is loaded and before any thread is
//! spawned, so the mask is inherited by every thread this process creates.

mod cold;
mod cpus;
mod selfstat;
mod stress;

use std::collections::VecDeque;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use inmotion_core::{Artifact, Model, Tensor};

fn usage() -> &'static str {
    "usage:\n  \
     inmotion-infer describe --artifact DIR\n  \
     inmotion-infer predict  --artifact DIR --input FILE\n  \
     inmotion-infer check    --artifact DIR --golden DIR\n  \
     inmotion-infer stream   --artifact DIR [--input FILE] [--json]\n  \
     inmotion-infer load     --artifact DIR [--cold] [--json]\n  \
     inmotion-infer loadgen  --artifact DIR --seconds S [--interval-ms N] [--cpus LIST] [--json]\n  \
     inmotion-infer bench    --artifact DIR [--iterations N] [--warmup N] [--cpus LIST] [--cold] [--json]\n  \
     inmotion-infer bench    --artifact DIR --concurrency N --seconds S [--cpus LIST] [--cold] [--json]\n  \
     inmotion-infer stress   --mode spin|mem|net|mixed --seconds S --workers N \\
     [--mem-mb M] [--target IP] [--pps N] [--dry-run] [--json]\n\n  \
     --cpus LIST pins the process before anything is loaded: 0, 0-2, 0,2-3."
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag_present(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// A flag that must carry a value, with a message that names it.
fn required_value(args: &[String], flag: &str) -> Result<String, String> {
    arg_value(args, flag).ok_or_else(|| format!("{flag} requires a value"))
}

/// Parse a required flag into `T`.
fn parse_flag<T>(args: &[String], flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = required_value(args, flag)?;
    raw.parse::<T>().map_err(|e| format!("{flag}: {e}"))
}

/// Parse an optional flag into `T`, rejecting a flag that has no value.
fn optional_flag<T>(args: &[String], flag: &str) -> Result<Option<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match arg_value(args, flag) {
        Some(raw) => raw
            .parse::<T>()
            .map(Some)
            .map_err(|e| format!("{flag}: {e}")),
        None => Ok(None),
    }
}

/// Reject a non-finite float before it reaches JSON, where it would become null.
fn finite(value: f64, what: &str) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{what} is not a finite number: {value}"))
    }
}

/// Round a metric to a fixed number of decimals, rejecting the non-finite.
fn rounded(value: f64, places: u32, what: &str) -> Result<f64, String> {
    let value = finite(value, what)?;
    if places > 6 {
        return Err(format!(
            "internal error: {places} decimal places is too many"
        ));
    }
    let factor = f64::from(10u32.pow(places));
    Ok((value * factor).round() / factor)
}

/// A positive, representable time budget in seconds.
fn seconds_budget(seconds: f64, flag: &str) -> Result<Duration, String> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(format!(
            "{flag} must be a finite number of seconds > 0, got {seconds}"
        ));
    }
    Duration::try_from_secs_f64(seconds).map_err(|e| format!("{flag}: {e}"))
}

/// Read a `seq_len x in_channels` window: whitespace- or comma-separated floats.
/// Parse whitespace- or comma-separated floats, rejecting the non-finite.
///
/// `nan` and `inf` parse happily and would then poison every comparison
/// downstream, so they are rejected at the trust boundary rather than compared.
fn parse_values(text: &str, context: &str) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    for field in text.split(|c: char| c.is_whitespace() || c == ',') {
        if field.is_empty() {
            continue;
        }
        let value: f32 = field
            .parse()
            .map_err(|e| format!("{context}: not a float: {e}"))?;
        if !value.is_finite() {
            return Err(format!("{context}: {field:?} is not a finite number"));
        }
        values.push(value);
    }
    Ok(values)
}

fn read_window(path: &Path, seq_len: usize, channels: usize) -> Result<Tensor, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values = parse_values(&text, &path.display().to_string())?;
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

/// Advance the sliding window by one input line.
///
/// A whole window replaces the buffer; a single timestep is appended and the
/// oldest is dropped once the window is full. Both cases leave exactly
/// `window_len` values, which is the invariant the classifier relies on.
fn slide(buffer: &mut VecDeque<f32>, values: &[f32], window_len: usize) {
    if values.len() == window_len {
        buffer.clear();
    }
    buffer.extend(values.iter().copied());
    while buffer.len() > window_len {
        buffer.pop_front();
    }
}

/// Classify a continuous stream, one line at a time.
///
/// A line carrying `in_channels` values is one timestep: the runtime keeps the
/// last `seq_len` of them and emits a classification for each new one once the
/// window is full. A line carrying `seq_len * in_channels` values is a whole
/// window, classified immediately, which is how a dataset is replayed through
/// the same command. The two arities cannot collide, and nothing is buffered
/// across a blank line, so a pause in the stream is not mistaken for data.
///
/// One classification goes to stdout per line and is flushed immediately, so
/// whatever is downstream sees it as it happens. The summary goes to stderr, so
/// a pipe stays pure data.
fn stream(artifact_dir: &Path, input: Option<&Path>, json: bool) -> Result<(), String> {
    let artifact = Artifact::load(artifact_dir)?;
    let model = Model::load(&artifact)?;
    let channels = model.in_channels();
    let seq_len = model.seq_len();
    let window_len = seq_len * channels;
    let classes = &artifact.meta.classes;

    let reader: Box<dyn BufRead> = match input {
        Some(path) => Box::new(BufReader::new(
            File::open(path).map_err(|e| format!("{}: {e}", path.display()))?,
        )),
        None => Box::new(BufReader::new(io::stdin())),
    };

    let mut buffer: VecDeque<f32> = VecDeque::with_capacity(window_len);
    let mut samples = 0usize;
    let mut emitted = 0usize;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("line {}: {e}", index + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let values = parse_values(trimmed, &format!("line {}", index + 1))?;
        let whole_window = values.len() == window_len;
        if !whole_window && values.len() != channels {
            return Err(format!(
                "line {}: expected {channels} values (one timestep) or {window_len} \
                 (a whole window), got {}",
                index + 1,
                values.len()
            ));
        }

        slide(&mut buffer, &values, window_len);
        samples += 1;
        if buffer.len() < window_len {
            continue; // still filling; nothing to classify yet
        }

        let x = Tensor::new(buffer.iter().copied().collect(), vec![seq_len, channels]);
        let logits = model.forward(&x);
        let probs = inmotion_core::ops::softmax_last(&logits);
        let (best, &confidence) = probs
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("model returned no logits")?;

        if json {
            let record = serde_json::json!({
                "sample": samples,
                "class": best,
                "label": classes.get(best),
                "confidence": confidence,
                "probabilities": probs.data,
            });
            writeln!(out, "{record}").map_err(|e| e.to_string())?;
        } else {
            let name = classes
                .get(best)
                .map_or_else(|| format!("class {best}"), Clone::clone);
            let spread = probs
                .data
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    format!(
                        "{} {p:.4}",
                        classes.get(i).map_or_else(|| i.to_string(), Clone::clone)
                    )
                })
                .collect::<Vec<_>>()
                .join("  ");
            writeln!(
                out,
                "t={samples:<6} {name:<12} confidence={confidence:.4}   [{spread}]"
            )
            .map_err(|e| e.to_string())?;
        }
        out.flush().map_err(|e| e.to_string())?;
        emitted += 1;
    }

    // stderr, so a consumer piping stdout gets only the classifications
    eprintln!("{samples} samples in, {emitted} classifications out");
    if emitted == 0 {
        return Err(format!(
            "no classification: a window needs {seq_len} timesteps of {channels} \
             values, and the stream never had that many"
        ));
    }
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

// ── Self-statistics shared by `load`, `loadgen` and `bench` ───────────────

/// What a load cost, in wall and CPU time, with the memory it left behind.
struct LoadMetrics {
    cold: bool,
    evicted: Option<usize>,
    load_ms: f64,
    /// Load wall time measured with a cold page cache, or `-1` when the run was
    /// warm. With `--cold` this is the same load as `load_ms`, executed cold.
    load_cold_ms: f64,
    cpu_ms_load: f64,
    artifact_bytes: u64,
    rss_after_load_kb: u64,
}

/// Prepare the page cache (optionally), then load the artifact and the model.
///
/// The stopwatches start *after* the cold purge so `load_ms` is the load itself
/// and stays comparable between a cold and a warm run; the purge is preparation,
/// not loading. Constructing the model is part of the measurement: reading
/// `meta.json` and the weights without building the graph would time a file read,
/// not a load.
fn measure_load(artifact_dir: &Path, cold: bool) -> Result<(Artifact, Model, LoadMetrics), String> {
    let evicted = if cold {
        Some(cold::purge_dir(artifact_dir)?)
    } else {
        None
    };
    let cpu_from = selfstat::process_cpu_ns()?;
    let started = Instant::now();
    let artifact = Artifact::load(artifact_dir)?;
    let model = Model::load(&artifact)?;
    let load_ms = finite(started.elapsed().as_secs_f64() * 1e3, "load_ms")?;
    let cpu_ms_load = selfstat::elapsed_cpu_ms(cpu_from)?;
    let artifact_bytes = selfstat::artifact_bytes(artifact_dir)?;
    let rss_after_load_kb = selfstat::rss_kb()?;
    Ok((
        artifact,
        model,
        LoadMetrics {
            cold,
            evicted,
            load_ms,
            load_cold_ms: if cold { load_ms } else { -1.0 },
            cpu_ms_load,
            artifact_bytes,
            rss_after_load_kb,
        },
    ))
}

/// The deterministic input every measurement command runs on.
///
/// Fixed values rather than random ones: two runs of the same binary must warm
/// the same caches and branch-predict the same way, or the comparison between
/// them measures the generator.
fn bench_input(model: &Model) -> Tensor {
    let seq_len = model.seq_len();
    let channels = model.in_channels();
    let values = (0..seq_len * channels)
        .map(|index| ((index * 37) % 41) as f32 - 60.0)
        .collect();
    Tensor::new(values, vec![seq_len, channels])
}

/// The nearest-rank quantile of a sorted, non-empty sample set.
fn quantile_ms(sorted: &[f64], percent: usize) -> f64 {
    // Callers guarantee a non-empty slice; the clamp keeps a one-sample run sane.
    sorted[(sorted.len() * percent / 100).min(sorted.len() - 1)]
}

fn print_json(record: &serde_json::Value) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string(record).map_err(|e| format!("cannot serialise the result: {e}"))?
    );
    Ok(())
}

fn load(artifact_dir: &Path, cold: bool, json: bool) -> Result<(), String> {
    let (artifact, _model, metrics) = measure_load(artifact_dir, cold)?;
    if json {
        return print_json(&load_record(&artifact, artifact_dir, &metrics)?);
    }
    println!("model      : {}", artifact.meta.model);
    println!("artifact   : {}", artifact_dir.display());
    match metrics.evicted {
        Some(files) => println!("cold       : true ({files} file(s) evicted)"),
        None => println!("cold       : false"),
    }
    println!("bytes      : {}", metrics.artifact_bytes);
    println!("load       : {:.4} ms", metrics.load_ms);
    println!("cpu        : {:.4} ms", metrics.cpu_ms_load);
    println!(
        "rss        : {} kB after load, {} kB peak",
        metrics.rss_after_load_kb,
        selfstat::peak_rss_kb()?
    );
    Ok(())
}

fn load_record(
    artifact: &Artifact,
    dir: &Path,
    metrics: &LoadMetrics,
) -> Result<serde_json::Value, String> {
    let load_ms = rounded(metrics.load_ms, 4, "load_ms")?;
    let cpu_ms_load = rounded(metrics.cpu_ms_load, 4, "cpu_ms_load")?;
    Ok(serde_json::json!({
        "schema": 2,
        "model": artifact.meta.model,
        "artifact": dir.display().to_string(),
        "cold": metrics.cold,
        "artifact_bytes": metrics.artifact_bytes,
        "load_ms": load_ms,
        "cpu_ms_load": cpu_ms_load,
        "rss_after_load_kb": metrics.rss_after_load_kb,
        "peak_rss_kb": selfstat::peak_rss_kb()?,
    }))
}

// ── `loadgen`: continuous inference workload ──────────────────────────────

/// One `loadgen` run's measurements.
struct LoadgenNumbers {
    seconds: f64,
    iterations: u64,
    throughput_per_s: f64,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    cpu_ms: f64,
    cpu_pct: f64,
    errors: u64,
}

/// Classify the fixed input repeatedly for `seconds`.
///
/// `interval_ms` inserts that much sleep after every inference, which is how the
/// interference phase runs a realistic duty cycle (one inference every 2 s)
/// instead of saturating a core. The sleep is outside the latency samples and
/// outside CPU time, but inside the wall window, so `cpu_pct` reports the duty
/// cycle rather than the inference cost.
///
/// A run that cannot complete a single inference is an error, not a zero-rate
/// measurement. `errors` counts inferences whose logits were not finite: on
/// this runtime that is the only way an inference can fail, and a nonzero count
/// means the numbers above it describe a broken model.
fn run_loadgen(
    artifact_dir: &Path,
    seconds: f64,
    interval_ms: u64,
) -> Result<(Artifact, LoadgenNumbers), String> {
    let budget = seconds_budget(seconds, "--seconds")?;
    let (artifact, model, _load) = measure_load(artifact_dir, false)?;
    let input = bench_input(&model);
    let _ = model.forward(&input); // one warm-up, not counted
    let interval = Duration::from_millis(interval_ms);
    let cpu_from = selfstat::process_cpu_ns()?;
    let started = Instant::now();
    let deadline = started + budget;
    // One latency sample per inference. The campaign's windows are 60 s (about
    // 6e4 samples); a multi-hour soak would want a histogram instead.
    // ponytail: in-memory sample vector, add a histogram if asked for soak runs.
    let mut samples: Vec<f64> = Vec::new();
    let mut errors = 0u64;
    while Instant::now() < deadline {
        let start = Instant::now();
        let logits = model.forward(&input);
        samples.push(start.elapsed().as_secs_f64() * 1e3);
        if !logits.data.iter().all(|value| value.is_finite()) {
            errors += 1;
        }
        if !interval.is_zero() {
            // Never sleep past the requested window: an interval larger than
            // the budget must end the run, not extend it.
            let remaining = deadline.saturating_duration_since(Instant::now());
            std::thread::sleep(interval.min(remaining));
        }
    }
    let wall = started.elapsed().as_secs_f64();
    let cpu_ms = selfstat::elapsed_cpu_ms(cpu_from)?;
    if samples.is_empty() {
        return Err("no inference completed in the time budget".into());
    }
    if !wall.is_finite() || wall <= 0.0 {
        return Err("wall time is zero: cannot report a rate".into());
    }
    samples.sort_by(f64::total_cmp);
    let numbers = LoadgenNumbers {
        seconds,
        iterations: samples.len() as u64,
        throughput_per_s: samples.len() as f64 / wall,
        p50_ms: quantile_ms(&samples, 50),
        p95_ms: quantile_ms(&samples, 95),
        max_ms: samples[samples.len() - 1],
        cpu_ms,
        cpu_pct: cpu_ms / (wall * 1e3) * 100.0,
        errors,
    };
    Ok((artifact, numbers))
}

fn loadgen(artifact_dir: &Path, seconds: f64, interval_ms: u64, json: bool) -> Result<(), String> {
    let (artifact, numbers) = run_loadgen(artifact_dir, seconds, interval_ms)?;
    if json {
        print_json(&loadgen_record(&artifact, artifact_dir, &numbers)?)?;
    } else {
        println!("model      : {}", artifact.meta.model);
        println!("artifact   : {}", artifact_dir.display());
        println!("seconds    : {:.3}", numbers.seconds);
        if interval_ms > 0 {
            println!("interval   : {interval_ms} ms between inferences");
        }
        println!("iterations : {}", numbers.iterations);
        println!("throughput : {:.1} inferences/s", numbers.throughput_per_s);
        println!("p50        : {:.4} ms", numbers.p50_ms);
        println!("p95        : {:.4} ms", numbers.p95_ms);
        println!("max        : {:.4} ms", numbers.max_ms);
        println!(
            "cpu        : {:.4} ms ({:.1} %)",
            numbers.cpu_ms, numbers.cpu_pct
        );
        println!("errors     : {}", numbers.errors);
    }
    if numbers.errors > 0 {
        return Err(format!(
            "{} inference(s) produced non-finite logits",
            numbers.errors
        ));
    }
    Ok(())
}

fn loadgen_record(
    artifact: &Artifact,
    dir: &Path,
    numbers: &LoadgenNumbers,
) -> Result<serde_json::Value, String> {
    let seconds = rounded(numbers.seconds, 3, "seconds")?;
    let throughput = rounded(numbers.throughput_per_s, 1, "throughput_per_s")?;
    let p50 = rounded(numbers.p50_ms, 4, "p50_ms")?;
    let p95 = rounded(numbers.p95_ms, 4, "p95_ms")?;
    let max = rounded(numbers.max_ms, 4, "max_ms")?;
    let cpu_ms = rounded(numbers.cpu_ms, 4, "cpu_ms")?;
    let cpu_pct = rounded(numbers.cpu_pct, 2, "cpu_pct")?;
    Ok(serde_json::json!({
        "schema": 2,
        "model": artifact.meta.model,
        "artifact": dir.display().to_string(),
        "seconds": seconds,
        "iterations": numbers.iterations,
        "throughput_per_s": throughput,
        "p50_ms": p50,
        "p95_ms": p95,
        "max_ms": max,
        "cpu_ms": cpu_ms,
        "cpu_pct": cpu_pct,
        "errors": numbers.errors,
    }))
}

/// `--interval-ms`, defaulting to 0. Only a non-negative integer is legal.
///
/// A flag with no value is an error, not a zero: silently treating
/// `--interval-ms` at the end of the command line as "saturate the core" would
/// hide the typo in the one measurement the flag exists to pace.
fn interval_ms_from_args(args: &[String]) -> Result<u64, String> {
    if !flag_present(args, "--interval-ms") {
        return Ok(0);
    }
    let raw = required_value(args, "--interval-ms")?;
    raw.parse::<u64>().map_err(|_| {
        format!("--interval-ms {raw:?} is not a non-negative whole number of milliseconds")
    })
}

// ── `bench`: latency over a fixed iteration budget ────────────────────────

/// The schema-v2 self-statistics block shared by both `bench` variants.
struct BenchContext {
    load_ms: f64,
    load_cold_ms: f64,
    artifact_bytes: u64,
    cpu_ms_load: f64,
    cpu_ms_bench: f64,
    cpu_pct_bench: f64,
    cpu_ms_per_infer: f64,
    rss_after_load_kb: u64,
    peak_rss_kb: u64,
    governor: Option<String>,
    freq_mhz: f64,
}

impl BenchContext {
    /// Derive the CPU ratios from one measured window.
    fn new(
        load: &LoadMetrics,
        cpu_ms_bench: f64,
        wall_ms: f64,
        iterations: u64,
    ) -> Result<Self, String> {
        if iterations == 0 {
            return Err("no inference completed: cannot report CPU per inference".into());
        }
        if !(wall_ms.is_finite() && wall_ms > 0.0) {
            return Err(format!("wall time {wall_ms} ms is not a positive duration"));
        }
        let cpu_ms_bench = finite(cpu_ms_bench, "cpu_ms_bench")?;
        Ok(Self {
            load_ms: load.load_ms,
            load_cold_ms: load.load_cold_ms,
            artifact_bytes: load.artifact_bytes,
            cpu_ms_load: load.cpu_ms_load,
            cpu_ms_bench,
            cpu_pct_bench: finite(cpu_ms_bench / wall_ms * 100.0, "cpu_pct_bench")?,
            cpu_ms_per_infer: finite(cpu_ms_bench / iterations as f64, "cpu_ms_per_infer")?,
            rss_after_load_kb: load.rss_after_load_kb,
            peak_rss_kb: selfstat::peak_rss_kb()?,
            governor: selfstat::governor(),
            freq_mhz: selfstat::freq_mhz(),
        })
    }
}

/// One timed `bench` run's measurements.
struct BenchNumbers {
    iterations: usize,
    warmup: usize,
    min_ms: f64,
    median_ms: f64,
    mean_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    wall_ms: f64,
    throughput_per_s: f64,
    total_ms: f64,
    ctx: BenchContext,
}

/// Time repeated inference.
///
/// A single timed run on a quiet machine is noise, so this warms up first and
/// reports the median and the tail, which is what an embedded deployment is
/// actually judged on. The CPU window is the timed loop only, so
/// `cpu_pct_bench` and `cpu_ms_per_infer` describe the benchmark rather than the
/// process's whole life.
fn run_bench(
    artifact_dir: &Path,
    iterations: usize,
    warmup: usize,
    cold: bool,
) -> Result<(Artifact, BenchNumbers), String> {
    if iterations == 0 {
        return Err("--iterations must be > 0".into());
    }
    let process_from = Instant::now();
    let (artifact, model, load) = measure_load(artifact_dir, cold)?;
    let input = bench_input(&model);
    for _ in 0..warmup {
        let _ = model.forward(&input);
    }

    let mut samples = Vec::with_capacity(iterations);
    let cpu_from = selfstat::process_cpu_ns()?;
    let timed_from = Instant::now();
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = model.forward(&input);
        samples.push(start.elapsed().as_secs_f64() * 1e3); // milliseconds
    }
    let wall_ms = timed_from.elapsed().as_secs_f64() * 1e3;
    let cpu_ms_bench = selfstat::elapsed_cpu_ms(cpu_from)?;
    // Whole command, not just the timed loop: a caller dividing CPU time by
    // `wall_ms` alone would report more than 100% for a single-threaded run,
    // because the load and the warmup happened inside the same process.
    let total_ms = process_from.elapsed().as_secs_f64() * 1e3;
    samples.sort_by(f64::total_cmp);

    if samples.is_empty() {
        return Err("no sample was recorded".into());
    }
    let numbers = BenchNumbers {
        iterations,
        warmup,
        min_ms: samples[0],
        median_ms: quantile_ms(&samples, 50),
        mean_ms: samples.iter().sum::<f64>() / samples.len() as f64,
        p95_ms: quantile_ms(&samples, 95),
        p99_ms: quantile_ms(&samples, 99),
        max_ms: samples[samples.len() - 1],
        wall_ms,
        throughput_per_s: 1000.0 / quantile_ms(&samples, 50),
        total_ms,
        ctx: BenchContext::new(&load, cpu_ms_bench, wall_ms, iterations as u64)?,
    };
    Ok((artifact, numbers))
}

/// Build the schema-v2 object: command-specific fields plus the shared ones.
fn bench_object(
    entries: Vec<(&str, serde_json::Value)>,
    ctx: &BenchContext,
    cpus: Option<&str>,
) -> Result<serde_json::Value, String> {
    let mut map = serde_json::Map::new();
    map.insert("schema".to_string(), serde_json::Value::from(2));
    for (key, value) in entries {
        map.insert(key.to_string(), value);
    }
    for (key, value) in shared_fields(ctx, cpus)? {
        map.insert(key.to_string(), value);
    }
    Ok(serde_json::Value::Object(map))
}

/// The v2 fields both `bench` variants report.
fn shared_fields(
    ctx: &BenchContext,
    cpus: Option<&str>,
) -> Result<Vec<(&'static str, serde_json::Value)>, String> {
    let load_ms = rounded(ctx.load_ms, 4, "load_ms")?;
    let load_cold_ms = rounded(ctx.load_cold_ms, 4, "load_cold_ms")?;
    let cpu_ms_load = rounded(ctx.cpu_ms_load, 4, "cpu_ms_load")?;
    let cpu_ms_bench = rounded(ctx.cpu_ms_bench, 4, "cpu_ms_bench")?;
    let cpu_pct_bench = rounded(ctx.cpu_pct_bench, 2, "cpu_pct_bench")?;
    let cpu_ms_per_infer = rounded(ctx.cpu_ms_per_infer, 6, "cpu_ms_per_infer")?;
    let freq_mhz = rounded(ctx.freq_mhz, 1, "freq_mhz")?;
    Ok(vec![
        ("load_ms", serde_json::json!(load_ms)),
        ("load_cold_ms", serde_json::json!(load_cold_ms)),
        ("artifact_bytes", serde_json::json!(ctx.artifact_bytes)),
        ("cpu_ms_load", serde_json::json!(cpu_ms_load)),
        ("cpu_ms_bench", serde_json::json!(cpu_ms_bench)),
        ("cpu_pct_bench", serde_json::json!(cpu_pct_bench)),
        ("cpu_ms_per_infer", serde_json::json!(cpu_ms_per_infer)),
        (
            "rss_after_load_kb",
            serde_json::json!(ctx.rss_after_load_kb),
        ),
        ("peak_rss_kb", serde_json::json!(ctx.peak_rss_kb)),
        ("cpus", serde_json::json!(cpus)),
        ("governor", serde_json::json!(ctx.governor)),
        ("freq_mhz", serde_json::json!(freq_mhz)),
    ])
}

fn bench_record(
    artifact: &Artifact,
    dir: &Path,
    numbers: &BenchNumbers,
    cpus: Option<&str>,
) -> Result<serde_json::Value, String> {
    let min_ms = rounded(numbers.min_ms, 4, "min_ms")?;
    let median_ms = rounded(numbers.median_ms, 4, "median_ms")?;
    let mean_ms = rounded(numbers.mean_ms, 4, "mean_ms")?;
    let p95_ms = rounded(numbers.p95_ms, 4, "p95_ms")?;
    let p99_ms = rounded(numbers.p99_ms, 4, "p99_ms")?;
    let max_ms = rounded(numbers.max_ms, 4, "max_ms")?;
    let throughput_per_s = rounded(numbers.throughput_per_s, 1, "throughput_per_s")?;
    let wall_ms = rounded(numbers.wall_ms, 4, "wall_ms")?;
    let total_ms = rounded(numbers.total_ms, 4, "total_ms")?;
    let entries = vec![
        ("model", serde_json::json!(artifact.meta.model)),
        ("artifact", serde_json::json!(dir.display().to_string())),
        ("iterations", serde_json::json!(numbers.iterations)),
        ("warmup", serde_json::json!(numbers.warmup)),
        ("quantization", serde_json::json!(artifact.meta.quant)),
        ("min_ms", serde_json::json!(min_ms)),
        ("median_ms", serde_json::json!(median_ms)),
        ("mean_ms", serde_json::json!(mean_ms)),
        ("p95_ms", serde_json::json!(p95_ms)),
        ("p99_ms", serde_json::json!(p99_ms)),
        ("max_ms", serde_json::json!(max_ms)),
        ("throughput_per_s", serde_json::json!(throughput_per_s)),
        ("wall_ms", serde_json::json!(wall_ms)),
        ("total_ms", serde_json::json!(total_ms)),
    ];
    bench_object(entries, &numbers.ctx, cpus)
}

fn bench_command(
    args: &[String],
    artifact_dir: &Path,
    pinned: Option<&[usize]>,
    json: bool,
) -> Result<(), String> {
    let cold = flag_present(args, "--cold");
    let cpus = pinned.map(cpus::canonical);
    if flag_present(args, "--concurrency") || flag_present(args, "--seconds") {
        let concurrency: usize = optional_flag(args, "--concurrency")?.unwrap_or(1);
        let seconds: f64 = optional_flag(args, "--seconds")?.unwrap_or(5.0);
        bench_concurrent(
            artifact_dir,
            concurrency,
            seconds,
            cold,
            cpus.as_deref(),
            json,
        )
    } else {
        let iterations: usize = optional_flag(args, "--iterations")?.unwrap_or(200);
        let warmup: usize = optional_flag(args, "--warmup")?.unwrap_or(20);
        bench(
            artifact_dir,
            iterations,
            warmup,
            cold,
            cpus.as_deref(),
            json,
        )
    }
}

fn bench(
    artifact_dir: &Path,
    iterations: usize,
    warmup: usize,
    cold: bool,
    cpus: Option<&str>,
    json: bool,
) -> Result<(), String> {
    let (artifact, numbers) = run_bench(artifact_dir, iterations, warmup, cold)?;
    if json {
        return print_json(&bench_record(&artifact, artifact_dir, &numbers, cpus)?);
    }
    let ctx = &numbers.ctx;
    println!("model       : {}", artifact.meta.model);
    println!("artifact    : {}", artifact_dir.display());
    println!("iterations  : {iterations} (+{warmup} warmup)");
    println!("min         : {:.4} ms", numbers.min_ms);
    println!("median      : {:.4} ms", numbers.median_ms);
    println!("mean        : {:.4} ms", numbers.mean_ms);
    println!("p95         : {:.4} ms", numbers.p95_ms);
    println!("p99         : {:.4} ms", numbers.p99_ms);
    println!("max         : {:.4} ms", numbers.max_ms);
    println!("throughput  : {:.1} inferences/s", numbers.throughput_per_s);
    if cold {
        println!(
            "load        : {:.4} ms cold, {} bytes",
            ctx.load_ms, ctx.artifact_bytes
        );
    } else {
        println!(
            "load        : {:.4} ms warm, {} bytes",
            ctx.load_ms, ctx.artifact_bytes
        );
    }
    println!(
        "cpu         : {:.4} ms bench ({:.1} %), {:.6} ms/infer",
        ctx.cpu_ms_bench, ctx.cpu_pct_bench, ctx.cpu_ms_per_infer
    );
    println!(
        "memory      : {} kB after load, {} kB peak",
        ctx.rss_after_load_kb, ctx.peak_rss_kb
    );
    match &ctx.governor {
        Some(governor) => println!("governor    : {governor}"),
        None => println!("governor    : -"),
    }
    println!("freq        : {:.1} MHz", ctx.freq_mhz);
    match cpus {
        Some(list) => println!("cpus        : {list}"),
        None => println!("cpus        : all"),
    }
    Ok(())
}

// ── `bench --concurrency`: throughput under N concurrent inferences ───────

/// One concurrent `bench` run's measurements.
struct ConcurrentNumbers {
    concurrency: usize,
    seconds: f64,
    iterations: u64,
    throughput_per_s: f64,
    latency_ms: f64,
    ctx: BenchContext,
}

/// Throughput under N concurrent inferences.
///
/// This is the axis that scales on a batch-1 model. One inference here is 4-11
/// MFLOP, small enough that splitting its own arithmetic costs more than it
/// saves, but a device serving many requests can run several at once. Both
/// numbers are reported because latency usually degrades before throughput
/// saturates, and the saturation point is where the second stops rising while
/// the first keeps falling.
fn bench_concurrent(
    artifact_dir: &Path,
    concurrency: usize,
    seconds: f64,
    cold: bool,
    cpus: Option<&str>,
    json: bool,
) -> Result<(), String> {
    if concurrency == 0 {
        return Err("--concurrency must be > 0".into());
    }
    let budget = seconds_budget(seconds, "--seconds")?;
    let (artifact, model, load) = measure_load(artifact_dir, cold)?;
    let input = bench_input(&model);

    let cpu_from = selfstat::process_cpu_ns()?;
    let started = Instant::now();
    let deadline = started + budget;
    let counts: Vec<u64> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..concurrency)
            .map(|_| {
                // One warm-up per thread, so a first-call page fault is not
                // counted as a slow inference.
                scope.spawn(|| {
                    let _ = model.forward(&input);
                    let mut done = 0u64;
                    while Instant::now() < deadline {
                        let _ = model.forward(&input);
                        done += 1;
                    }
                    done
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or(0))
            .collect()
    });

    let wall = finite(started.elapsed().as_secs_f64(), "wall seconds")?;
    let cpu_ms_bench = selfstat::elapsed_cpu_ms(cpu_from)?;
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return Err("no inference completed in the time budget".into());
    }
    if wall <= 0.0 {
        return Err("wall time is zero: cannot report a rate".into());
    }
    // Mean service time per inference seen by ONE thread while all of them run:
    // the honest loaded latency, not the idle one.
    let latency_ms = wall * concurrency as f64 / total as f64 * 1e3;
    let numbers = ConcurrentNumbers {
        concurrency,
        seconds,
        iterations: total,
        throughput_per_s: total as f64 / wall,
        latency_ms,
        ctx: BenchContext::new(&load, cpu_ms_bench, wall * 1e3, total)?,
    };

    if json {
        return print_json(&concurrent_record(&artifact, artifact_dir, &numbers, cpus)?);
    }
    let ctx = &numbers.ctx;
    println!("model       : {}", artifact.meta.model);
    println!("artifact    : {}", artifact_dir.display());
    println!("concurrency : {concurrency}");
    println!("wall        : {wall:.2} s");
    println!("iterations  : {total}");
    println!("throughput  : {:.1} inferences/s", numbers.throughput_per_s);
    println!(
        "latency     : {:.4} ms (loaded, per thread)",
        numbers.latency_ms
    );
    println!(
        "load        : {:.4} ms, {} bytes{}",
        ctx.load_ms,
        ctx.artifact_bytes,
        if cold { ", cold" } else { ", warm" }
    );
    println!(
        "cpu         : {:.4} ms bench ({:.1} %), {:.6} ms/infer",
        ctx.cpu_ms_bench, ctx.cpu_pct_bench, ctx.cpu_ms_per_infer
    );
    println!(
        "memory      : {} kB after load, {} kB peak",
        ctx.rss_after_load_kb, ctx.peak_rss_kb
    );
    match &ctx.governor {
        Some(governor) => println!("governor    : {governor}"),
        None => println!("governor    : -"),
    }
    println!("freq        : {:.1} MHz", ctx.freq_mhz);
    match cpus {
        Some(list) => println!("cpus        : {list}"),
        None => println!("cpus        : all"),
    }
    Ok(())
}

fn concurrent_record(
    artifact: &Artifact,
    dir: &Path,
    numbers: &ConcurrentNumbers,
    cpus: Option<&str>,
) -> Result<serde_json::Value, String> {
    let seconds = rounded(numbers.seconds, 3, "seconds")?;
    let throughput_per_s = rounded(numbers.throughput_per_s, 1, "throughput_per_s")?;
    let latency_ms = rounded(numbers.latency_ms, 4, "latency_ms")?;
    let entries = vec![
        ("model", serde_json::json!(artifact.meta.model)),
        ("artifact", serde_json::json!(dir.display().to_string())),
        ("concurrency", serde_json::json!(numbers.concurrency)),
        ("seconds", serde_json::json!(seconds)),
        ("iterations", serde_json::json!(numbers.iterations)),
        ("throughput_per_s", serde_json::json!(throughput_per_s)),
        ("latency_ms", serde_json::json!(latency_ms)),
    ];
    bench_object(entries, &numbers.ctx, cpus)
}

// ── `stress`: bounded synthetic load ─────────────────────────────────────

fn stress_plan(args: &[String]) -> Result<stress::Plan, String> {
    let mode = required_value(args, "--mode")?;
    let seconds: f64 = parse_flag(args, "--seconds")?;
    let workers: usize = parse_flag(args, "--workers")?;
    let mem_mb: Option<u64> = optional_flag(args, "--mem-mb")?;
    let pps: Option<u64> = optional_flag(args, "--pps")?;
    let target = arg_value(args, "--target");
    stress::Plan::new(
        &mode,
        seconds,
        workers,
        mem_mb,
        target.as_deref(),
        pps,
        flag_present(args, "--dry-run"),
    )
}

// ── Entry point ──────────────────────────────────────────────────────────

/// Apply `--cpus`, if it was given, before anything else happens.
///
/// The exits are deliberate: a malformed list is a usage error (2), a rejected
/// syscall is a runtime error (1). Both are fatal, because continuing unpinned
/// after `--cpus` was requested would make the measurement a lie.
fn pin_from_args(args: &[String]) -> Result<Option<Vec<usize>>, ExitCode> {
    let Some(spec) = arg_value(args, "--cpus") else {
        if flag_present(args, "--cpus") {
            eprintln!("error: --cpus requires a value\n{}", usage());
            return Err(ExitCode::from(2));
        }
        return Ok(None);
    };
    match cpus::parse_cpu_list(&spec) {
        Ok(list) => match cpus::apply(&list) {
            Ok(()) => Ok(Some(list)),
            Err(error) => {
                eprintln!("error: {error}");
                Err(ExitCode::from(1))
            }
        },
        Err(error) => {
            eprintln!("error: --cpus: {error}\n{}", usage());
            Err(ExitCode::from(2))
        }
    }
}

fn finish(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    let pinned = match pin_from_args(&args) {
        Ok(pinned) => pinned,
        Err(code) => return code,
    };
    let command = args[1].as_str();
    let json = flag_present(&args, "--json");

    if command == "stress" {
        // Stress has no artifact; its argument errors are usage errors (2).
        let plan = match stress_plan(&args) {
            Ok(plan) => plan,
            Err(error) => {
                eprintln!("error: {error}\n{}", usage());
                return ExitCode::from(2);
            }
        };
        return finish(stress::run(&plan, json));
    }

    let Some(artifact_dir) = arg_value(&args, "--artifact").map(PathBuf::from) else {
        eprintln!("--artifact is required\n{}", usage());
        return ExitCode::from(2);
    };
    let dir = artifact_dir.as_path();

    let result = match command {
        "describe" => describe(dir),
        "predict" => match arg_value(&args, "--input") {
            Some(input) => predict(dir, Path::new(&input)),
            None => Err("--input is required".into()),
        },
        "stream" => stream(
            dir,
            arg_value(&args, "--input").map(PathBuf::from).as_deref(),
            json,
        ),
        "check" => match arg_value(&args, "--golden") {
            Some(golden) => check(dir, Path::new(&golden)),
            None => Err("--golden is required".into()),
        },
        "load" => load(dir, flag_present(&args, "--cold"), json),
        "loadgen" => {
            let seconds: f64 = match parse_flag(&args, "--seconds") {
                Ok(seconds) => seconds,
                Err(error) => return finish(Err(error)),
            };
            // An invalid interval is a usage error (2), like invalid --cpus.
            let interval_ms = match interval_ms_from_args(&args) {
                Ok(interval) => interval,
                Err(error) => {
                    eprintln!("error: {error}\n{}", usage());
                    return ExitCode::from(2);
                }
            };
            loadgen(dir, seconds, interval_ms, json)
        }
        "bench" => bench_command(&args, dir, pinned.as_deref(), json),
        other => Err(format!("unknown command {other:?}\n{}", usage())),
    };
    finish(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact_dir(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../20-aug/models/exotic/normal-new-ds/full-after-hpo/artifacts")
            .join(name)
    }

    /// The real q4 artifact the campaign benchmarks; a test that cannot find it
    /// is a broken checkout, not a reason to skip.
    fn real_artifact() -> PathBuf {
        let dir = artifact_dir("lejepa_ft_seed42-q4");
        assert!(
            dir.join("meta.json").is_file(),
            "missing test artifact at {}",
            dir.display()
        );
        dir
    }

    #[test]
    fn a_timestep_slides_the_window_by_one() {
        let mut buffer: VecDeque<f32> = VecDeque::new();
        for step in 0..5 {
            slide(&mut buffer, &[step as f32, step as f32 + 0.5], 4);
        }
        // Five two-value timesteps into a four-value window: the last two remain.
        assert_eq!(buffer, VecDeque::from([3.0, 3.5, 4.0, 4.5]));
    }

    #[test]
    fn a_whole_window_replaces_the_buffer() {
        let mut buffer: VecDeque<f32> = VecDeque::from([1.0, 2.0, 3.0, 4.0]);
        slide(&mut buffer, &[9.0, 8.0, 7.0, 6.0], 4);
        assert_eq!(buffer, VecDeque::from([9.0, 8.0, 7.0, 6.0]));
    }

    #[test]
    fn the_window_never_exceeds_its_length() {
        let mut buffer: VecDeque<f32> = VecDeque::new();
        for step in 0..50 {
            slide(&mut buffer, &[step as f32], 10);
            assert!(buffer.len() <= 10);
        }
        assert_eq!(
            buffer,
            VecDeque::from((40..50).map(|v| v as f32).collect::<Vec<_>>())
        );
    }

    #[test]
    fn parse_values_rejects_the_non_finite() {
        assert_eq!(parse_values("1, 2 3", "t").unwrap(), vec![1.0, 2.0, 3.0]);
        assert!(parse_values("1 nan", "t").is_err());
        assert!(parse_values("1 inf", "t").is_err());
        assert!(parse_values("1,oops", "t").is_err());
    }

    #[test]
    fn non_finite_metrics_are_rejected_before_json() {
        assert!(finite(f64::NAN, "x").is_err());
        assert!(finite(f64::INFINITY, "x").is_err());
        assert_eq!(finite(1.5, "x").unwrap(), 1.5);
        assert!(rounded(f64::NAN, 4, "x").is_err());
        assert_eq!(rounded(1.234_567, 4, "x").unwrap(), 1.2346);
    }

    #[test]
    fn seconds_budgets_reject_nonsense() {
        assert!(seconds_budget(0.0, "--seconds").is_err());
        assert!(seconds_budget(-1.0, "--seconds").is_err());
        assert!(seconds_budget(f64::NAN, "--seconds").is_err());
        assert!(seconds_budget(f64::INFINITY, "--seconds").is_err());
        assert_eq!(
            seconds_budget(2.5, "--seconds").unwrap(),
            Duration::from_millis(2500)
        );
    }

    #[test]
    fn load_json_reports_the_contract_fields() {
        let dir = real_artifact();
        let (artifact, _model, metrics) = measure_load(&dir, false).expect("load the artifact");
        let record = load_record(&artifact, &dir, &metrics).unwrap();
        let object = record.as_object().expect("an object");
        assert_eq!(object["schema"].as_u64(), Some(2));
        for key in [
            "schema",
            "model",
            "artifact",
            "cold",
            "artifact_bytes",
            "load_ms",
            "cpu_ms_load",
            "rss_after_load_kb",
            "peak_rss_kb",
        ] {
            assert!(object.contains_key(key), "load JSON is missing {key}");
        }
        assert_eq!(object["cold"].as_bool(), Some(false));
        assert_eq!(object["model"].as_str(), Some("lejepa_ft_seed42"));
        let bytes = object["artifact_bytes"].as_u64().unwrap();
        assert!(bytes > 0, "an artifact has bytes: {record}");
        assert!(object["load_ms"].as_f64().unwrap() > 0.0);
        assert!(object["cpu_ms_load"].as_f64().unwrap() >= 0.0);
        assert!(object["rss_after_load_kb"].as_u64().unwrap() > 0);
        assert!(object["peak_rss_kb"].as_u64().unwrap() > 0);
        assert!(
            object["peak_rss_kb"].as_u64().unwrap()
                >= object["rss_after_load_kb"].as_u64().unwrap()
        );
    }

    #[test]
    fn cold_load_reports_a_cold_load_time() {
        let dir = real_artifact();
        let (_artifact, _model, metrics) = measure_load(&dir, true).expect("cold load");
        assert!(metrics.cold);
        assert_eq!(metrics.load_cold_ms, metrics.load_ms);
        assert!(metrics.evicted.unwrap_or(0) >= 2);
    }

    #[test]
    fn bench_json_keeps_every_v1_field_and_adds_v2() {
        let dir = real_artifact();
        let (artifact, numbers) = run_bench(&dir, 3, 1, false).expect("bench the artifact");
        let record = bench_record(&artifact, &dir, &numbers, Some("0")).unwrap();
        let object = record.as_object().expect("an object");
        assert_eq!(object["schema"].as_u64(), Some(2));
        for key in [
            // schema v1, kept verbatim
            "model",
            "artifact",
            "iterations",
            "warmup",
            "quantization",
            "min_ms",
            "median_ms",
            "mean_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
            "throughput_per_s",
            "wall_ms",
            "total_ms",
            // schema v2
            "load_ms",
            "load_cold_ms",
            "artifact_bytes",
            "cpu_ms_load",
            "cpu_ms_bench",
            "cpu_pct_bench",
            "cpu_ms_per_infer",
            "rss_after_load_kb",
            "peak_rss_kb",
            "cpus",
            "governor",
            "freq_mhz",
        ] {
            assert!(
                object.contains_key(key),
                "bench JSON is missing {key}: {record}"
            );
        }
        assert_eq!(object["load_cold_ms"].as_f64(), Some(-1.0));
        assert_eq!(object["cpus"].as_str(), Some("0"));
        assert_eq!(object["iterations"].as_u64(), Some(3));
        assert!(object["artifact_bytes"].as_u64().unwrap() > 0);
        // A single-threaded timed loop cannot consume more CPU than it spent.
        assert!(object["cpu_pct_bench"].as_f64().unwrap() > 0.0);
        assert!(object["cpu_ms_per_infer"].as_f64().unwrap() > 0.0);
        assert!(object["peak_rss_kb"].as_u64().unwrap() > 0);
        let median = object["median_ms"].as_f64().unwrap();
        let p95 = object["p95_ms"].as_f64().unwrap();
        let p99 = object["p99_ms"].as_f64().unwrap();
        assert!(
            p99 >= p95 && p95 >= median,
            "p99 {p99} >= p95 {p95} >= median {median} must hold"
        );
    }

    #[test]
    fn the_tail_quantiles_land_on_the_documented_indices() {
        // 0..1000: index (len * percent / 100), clamped to the last sample.
        let sorted: Vec<f64> = (0..1000).map(f64::from).collect();
        assert_eq!(quantile_ms(&sorted, 50), 500.0);
        assert_eq!(quantile_ms(&sorted, 95), 950.0);
        assert_eq!(quantile_ms(&sorted, 99), 990.0);
    }

    #[test]
    fn loadgen_json_reports_the_contract_fields() {
        let dir = real_artifact();
        let (artifact, numbers) = run_loadgen(&dir, 0.05, 0).expect("run the workload");
        let record = loadgen_record(&artifact, &dir, &numbers).unwrap();
        let object = record.as_object().expect("an object");
        assert_eq!(object["schema"].as_u64(), Some(2));
        for key in [
            "schema",
            "model",
            "artifact",
            "seconds",
            "iterations",
            "throughput_per_s",
            "p50_ms",
            "p95_ms",
            "max_ms",
            "cpu_ms",
            "cpu_pct",
            "errors",
        ] {
            assert!(object.contains_key(key), "loadgen JSON is missing {key}");
        }
        assert!(object["iterations"].as_u64().unwrap() > 0);
        assert_eq!(object["errors"].as_u64(), Some(0));
        assert!(object["max_ms"].as_f64().unwrap() >= object["p95_ms"].as_f64().unwrap());
        assert!(object["p95_ms"].as_f64().unwrap() >= object["p50_ms"].as_f64().unwrap());
    }

    #[test]
    fn an_interval_lowers_the_iteration_count_and_keeps_the_sleep_out_of_the_samples() {
        // The same 0.5 s window, once saturated and once with a 100 ms duty
        // cycle. A tiny artifact (q4, 465 kB) keeps the run short.
        let dir = real_artifact();
        let (_, saturated) = run_loadgen(&dir, 0.5, 0).expect("run without an interval");
        let (_, paced) = run_loadgen(&dir, 0.5, 100).expect("run with a 100 ms interval");
        assert!(saturated.iterations > 0);
        assert!(
            paced.iterations < saturated.iterations,
            "interval run did {} iterations, saturated run {}",
            paced.iterations,
            saturated.iterations
        );
        // The sleep must not be counted as inference latency or as CPU time,
        // but it must appear in the wall window as a lower duty cycle.
        assert!(
            paced.p95_ms < 100.0,
            "interval sleep leaked into the latency samples: p95 {} ms",
            paced.p95_ms
        );
        assert!(
            paced.cpu_pct < saturated.cpu_pct,
            "paced CPU {:.2} % should be below saturated {:.2} %",
            paced.cpu_pct,
            saturated.cpu_pct
        );
    }

    #[test]
    fn interval_milliseconds_must_be_a_non_negative_integer() {
        let args = |value: &str| vec!["--interval-ms".to_string(), value.to_string()];
        assert_eq!(interval_ms_from_args(&args("50")).unwrap(), 50);
        assert_eq!(interval_ms_from_args(&args("0")).unwrap(), 0);
        assert_eq!(interval_ms_from_args(&[]).unwrap(), 0);
        for bad in ["-5", "1.5", "abc"] {
            assert!(
                interval_ms_from_args(&args(bad)).is_err(),
                "{bad:?} must not parse as --interval-ms"
            );
        }
        assert!(interval_ms_from_args(&["--interval-ms".to_string()]).is_err());
    }
}
