//! Kernel-reported self-statistics: CPU time, resident memory, artifact size.
//!
//! Every number in a `bench`/`load` record that is not a stopwatch reading comes
//! from here, and the campaign compares those numbers across devices and
//! precisions. That is why nothing in this module substitutes a zero for an
//! unavailable value and why every parser rejects a field it cannot read
//! exactly: a `0 kB` RSS is indistinguishable from a real measurement, while an
//! error names the problem.
//!
//! Sources:
//!
//! * CPU time — `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`, process-wide, so it
//!   stays correct when `bench --concurrency` runs several threads.
//! * RSS and peak RSS — `/proc/self/status` (`VmRSS`, `VmHWM`), with
//!   `/proc/self/stat` as the fallback when `status` cannot be read.
//! * CPU frequency and governor — `/sys/devices/system/cpu/cpu0/cpufreq/`; `-1`
//!   means the device does not expose them, which is a fact about the device,
//!   not a zero measurement.
//! * Artifact size — the bytes the runtime actually reads, matching the Python
//!   exporter's `total_bytes` (weights plus `meta.json`, not the golden file).

use std::fs;
use std::io;
use std::path::Path;

/// Directory holding CPU 0's `cpufreq` attributes on this kernel.
const CPU0_FREQ_DIR: &str = "/sys/devices/system/cpu/cpu0/cpufreq";

/// Page size in bytes, as the kernel reports it.
///
/// Falls back to 4 KiB, which is the page size of every target this runtime is
/// built for (x86-64, glibc and musl); the fallback exists only so the
/// `/proc/self/stat` path cannot divide by zero when `sysconf` is unavailable.
pub fn page_size_bytes() -> u64 {
    // SAFETY: `_SC_PAGESIZE` is a constant query; `sysconf` has no memory side
    // effects and returns a `c_long`.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size > 0 {
        u64::try_from(size).unwrap_or(4096)
    } else {
        4096
    }
}

/// Process CPU time in nanoseconds, sleeping excluded.
///
/// The kernel is trusted for the value, but not for its range: a negative or
/// out-of-range `timespec` is reported as an error rather than cast into a
/// huge unsigned number.
pub fn process_cpu_ns() -> Result<u64, String> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a properly aligned, writable `timespec` that outlives the
    // call, and the clock id is a valid constant.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &raw mut ts) };
    if rc != 0 {
        return Err(format!(
            "clock_gettime(CLOCK_PROCESS_CPUTIME_ID): {}",
            io::Error::last_os_error()
        ));
    }
    let seconds = u64::try_from(ts.tv_sec).map_err(|_| {
        format!(
            "clock_gettime returned a negative second count: {}",
            ts.tv_sec
        )
    })?;
    let nanos = u64::try_from(ts.tv_nsec).map_err(|_| {
        format!(
            "clock_gettime returned a negative nanosecond count: {}",
            ts.tv_nsec
        )
    })?;
    if nanos >= 1_000_000_000 {
        return Err(format!(
            "clock_gettime returned {nanos} ns, which is not a fraction of a second"
        ));
    }
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanos))
        .ok_or_else(|| "clock_gettime CPU time overflows a u64".to_string())
}

/// Milliseconds of CPU time consumed since `start_ns`.
///
/// A clock that goes backwards is an error, not a negative duration: silently
/// reporting `-0.1 ms` of CPU would poison every ratio computed from it.
pub fn elapsed_cpu_ms(start_ns: u64) -> Result<f64, String> {
    let now = process_cpu_ns()?;
    let delta = now
        .checked_sub(start_ns)
        .ok_or("CLOCK_PROCESS_CPUTIME_ID went backwards")?;
    Ok(delta as f64 / 1e6)
}

/// Resident set size in KiB, current.
pub fn rss_kb() -> Result<u64, String> {
    match read_status_kb("VmRSS") {
        Ok(value) => Ok(value),
        Err(status_error) => {
            let text = fs::read_to_string("/proc/self/stat")
                .map_err(|e| format!("{status_error}; /proc/self/stat: {e}"))?;
            let pages = parse_stat_rss_pages(&text)
                .map_err(|e| format!("{status_error}; fallback: {e}"))?;
            Ok(pages.saturating_mul((page_size_bytes() / 1024).max(1)))
        }
    }
}

/// Peak resident set size in KiB (the high-water mark since exec).
///
/// `/proc/self/stat` has no high-water mark, so the fallback reports the current
/// RSS instead: a documented lower bound, never a fabricated zero.
pub fn peak_rss_kb() -> Result<u64, String> {
    match read_status_kb("VmHWM") {
        Ok(value) => Ok(value),
        Err(status_error) => rss_kb().map_err(|e| format!("{status_error}; fallback: {e}")),
    }
}

/// Bytes the runtime reads for an artifact: weights plus `meta.json`.
///
/// The golden vectors are deliberately excluded: this is the deployment size the
/// Python exporter reports as `total_bytes`, so the two must agree when the
/// campaign compares on-device size with exported size.
pub fn artifact_bytes(dir: &Path) -> Result<u64, String> {
    let weights = dir.join("model.safetensors");
    let meta = dir.join("meta.json");
    let size = |path: &Path| {
        fs::metadata(path)
            .map(|metadata| metadata.len())
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    Ok(size(&weights)? + size(&meta)?)
}

/// CPU 0's scaling governor, when the kernel exposes one.
pub fn governor() -> Option<String> {
    let text = fs::read_to_string(format!("{CPU0_FREQ_DIR}/scaling_governor")).ok()?;
    let value = text.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

/// CPU 0's current frequency in MHz, or `-1.0` when unavailable.
///
/// `scaling_cur_freq` is the driver's last requested frequency and is readable
/// without privileges; `cpuinfo_cur_freq` is the hardware reading and often is
/// not, so it is only a fallback.
pub fn freq_mhz() -> f64 {
    for name in ["scaling_cur_freq", "cpuinfo_cur_freq"] {
        let Ok(text) = fs::read_to_string(format!("{CPU0_FREQ_DIR}/{name}")) else {
            continue;
        };
        let Ok(khz) = text.trim().parse::<u64>() else {
            continue;
        };
        if khz == 0 {
            continue;
        }
        return khz as f64 / 1000.0;
    }
    -1.0
}

/// Read one `key: N kB` line from `/proc/self/status`.
fn read_status_kb(field: &str) -> Result<u64, String> {
    let text =
        fs::read_to_string("/proc/self/status").map_err(|e| format!("/proc/self/status: {e}"))?;
    parse_status_kb(&text, field)
}

/// Parse `field` out of a `/proc/self/status` body.
///
/// Strict on purpose: the value must be a whole number of kB with no extra
/// tokens, so `VmRSS: 12.5 kB` or `VmRSS: nan kB` is an error, not `12`.
fn parse_status_kb(text: &str, field: &str) -> Result<u64, String> {
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        if key.trim() != field {
            continue;
        }
        let mut parts = rest.split_whitespace();
        let number = parts
            .next()
            .ok_or_else(|| format!("/proc/self/status: {field} has no value"))?;
        let unit = parts.next().unwrap_or("");
        if !matches!(unit, "" | "kB") {
            return Err(format!(
                "/proc/self/status: {field} is in {unit:?}, expected kB"
            ));
        }
        if parts.next().is_some() {
            return Err(format!(
                "/proc/self/status: {field} has trailing tokens: {:?}",
                rest.trim()
            ));
        }
        return number.parse::<u64>().map_err(|_| {
            format!("/proc/self/status: {field} value {number:?} is not a whole number of kB")
        });
    }
    Err(format!("/proc/self/status: no {field} field"))
}

/// Parse field 24 (`rss`, in pages) out of `/proc/self/stat`.
///
/// The process name in field 2 is arbitrary user data and may itself contain
/// spaces, parentheses and newlines, so the line is split after the **last**
/// `)` rather than by whitespace: the fields after it are fixed by the kernel.
fn parse_stat_rss_pages(text: &str) -> Result<u64, String> {
    let close = text
        .rfind(')')
        .ok_or("/proc/self/stat: no `)` closing the process name")?;
    let rest = text
        .get(close + 1..)
        .ok_or("/proc/self/stat: truncated after the process name")?;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // Field 3 (`state`) is `fields[0]`, so field 24 (`rss`) is `fields[21]`.
    let raw = fields
        .get(21)
        .ok_or("/proc/self/stat: too few fields to carry rss")?;
    raw.parse::<u64>()
        .map_err(|_| format!("/proc/self/stat: rss {raw:?} is not a page count"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATUS: &str = "Name:\tinmotion-infer\n\
                          VmPeak:\t     123456 kB\n\
                          VmHWM:\t       2345 kB\n\
                          VmRSS:\t       2100 kB\n\
                          RssAnon:\t       1500 kB\n\
                          Threads:\t1\n";

    /// `/proc/self/stat` with `rss_pages` in field 24, and a process name that
    /// contains both a space and parentheses.
    fn stat_line(rss_pages: &str) -> String {
        let mut fields = vec!["S"; 48]; // fields 3..=50
        fields[21] = rss_pages;
        format!("12345 (inmotion (infer)) {}", fields.join(" "))
    }

    #[test]
    fn status_fields_parse_exactly() {
        assert_eq!(parse_status_kb(STATUS, "VmRSS").unwrap(), 2100);
        assert_eq!(parse_status_kb(STATUS, "VmHWM").unwrap(), 2345);
    }

    #[test]
    fn malformed_status_values_are_rejected() {
        for bad in ["12.5 kB", "nan kB", "-1 kB", "2100 MB", "2100 kB extra"] {
            let text = format!("VmRSS:\t{bad}\n");
            assert!(
                parse_status_kb(&text, "VmRSS").is_err(),
                "{bad:?} should be rejected"
            );
        }
        assert!(parse_status_kb("VmRSS:\n", "VmRSS").is_err());
        assert!(parse_status_kb(STATUS, "VmSwap").is_err());
    }

    #[test]
    fn stat_rss_parses_after_the_process_name() {
        assert_eq!(parse_stat_rss_pages(&stat_line("4096")).unwrap(), 4096);
    }

    #[test]
    fn malformed_stat_rss_is_rejected() {
        assert!(parse_stat_rss_pages(&stat_line("nan")).is_err());
        assert!(parse_stat_rss_pages(&stat_line("-2")).is_err());
        assert!(parse_stat_rss_pages("no closing paren").is_err());
        assert!(parse_stat_rss_pages("1 (short) S 1 2 3").is_err());
    }

    #[test]
    fn live_self_statistics_are_sane() {
        let rss = rss_kb().expect("VmRSS");
        let peak = peak_rss_kb().expect("VmHWM");
        assert!(rss > 0, "a live process cannot have zero RSS");
        assert!(peak >= rss, "peak {peak} kB < current {rss} kB");
        assert!(page_size_bytes() >= 4096);
    }

    #[test]
    fn cpu_time_advances_while_spinning() {
        let before = process_cpu_ns().expect("CLOCK_PROCESS_CPUTIME_ID");
        let mut state = 0u64;
        for step in 0..2_000_000u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(step);
        }
        let _ = std::hint::black_box(state);
        let after = process_cpu_ns().expect("CLOCK_PROCESS_CPUTIME_ID");
        assert!(
            after > before,
            "CPU time did not advance: {before} -> {after}"
        );
        assert!(elapsed_cpu_ms(before).unwrap() > 0.0);
        // A start in the future is a broken clock, not a negative duration.
        assert!(elapsed_cpu_ms(after + 1_000_000_000).is_err());
    }

    #[test]
    fn artifact_bytes_counts_weights_and_meta_only() {
        let dir = std::env::temp_dir().join(format!(
            "inmotion-selfstat-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.safetensors"), b"12345678").unwrap();
        fs::write(dir.join("meta.json"), b"{}").unwrap();
        fs::write(dir.join("golden.json"), b"ignored").unwrap();
        assert_eq!(artifact_bytes(&dir).unwrap(), 10);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn absent_cpufreq_reports_minus_one() {
        // The path is fixed at compile time; on a host without cpufreq the
        // contract is `-1`, and on a host with it the value must be positive.
        let freq = freq_mhz();
        assert!(
            (freq + 1.0).abs() < f64::EPSILON || freq > 0.0,
            "unexpected frequency {freq}"
        );
        if let Some(governor) = governor() {
            assert_ne!(governor, "");
        }
    }
}
