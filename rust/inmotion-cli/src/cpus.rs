//! The `--cpus` grammar and the affinity syscall behind it.
//!
//! The campaign pins the runtime so a measurement names the cores it ran on.
//! Both failure modes are fatal by design:
//!
//! * a malformed list exits 2 *before* anything is loaded — a typo must never
//!   decay into "whatever mask was inherited";
//! * a failing `sched_setaffinity` reports its errno and exits non-zero, because
//!   a run that silently continued unpinned would publish pinned-core numbers
//!   for an unpinned execution.
//!
//! The grammar is the one `taskset` and the Linux kernel's own `isolcpus=`
//! syntax use: single CPUs, inclusive ranges, comma separated (`0`, `0-2`,
//! `0,2-3`). The mask is applied from the main thread before any thread is
//! spawned, which is what makes it process-wide: every thread created later
//! inherits the mask.

use std::io;

/// Number of CPUs a `cpu_set_t` can represent (`CPU_SETSIZE`, 1024 on Linux).
pub const MAX_CPUS: usize = 1024;

/// Parse the `--cpus` grammar into a sorted, deduplicated CPU list.
///
/// Duplicates are collapsed rather than rejected: `0,0-2` names the same mask as
/// `0-2`, and the JSON echo is canonical anyway.
pub fn parse_cpu_list(spec: &str) -> Result<Vec<usize>, String> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Err("empty CPU list".into());
    }
    let mut cpus = Vec::new();
    for term in trimmed.split(',') {
        let term = term.trim();
        if term.is_empty() {
            return Err(format!("{trimmed:?}: empty term in the CPU list"));
        }
        // Whitespace is allowed around the commas (a shell-quoted list is easy
        // to read) but not inside a term: `0 - 2` is a typo, not a range.
        if term.chars().any(char::is_whitespace) {
            return Err(format!("{trimmed:?}: whitespace inside {term:?}"));
        }
        match term.split_once('-') {
            Some((low, high)) => {
                let low = parse_cpu(low, trimmed)?;
                let high = parse_cpu(high, trimmed)?;
                if low > high {
                    return Err(format!("{trimmed:?}: range {term:?} is reversed"));
                }
                cpus.extend(low..=high);
            }
            None => cpus.push(parse_cpu(term, trimmed)?),
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    Ok(cpus)
}

/// Parse one CPU index, rejected unless it fits a `cpu_set_t`.
fn parse_cpu(text: &str, spec: &str) -> Result<usize, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err(format!("{spec:?}: empty CPU index"));
    }
    let cpu: usize = text
        .parse()
        .map_err(|_| format!("{spec:?}: {text:?} is not a CPU index"))?;
    if cpu >= MAX_CPUS {
        return Err(format!(
            "{spec:?}: CPU {cpu} does not fit the {MAX_CPUS}-CPU affinity mask"
        ));
    }
    Ok(cpu)
}

/// Render a sorted, deduplicated CPU list back into `taskset` notation.
pub fn canonical(cpus: &[usize]) -> String {
    let mut parts = Vec::new();
    let mut index = 0;
    while index < cpus.len() {
        let first = cpus[index];
        let mut last = first;
        while index + 1 < cpus.len() && cpus[index + 1] == last + 1 {
            index += 1;
            last = cpus[index];
        }
        if first == last {
            parts.push(first.to_string());
        } else {
            parts.push(format!("{first}-{last}"));
        }
        index += 1;
    }
    parts.join(",")
}

/// Pin the calling thread (and every thread spawned after it) to `cpus`.
///
/// Returns the errno on failure; the caller must treat that as fatal.
pub fn apply(cpus: &[usize]) -> Result<(), String> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `set` is a local, correctly aligned `cpu_set_t`; `CPU_ZERO` and
    // `CPU_SET` only write inside it, and every index was bounded by
    // `parse_cpu` (or is checked here for callers that build the slice
    // themselves).
    unsafe {
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            if cpu >= MAX_CPUS {
                return Err(format!("CPU {cpu} does not fit the affinity mask"));
            }
            libc::CPU_SET(cpu, &mut set);
        }
    }
    // SAFETY: `set` is a valid, initialised `cpu_set_t` and the size matches it;
    // pid 0 means "the calling thread".
    let rc = unsafe {
        libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            std::ptr::from_ref(&set),
        )
    };
    if rc != 0 {
        let error = io::Error::last_os_error();
        return Err(format!(
            "sched_setaffinity({}): {error} (errno {})",
            canonical(cpus),
            error.raw_os_error().unwrap_or(-1)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The calling thread's current affinity mask.
    fn current_mask() -> Result<Vec<usize>, String> {
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        // SAFETY: `set` is a valid writable `cpu_set_t` of the advertised size.
        let rc = unsafe {
            libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw mut set)
        };
        if rc != 0 {
            return Err(format!("sched_getaffinity: {}", io::Error::last_os_error()));
        }
        let mut cpus = Vec::new();
        for cpu in 0..MAX_CPUS {
            // SAFETY: `cpu` is inside the mask and `set` is initialised.
            if unsafe { libc::CPU_ISSET(cpu, &set) } {
                cpus.push(cpu);
            }
        }
        Ok(cpus)
    }

    #[test]
    fn single_range_and_mixed_lists_parse() {
        assert_eq!(parse_cpu_list("0").unwrap(), vec![0]);
        assert_eq!(parse_cpu_list("0-2").unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_cpu_list("0,2-3").unwrap(), vec![0, 2, 3]);
        assert_eq!(parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list(" 2 , 0 ").unwrap(), vec![0, 2]);
        // Duplicates name the same mask and are collapsed, not rejected.
        assert_eq!(parse_cpu_list("1,1-2").unwrap(), vec![1, 2]);
    }

    #[test]
    fn malformed_lists_are_rejected() {
        for bad in [
            "",
            " ",
            "0-",
            "-2",
            "0,",
            ",0",
            "0,,1",
            "1-0",
            "0-2-3",
            "cpu0",
            "0x1",
            "4096",
            "99999999999999999999999",
            "0 - 2",
            "-",
            "0..2",
        ] {
            assert!(
                parse_cpu_list(bad).is_err(),
                "{bad:?} should not parse as a CPU list"
            );
        }
    }

    #[test]
    fn canonical_renders_ranges() {
        assert_eq!(canonical(&[0]), "0");
        assert_eq!(canonical(&[0, 1, 2]), "0-2");
        assert_eq!(canonical(&[0, 2, 3]), "0,2-3");
        assert_eq!(canonical(&[0, 1, 3, 4, 5, 7]), "0-1,3-5,7");
    }

    #[test]
    fn affinity_is_applied_to_the_calling_thread() {
        let before = current_mask().expect("sched_getaffinity");
        assert!(!before.is_empty(), "the test thread must be runnable");
        let target = vec![before[0]];
        apply(&target).expect("pinning to an already-allowed CPU must succeed");
        assert_eq!(current_mask().unwrap(), target);
    }

    #[test]
    fn an_impossible_mask_reports_its_errno() {
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        if cores >= MAX_CPUS {
            return; // this host really can run on the highest CPU; nothing to test
        }
        let error = apply(&[MAX_CPUS - 1]).expect_err("CPU 1023 must not be runnable here");
        assert!(
            error.contains("sched_setaffinity"),
            "error must name the syscall: {error}"
        );
        assert!(
            error.contains("errno 22"),
            "error must carry EINVAL's errno: {error}"
        );
    }
}
