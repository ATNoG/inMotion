//! Bounded synthetic load: `spin`, `mem`, `net` and `mixed`.
//!
//! The device campaign measures how a router's forwarding and inference behave
//! while the CPU, the memory bus or the network stack is loaded. That requires a
//! load generator, and a load generator running as root on a production router
//! needs a ceiling built into the code rather than into a README:
//!
//! * at most `2 x cores` workers — a toy router has four slow cores, and more
//!   threads only move time from the measurement into the scheduler;
//! * at most 600 s and 512 MiB in total — a wedged run must not need a reboot to
//!   end, and the memory must fit a 1.9 GB device with inference resident;
//! * `--dry-run` validates the same caps and generates nothing.
//!
//! Every worker stops on the same deadline. Nothing here depends on a signal
//! handler, so Ctrl-C remains the operator's kill switch.

use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use crate::selfstat;

/// Longest single stress window, seconds.
const MAX_SECONDS: f64 = 600.0;
/// Largest total allocation, MiB.
const MAX_MEM_MB: u64 = 512;
/// Largest UDP packet rate, packets per second across all workers.
const MAX_PPS: u64 = 100_000;
/// Default allocation for `mem`/`mixed`, MiB.
const DEFAULT_MEM_MB: u64 = 64;
/// Default packet rate for `net`/`mixed`, packets per second.
const DEFAULT_PPS: u64 = 1_000;
/// UDP payload size; the point is packet rate, not throughput.
const PAYLOAD_BYTES: usize = 32;

/// Which synthetic load to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Spin,
    Memory,
    Network,
    Mixed,
}

impl Mode {
    fn parse(text: &str) -> Result<Self, String> {
        match text {
            "spin" => Ok(Self::Spin),
            "mem" => Ok(Self::Memory),
            "net" => Ok(Self::Network),
            "mixed" => Ok(Self::Mixed),
            other => Err(format!(
                "--mode {other:?} is not one of spin, mem, net, mixed"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Spin => "spin",
            Self::Memory => "mem",
            Self::Network => "net",
            Self::Mixed => "mixed",
        }
    }

    fn involves_net(self) -> bool {
        matches!(self, Self::Network | Self::Mixed)
    }

    /// Which task a worker runs. `mixed` interleaves the three loads: with three
    /// or more workers, each one has at least one worker (enforced by `Plan`).
    fn task(self, worker: usize) -> Task {
        match self {
            Self::Spin => Task::Spin,
            Self::Memory => Task::Memory,
            Self::Network => Task::Network,
            Self::Mixed => match worker % 3 {
                0 => Task::Spin,
                1 => Task::Memory,
                _ => Task::Network,
            },
        }
    }
}

/// One worker's assignment.
#[derive(Debug, Clone, Copy)]
enum Task {
    Spin,
    Memory,
    Network,
}

/// A validated, bounded stress plan.
#[derive(Debug)]
pub struct Plan {
    mode: Mode,
    workers: usize,
    seconds: f64,
    mem_mb: u64,
    target: Option<SocketAddr>,
    pps: u64,
    dry_run: bool,
}

impl Plan {
    /// Validate the requested load against the hard caps.
    ///
    /// Every cap is checked before anything runs, including in `--dry-run` mode:
    /// a plan that cannot be executed must not be printed as if it could.
    pub fn new(
        mode: &str,
        seconds: f64,
        workers: usize,
        mem_mb: Option<u64>,
        target: Option<&str>,
        pps: Option<u64>,
        dry_run: bool,
    ) -> Result<Self, String> {
        let mode = Mode::parse(mode)?;
        let cores = available_cores();
        if workers == 0 {
            return Err("--workers must be at least 1".into());
        }
        if workers > cores * 2 {
            return Err(format!(
                "--workers {workers} exceeds the hard cap of 2 x {cores} cores = {}",
                cores * 2
            ));
        }
        if !seconds.is_finite() {
            return Err(format!("--seconds {seconds} is not a finite number"));
        }
        if seconds <= 0.0 {
            return Err("--seconds must be greater than 0".into());
        }
        if seconds > MAX_SECONDS {
            return Err(format!(
                "--seconds {seconds} exceeds the {MAX_SECONDS:.0} s hard cap"
            ));
        }
        let mem_mb = mem_mb.unwrap_or(DEFAULT_MEM_MB);
        if !(1..=MAX_MEM_MB).contains(&mem_mb) {
            return Err(format!(
                "--mem-mb {mem_mb} is outside the 1-{MAX_MEM_MB} MiB hard cap"
            ));
        }
        let pps = pps.unwrap_or(DEFAULT_PPS);
        if !(1..=MAX_PPS).contains(&pps) {
            return Err(format!(
                "--pps {pps} is outside the 1-{MAX_PPS} packets/s hard cap"
            ));
        }
        // Parse a target even for a mode that will not use it, so a typo cannot
        // hide behind "this mode ignores --target".
        let target = target.map(parse_target).transpose()?;
        // `target` is reported as null unless the load actually touches the
        // network, which is what the contract's example shows for `spin`.
        let target = if mode.involves_net() {
            Some(target.ok_or("--mode net and mixed need --target IP")?)
        } else {
            None
        };
        if mode == Mode::Mixed && workers < 3 {
            return Err("--mode mixed needs at least 3 workers so each load type gets one".into());
        }
        Ok(Self {
            mode,
            workers,
            seconds,
            mem_mb,
            target,
            pps,
            dry_run,
        })
    }
}

/// Run (or dry-run) a plan and print its summary.
pub fn run(plan: &Plan, json: bool) -> Result<(), String> {
    let cpu_from = selfstat::process_cpu_ns()?;
    let started = Instant::now();
    let counters = if plan.dry_run {
        Counters::default()
    } else {
        let budget =
            Duration::try_from_secs_f64(plan.seconds).map_err(|e| format!("--seconds: {e}"))?;
        run_workers(plan, started, budget)?
    };
    let wall = started.elapsed().as_secs_f64();
    let cpu_ms = selfstat::elapsed_cpu_ms(cpu_from)?;
    let peak_rss_kb = selfstat::peak_rss_kb()?;

    // `iterations` counts work units: spin-loop iterations, memory pages
    // touched, or packets delivered, summed over workers. `pps` is only a real
    // rate for the modes that send packets; everywhere else it is `-1`, the
    // repo's "this device cannot supply the value" marker.
    let iterations = counters.work_units();
    let pps = if plan.mode.involves_net() && wall > 0.0 && !plan.dry_run {
        counters.packets as f64 / wall
    } else if plan.mode.involves_net() {
        0.0
    } else {
        -1.0
    };

    if counters.send_errors > 0 {
        eprintln!(
            "warning: {} UDP send(s) failed; iterations counts delivered packets only",
            counters.send_errors
        );
    }

    let record = serde_json::json!({
        "schema": 2,
        "mode": plan.mode.as_str(),
        "workers": plan.workers,
        "seconds": rounded(plan.seconds, 3, "seconds")?,
        "iterations": iterations,
        "pps": rounded(pps, 1, "pps")?,
        "cpu_ms": rounded(cpu_ms, 4, "cpu_ms")?,
        "peak_rss_kb": peak_rss_kb,
        "target": plan.target.map(|addr| addr.to_string()),
        "dry_run": plan.dry_run,
        "errors": counters.send_errors,
    });
    if json {
        println!(
            "{}",
            serde_json::to_string(&record)
                .map_err(|e| format!("cannot serialise the stress summary: {e}"))?
        );
    } else {
        println!("mode       : {}", plan.mode.as_str());
        println!("workers    : {}", plan.workers);
        println!("seconds    : {:.3}", plan.seconds);
        println!("iterations : {iterations}");
        println!("pps        : {pps:.1}");
        println!("cpu        : {cpu_ms:.4} ms");
        println!("peak rss   : {peak_rss_kb} kB");
        match plan.target {
            Some(target) => println!("target     : {target}"),
            None => println!("target     : -"),
        }
        if plan.dry_run {
            println!("dry run    : plan accepted, no load generated");
        }
    }
    Ok(())
}

/// Validated numbers are finite; anything else would silently become JSON null.
fn rounded(value: f64, places: u32, what: &str) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!("{what} is not a finite number: {value}"));
    }
    if places > 6 {
        return Err(format!(
            "internal error: {places} decimal places is too many"
        ));
    }
    let factor = f64::from(10u32.pow(places));
    Ok((value * factor).round() / factor)
}

fn available_cores() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

fn parse_target(raw: &str) -> Result<SocketAddr, String> {
    if let Ok(addr) = raw.parse::<SocketAddr>() {
        return Ok(addr);
    }
    let ip: IpAddr = raw
        .parse()
        .map_err(|_| format!("--target {raw:?} is not an IP address or IP:port"))?;
    // Port 9 (discard) is the conventional sink when only an address is given.
    Ok(SocketAddr::new(ip, 9))
}

/// Run every worker until `deadline` and sum what they did.
fn run_workers(plan: &Plan, started: Instant, budget: Duration) -> Result<Counters, String> {
    let deadline = started + budget;
    let mem_bytes = plan.mem_mb * 1024 * 1024;
    let mem_per_worker = mem_bytes / plan.workers as u64;
    let pps_per_worker = (plan.pps / plan.workers as u64).max(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(plan.workers);
        for worker in 0..plan.workers {
            let task = plan.mode.task(worker);
            let target = plan.target;
            handles.push(scope.spawn(move || match task {
                Task::Spin => Ok(spin_until(deadline)),
                Task::Memory => mem_until(deadline, mem_per_worker),
                Task::Network => match target {
                    Some(target) => net_until(deadline, target, pps_per_worker),
                    None => Err("net stress was validated without a target".to_string()),
                },
            }));
        }
        let mut total = Counters::default();
        for handle in handles {
            let counters = handle
                .join()
                .map_err(|_| "a stress worker panicked".to_string())??;
            total.add(counters);
        }
        Ok(total)
    })
}

/// What one worker did, so the summary can report real work, not intent.
#[derive(Debug, Default, Clone, Copy)]
struct Counters {
    spin_loops: u64,
    pages: u64,
    packets: u64,
    send_errors: u64,
}

impl Counters {
    fn add(&mut self, other: Self) {
        self.spin_loops += other.spin_loops;
        self.pages += other.pages;
        self.packets += other.packets;
        self.send_errors += other.send_errors;
    }

    fn work_units(self) -> u64 {
        self.spin_loops + self.pages + self.packets
    }
}

/// Busy-loop one worker until the deadline.
fn spin_until(deadline: Instant) -> Counters {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut loops = 0u64;
    while Instant::now() < deadline {
        for _ in 0..10_000u32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
        }
        loops += 10_000;
    }
    let _ = std::hint::black_box(state);
    Counters {
        spin_loops: loops,
        ..Counters::default()
    }
}

/// Allocate, touch every page, and hold the memory until the deadline.
///
/// The allocation is capped at 512 MiB by `Plan`, so a failed allocation cannot
/// take the device down mid-measurement. Pages are touched one at a time because
/// a zeroed `vec` is lazily mapped: without the writes, RSS would not move and
/// the load would not exist.
fn mem_until(deadline: Instant, bytes: u64) -> Result<Counters, String> {
    let size = usize::try_from(bytes)
        .map_err(|_| format!("{bytes} bytes of memory does not fit this platform"))?;
    let page = usize::try_from(selfstat::page_size_bytes()).map_or(4096, |value| value.max(1));
    let mut buffer = vec![0u8; size];
    let mut pages = 0u64;
    let mut offset = 0usize;
    while offset < size {
        buffer[offset] = 1;
        offset += page;
        pages += 1;
    }
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        std::thread::sleep(remaining.min(Duration::from_millis(5)));
    }
    let _ = std::hint::black_box(&buffer);
    Ok(Counters {
        pages,
        ..Counters::default()
    })
}

/// Send UDP datagrams to `target` at `pps` packets per second until the deadline.
///
/// Delivery is not guaranteed (UDP, discard port); `iterations` counts successful
/// sends only, and failures are recounted to stderr rather than silently folded
/// into the rate.
fn net_until(deadline: Instant, target: SocketAddr, pps: u64) -> Result<Counters, String> {
    let bind = if target.is_ipv4() {
        "0.0.0.0:0"
    } else {
        "[::]:0"
    };
    let socket = UdpSocket::bind(bind).map_err(|e| format!("UDP bind({bind}): {e}"))?;
    let payload = [0u8; PAYLOAD_BYTES];
    let interval = Duration::from_secs_f64(1.0 / pps as f64);
    let mut next = Instant::now();
    let mut counters = Counters::default();
    while Instant::now() < deadline {
        match socket.send_to(&payload, target) {
            Ok(_) => counters.packets += 1,
            Err(_) => counters.send_errors += 1,
        }
        next += interval;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else {
            // Behind schedule: reset the pacing clock instead of trying to
            // catch up with a burst that would distort the measured rate.
            next = now;
        }
    }
    Ok(counters)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spin_plan(seconds: f64, workers: usize) -> Plan {
        Plan::new("spin", seconds, workers, None, None, None, false).unwrap()
    }

    #[test]
    fn the_caps_are_enforced() {
        let cores = available_cores();
        let too_many = Plan::new("spin", 1.0, cores * 2 + 1, None, None, None, false);
        assert!(too_many.unwrap_err().contains("hard cap"));

        for seconds in [0.0, -1.0, MAX_SECONDS + 1.0, f64::NAN, f64::INFINITY] {
            assert!(
                Plan::new("spin", seconds, 1, None, None, None, false).is_err(),
                "seconds {seconds} should be rejected"
            );
        }
        assert!(Plan::new("spin", 1.0, 0, None, None, None, false).is_err());
        assert!(Plan::new("spin", 1.0, 1, Some(0), None, None, false).is_err());
        assert!(Plan::new("spin", 1.0, 1, Some(MAX_MEM_MB + 1), None, None, false).is_err());
        assert!(Plan::new("spin", 1.0, 1, None, None, Some(0), false).is_err());
        assert!(Plan::new("spin", 1.0, 1, None, None, Some(MAX_PPS + 1), false).is_err());
        assert!(Plan::new("idle", 1.0, 1, None, None, None, false).is_err());
        assert!(Plan::new("spin", 1.0, 1, None, Some("not-an-ip"), None, false).is_err());
    }

    #[test]
    fn the_caps_allow_the_contractual_window() {
        // A 4-core router: 8 workers, 600 s, 512 MiB must be accepted.
        let plan = Plan::new(
            "mem",
            MAX_SECONDS,
            available_cores() * 2,
            Some(MAX_MEM_MB),
            None,
            None,
            true,
        )
        .expect("the documented maximum must be representable");
        assert!(plan.dry_run);
    }

    #[test]
    fn net_and_mixed_need_a_target_and_mixed_needs_three_workers() {
        assert!(Plan::new("net", 1.0, 1, None, None, None, false).is_err());
        assert!(Plan::new("mixed", 1.0, 2, None, Some("127.0.0.1"), None, false).is_err());
        assert!(Plan::new("mixed", 1.0, 3, None, None, None, false).is_err());
        let plan = Plan::new("mixed", 1.0, 3, None, Some("127.0.0.1:9"), None, false).unwrap();
        assert_eq!(plan.target.unwrap().to_string(), "127.0.0.1:9");
    }

    #[test]
    fn spin_stops_at_the_deadline() {
        let plan = spin_plan(0.05, 1);
        let started = Instant::now();
        let counters = run_workers(&plan, started, Duration::from_secs_f64(plan.seconds)).unwrap();
        let elapsed = started.elapsed();
        assert!(counters.spin_loops > 0, "spin did no work");
        assert!(
            elapsed < Duration::from_secs(5),
            "spin overran its deadline: {elapsed:?}"
        );
    }

    #[test]
    fn memory_is_allocated_touched_and_released() {
        let plan = Plan::new("mem", 0.05, 1, Some(8), None, None, false).unwrap();
        let started = Instant::now();
        let counters = run_workers(&plan, started, Duration::from_secs_f64(plan.seconds)).unwrap();
        assert!(
            counters.pages >= 8 * 1024 * 1024 / 65536,
            "too few pages touched"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn net_sends_packets_to_loopback() {
        let plan = Plan::new("net", 0.05, 1, None, Some("127.0.0.1:9"), Some(1000), false).unwrap();
        let started = Instant::now();
        let counters = run_workers(&plan, started, Duration::from_secs_f64(plan.seconds)).unwrap();
        assert!(counters.packets > 0, "no packet was sent: {counters:?}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_dry_run_generates_nothing_and_returns_at_once() {
        let plan = Plan::new("spin", MAX_SECONDS, 1, None, None, None, true).unwrap();
        let started = Instant::now();
        run(&plan, false).unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "a dry run must not wait out the window"
        );
    }
}
