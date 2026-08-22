# Hasher shootout for `Frequencies<Vec<u8>>` (2026-08)

**Verdict: keep foldhash** (hashbrown 0.17's default hasher). It wins or ties every
workload shape that dominates qsv's `frequency`/cardinality path; no challenger
earns a dependency change.

Context: pre-0.55.0 perf sweep, tracked in
[dathere/qsv#4391](https://github.com/dathere/qsv/issues/4391).

## Method

| | |
|---|---|
| Workload | 1M rows per shape → `HashMap<Vec<u8>, u64>` build via `entry_ref(...).and_modify(...).or_insert(1)`, mirroring `Frequencies::add_borrowed` (borrowed probe, alloc only on vacant) |
| Build | rustc 1.98.0 · `opt-level=3` · `lto=true` · `codegen-units=1` · `-C target-cpu=native` |
| Timing | min-of-7 per cell · Apple Silicon (aarch64) |
| Data | deterministic xorshift, no rand dependency (source below) |
| Hashers | foldhash 0.2.0 (hashbrown default) · rapidhash 4.5.1 · gxhash 3.5.0 · rustc-hash 2.1.3 · ahash 0.8.12 · twox-hash 2.1.3 (XXH3-64) |

## Results

Absolute build times:

| shape (1M rows) | foldhash | rapidhash | gxhash | fxhash | ahash | xxh3 (twox) |
|---|---:|---:|---:|---:|---:|---:|
| low_card_5000 (`category_NNNN`) | **5.25 ms** | 5.59 | 5.78 | 5.69 | 5.81 | 18.65 |
| timestamps (19-byte) | 56.80 ms | 55.92 | **55.11** | 55.20 | 60.02 | 116.69 |
| hi_card_short (8–16B hex, ~1M uniques) | 37.28 ms | **36.22** | 39.28 | 36.60 | 38.59 | 91.39 |
| hi_card_long (~56B URLs, ~1M uniques) | 89.63 ms | 87.70 | **77.14** | 85.19 | 95.91 | 163.82 |

Delta vs foldhash (negative = challenger faster):

| shape | rapidhash | gxhash | fxhash | ahash | xxh3 |
|---|---:|---:|---:|---:|---:|
| low_card_5000 | +6.5% | +10.1% | +8.4% | +10.7% | +255% |
| timestamps | −1.5% | −3.0% | −2.8% | +5.7% | +105% |
| hi_card_short | −2.8% | +5.4% | −1.8% | +3.5% | +145% |
| hi_card_long | −2.2% | **−13.9%** | −5.0% | +7.0% | +83% |

## Findings

- **foldhash wins where it counts.** Low-cardinality categoricals dominate real
  qsv frequency/cardinality workloads, and every challenger is 4–11% slower
  there. Timestamps and short high-card keys are ties within ~3%.
- **gxhash's −13.9% is a narrow win.** It appears only on long unique keys, where
  each new key also pays a `to_vec` allocation — the build there is
  allocation-dominated, so even the fastest hasher moves the shape just 14%.
  Against that: +10% on the common case, an unsafe SIMD dependency, and AES
  target-feature requirements that break plain `cargo install` builds. Same
  narrow-win/common-regression pattern as the previously rejected string
  prefix-key sort.
- **ahash loses to its successor.** Behind foldhash on all four shapes
  (+3.5% to +10.7%) — expected, since foldhash is the designed follow-on and the
  reason hashbrown switched defaults.
- **xxh3's numbers are an interface artifact — but binding.** XXH3 is a one-shot
  hasher; hashbrown's `Hasher` trait forces the streaming API, whose per-key
  state setup costs 2–3.5× here. That is the only way a `HashMap` can use it, so
  it is disqualified regardless.
- **fxhash's small long-key win isn't free.** It gives up per-instance seeding
  (HashDoS resistance) for parity elsewhere — the same trade rejected in the
  earlier FixedState/HashTable investigation.

Do not re-litigate without a new workload profile showing long-string frequency
build as a real qsv hotspot.

## Reproducing

Throwaway harness (deliberately not in this crate's `Cargo.toml` — repo
convention keeps bench-only dependencies out of the tree):

```toml
# Cargo.toml
[package]
name = "hashbench"
version = "0.1.0"
edition = "2021"

[dependencies]
hashbrown = "0.17"
rapidhash = "*"
gxhash = "*"
rustc-hash = "*"
ahash = "*"
twox-hash = "*"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

```rust
// src/main.rs
use std::hint::black_box;
use std::hash::BuildHasher;
use std::time::Instant;
use hashbrown::HashMap;

fn shapes() -> Vec<(&'static str, Vec<Vec<u8>>)> {
    let n = 1_000_000usize;
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut r = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    vec![
        ("low_card_5000", (0..n).map(|_| format!("category_{}", r() % 5000).into_bytes()).collect()),
        ("timestamps",    (0..n).map(|_| format!("2020-{:02}-{:02}T{:02}:{:02}:{:02}",
                              r()%12+1, r()%28+1, r()%24, r()%60, r()%60).into_bytes()).collect()),
        ("hi_card_short", (0..n).map(|_| format!("{:x}", r()).into_bytes()).collect()),
        ("hi_card_long",  (0..n).map(|_| format!(
                              "https://example.com/api/v2/resource/{:016x}/detail?page={}",
                              r(), r()%100).into_bytes()).collect()),
    ]
}

fn build<S: BuildHasher + Default>(data: &[Vec<u8>]) -> usize {
    let mut m: HashMap<Vec<u8>, u64, S> = HashMap::default();
    for k in data {
        // mirrors Frequencies::add_borrowed: borrowed probe, alloc only on vacant
        m.entry_ref(k.as_slice()).and_modify(|c| *c += 1).or_insert(1);
    }
    m.len()
}

fn bench<S: BuildHasher + Default>(name: &str, data: &[Vec<u8>], reps: usize) {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        black_box(build::<S>(black_box(data)));
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("  {name:<12} {:>8.2} ms", best * 1e3);
}

fn main() {
    for (shape, data) in shapes() {
        println!("{shape} (1M rows):");
        bench::<hashbrown::DefaultHashBuilder>("foldhash", &data, 7);
        bench::<rapidhash::fast::RandomState>("rapidhash", &data, 7);
        bench::<gxhash::GxBuildHasher>("gxhash", &data, 7);
        bench::<rustc_hash::FxBuildHasher>("fxhash", &data, 7);
        bench::<ahash::RandomState>("ahash", &data, 7);
        bench::<std::hash::BuildHasherDefault<twox_hash::XxHash3_64>>("xxh3(twox)", &data, 7);
    }
}
```
