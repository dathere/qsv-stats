# qsv-stats Project Overview

High-performance Rust streaming statistics library. Fork of BurntSushi's streaming-stats, used by qsv's `stats` command.

## Tech Stack
- Rust (edition 2024, MSRV 1.93)
- Dependencies: rayon (parallelism), serde (serialization), num-traits, foldhash

## Structure
- `src/lib.rs` - Public API, Commute trait, Partial<T> wrapper
- `src/online.rs` - OnlineStats: streaming mean/variance/stddev (Welford's algorithm)
- `src/unsorted.rs` - Unsorted<T>: lazy-sorted quantiles, gini, kurtosis, MAD, mode
- `src/frequency.rs` - Frequencies<T>: exact frequency counting with foldhash
- `src/minmax.rs` - MinMax<T>: min/max tracking with sort order detection
- `src/sorted.rs` - Sorted<T>: BinaryHeap-based (rarely used)

## Commands
- `cargo build` / `cargo build --release`
- `cargo test --lib` (141+ tests inline)
- `cargo clippy --lib`
- `cargo fmt`

## Key Patterns
- FMA (mul_add) throughout for numerical precision
- Parallel threshold at 10,000 elements (rayon)
- Unsafe blocks for unchecked f64 conversion and bounds-verified indexing
- Cache-optimized field layout in OnlineStats (hot/warm/cold paths)