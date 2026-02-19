# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

qsv-stats is a high-performance Rust statistics library forked from BurntSushi's `streaming-stats`. It's designed for streaming computation on large datasets and is used by qsv's `stats` command. Key characteristics: FMA (fused multiply-add) instructions, adaptive parallelism via rayon, and composable statistics via the `Commute` trait.

## Build & Test Commands

```bash
cargo build                  # Debug build
cargo build --release        # Release build
cargo test                   # Run all tests
cargo test <test_name>       # Run a single test by name
cargo test --lib             # Library tests only
cargo doc --open             # Generate and view documentation
```

MSRV: 1.92 (Rust edition 2024)

## Architecture

### Core Trait

All statistics structs implement `Commute`, enabling parallel aggregation by merging partial results:

```rust
pub trait Commute: Sized {
    fn merge(&mut self, other: Self);
}
```

### Modules

| Module | Struct | Purpose |
|--------|--------|---------|
| **online.rs** | `OnlineStats` | Constant-space streaming mean/variance/stddev using Welford's method. Cache-optimized field layout (hot/warm/cold paths). |
| **unsorted.rs** | `Unsorted<T>` | Collects data, lazily sorts on demand. Median, quartiles (Method 3), mode/antimodes, Gini, kurtosis, Atkinson index, MAD, percentile rank. |
| **sorted.rs** | — | Statistics on pre-sorted sequences via `BinaryHeap`. |
| **frequency.rs** | `Frequencies<T>` | Exact frequency counting using `foldhash::HashMap`. Cardinality, most/least frequent. |
| **minmax.rs** | `MinMax<T>` | Min/max tracking with sort order detection (`sortiness()` returns -1.0 to 1.0). 56-byte struct. |

### Performance Patterns

- **Parallel threshold:** Datasets >10,000 elements use rayon parallel sort; smaller use sequential.
- **FMA:** `.mul_add()` used throughout for precision and speed.
- **Precalculated values:** `gini()`, `kurtosis()`, `atkinson()` accept optional precalculated mean/variance/geometric_sum to avoid redundant computation.
- **Lazy sorting:** `Unsorted<T>` defers sorting until a statistic is requested.
- **Quickselect:** O(n) average selection algorithm used for median/quartile computation.

### Type System

- `Partial<T>` wraps `PartialOrd` types (like f64) to provide `Ord`, enabling use in sorted collections.
- Statistics are generic over types implementing `num_traits::ToPrimitive` + `PartialOrd`.

## Testing

Tests are embedded in each module via `#[cfg(test)]` blocks (100+ tests total). Tests cover edge cases including empty data, negative values, zeros, and precision.

## Dependencies

Only 4 production dependencies: `foldhash`, `num-traits`, `rayon`, `serde`.
