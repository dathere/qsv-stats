// Regression guard for the select-based `Frequencies::modes_antimodes`.
//
// `sort_baseline` mirrors the ORIGINAL implementation (collect all (key, count)
// pairs, fully sort them ascending by key - O(c log c) - then derive
// modes/antimodes from the sorted runs). The lib's shipped implementation
// (`select_lib`) instead only sorts what the output contains: one O(c) pass for
// highest/lowest counts, collect only matching keys, sort the tiny mode set,
// and pick the 10 smallest antimodes via select_nth_unstable - O(c) average.
// Uniform-count data (highest == lowest) falls back to a single full key sort.
//
// The baseline operates on a mirrored hashbrown::HashMap<Vec<u8>, u64> with
// identical contents/hasher to the Frequencies internal map, so both sides pay
// the same map-iteration cost and only the algorithm differs. Equivalence is
// asserted before each shape is benched.
//
// RESULT (1M rows, justifies the select path):
//   all_unique (c=1M, hi==1):      ~28 ms  -> ~15 ms  (-47%)
//   low_card_5000 (c=5k):          ~157 µs -> ~9 µs   (-94%)
//   high_card_100k (c=100k):       ~1.33 ms -> ~262 µs (-80%)
//   uniform_x2 (c=500k, hi==lo):   ~parity (+2%) via the full-sort fallback
//     (the naive select path without the fallback was +16% SLOWER here: double
//     key-collect + serial select_nth on top of the unavoidable full sort)
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;
use stats::Frequencies;

const PARALLEL_THRESHOLD: usize = 10_000; // mirrors src/frequency.rs

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    }
}

fn gen_low_card(n: usize, card: usize) -> Vec<Vec<u8>> {
    let mut next = lcg(0xABCD_1234_5678_9999);
    let pool: Vec<Vec<u8>> = (0..card)
        .map(|i| format!("category_{i:04}").into_bytes())
        .collect();
    (0..n)
        .map(|_| pool[(next() as usize) % card].clone())
        .collect()
}

fn gen_all_unique(n: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| format!("{i:08}").into_bytes()).collect()
}

fn gen_high_card(n: usize, card: usize) -> Vec<Vec<u8>> {
    let mut next = lcg(0x5151_F00D_CAFE_0042);
    (0..n)
        .map(|_| format!("{:08}", (next() as usize) % card).into_bytes())
        .collect()
}

fn gen_uniform_x2(n: usize) -> Vec<Vec<u8>> {
    (0..n / 2)
        .flat_map(|i| {
            let k = format!("{i:08}").into_bytes();
            [k.clone(), k]
        })
        .collect()
}

type ModesResult = ((Vec<Vec<u8>>, usize, u32), (Vec<Vec<u8>>, usize, u32));

/// The ORIGINAL full-sort algorithm: sort all (key, count) pairs ascending by
/// key, then derive modes/antimodes from the runs (inlined replica of the old
/// `modes_antimodes` + `modes_antimodes_from_runs` composition).
fn sort_baseline(map: &hashbrown::HashMap<Vec<u8>, u64>) -> ModesResult {
    let mut runs: Vec<(&Vec<u8>, u32)> = map
        .iter()
        .map(|(k, &c)| (k, u32::try_from(c).unwrap_or(u32::MAX)))
        .collect();
    if runs.len() > PARALLEL_THRESHOLD {
        runs.par_sort_unstable_by(|a, b| a.0.cmp(b.0));
    } else {
        runs.sort_unstable_by(|a, b| a.0.cmp(b.0));
    }
    let mut highest = 1_u32;
    let mut lowest = u32::MAX;
    for &(_, c) in &runs {
        highest = highest.max(c);
        lowest = lowest.min(c);
    }

    if runs.is_empty() {
        return ((Vec::new(), 0, 0), (Vec::new(), 0, 0));
    }
    if runs.len() == 1 {
        let (val, count) = runs.pop().unwrap();
        return ((vec![val.clone()], 1, count), (Vec::new(), 0, 0));
    }
    if highest == 1 {
        let total = runs.len();
        let take = total.min(10);
        let anti: Vec<Vec<u8>> = runs
            .into_iter()
            .take(take)
            .map(|(v, _)| v.clone())
            .collect();
        return ((Vec::new(), 0, 0), (anti, total, 1));
    }
    let mut modes = Vec::new();
    let mut anti = Vec::new();
    let mut mode_count = 0_usize;
    let mut anti_count = 0_usize;
    let mut anti_collected = 0_u32;
    for (v, c) in &runs {
        if *c == highest {
            modes.push((*v).clone());
            mode_count += 1;
        }
        if *c == lowest {
            anti_count += 1;
            if anti_collected < 10 {
                anti.push((*v).clone());
                anti_collected += 1;
            }
        }
    }
    ((modes, mode_count, highest), (anti, anti_count, lowest))
}

#[allow(clippy::type_complexity)]
fn shapes() -> Vec<(&'static str, Box<dyn Fn(usize) -> Vec<Vec<u8>>>)> {
    vec![
        ("all_unique", Box::new(gen_all_unique)),
        ("low_card_5000", Box::new(|n| gen_low_card(n, 5000))),
        ("high_card_100k", Box::new(|n| gen_high_card(n, 100_000))),
        ("uniform_x2", Box::new(gen_uniform_x2)),
    ]
}

fn bench_modes_antimodes(c: &mut Criterion) {
    let n = 1_000_000_usize;
    let mut group = c.benchmark_group("modes_antimodes");
    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(10);
    // Datasets generated lazily, one shape resident at a time (RSS pattern from
    // benches/freqhb.rs).
    for (label, make) in shapes() {
        let data = make(n);
        let mut freqs: Frequencies<Vec<u8>> = Frequencies::new();
        let mut mirror: hashbrown::HashMap<Vec<u8>, u64> = hashbrown::HashMap::new();
        for v in &data {
            freqs.add_borrowed(v);
            *mirror.entry_ref(v.as_slice()).or_insert(0) += 1;
        }
        drop(data);
        // Behavior-preservation proof before timing anything.
        assert_eq!(
            freqs.modes_antimodes(),
            sort_baseline(&mirror),
            "lib select path diverges from the original sort algorithm on {label}"
        );
        group.bench_with_input(BenchmarkId::new("sort_baseline", label), &mirror, |b, m| {
            b.iter(|| black_box(sort_baseline(m)));
        });
        group.bench_with_input(BenchmarkId::new("select_lib", label), &freqs, |b, f| {
            b.iter(|| black_box(f.modes_antimodes()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_modes_antimodes);
criterion_main!(benches);
