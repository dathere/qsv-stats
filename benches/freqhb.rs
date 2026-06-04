// Faithful comparison of Frequencies<Vec<u8>>'s std-HashMap backing vs a
// hashbrown 0.17 backing, for both hot paths qsv uses:
//   - build  (add_borrowed per cell)
//   - merge  (per-column partial tables combined via Commute::merge)
//
// hashbrown 0.16+ uses foldhash 0.2 as its DEFAULT hasher — the SAME hash
// qsv-stats already uses — so hashbrown::HashMap::new() here hashes identically
// to foldhash::HashMap. This isolates the table implementation, not the hash.
//
// Build: std path mirrors the library (get_mut -> insert-on-miss); hashbrown
// path uses entry_ref (single borrowed probe).
// Merge: both use the library's entry(owned-key) vacant/occupied pattern.
//
// RESULT (justifies the swap, 1M rows): hashbrown wins on both paths with
// identical hashing. Build: -7-8% high-cardinality (the dominant cost), -14-20%
// low-cardinality. Merge: -15% low-card, ~-3% high-card. foldhash stays as a
// dev-dependency solely to keep the `std` baseline in this bench.
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::collections::hash_map::Entry as StdEntry;

use hashbrown::hash_map::Entry as HbEntry;

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

fn gen_high_card_short(n: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| format!("{i:08}").into_bytes()).collect()
}

fn gen_high_card_long(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| format!("urn:example:record:{i:020}:payload-segment-xyz").into_bytes())
        .collect()
}

// ---- build ----
fn build_std(data: &[Vec<u8>]) -> u64 {
    let mut map: foldhash::HashMap<Vec<u8>, u64> = foldhash::HashMapExt::with_capacity(1024);
    for v in data {
        if let Some(c) = map.get_mut(v.as_slice()) {
            *c += 1;
        } else {
            map.insert(v.clone(), 1);
        }
    }
    map.len() as u64
}

fn build_hb(data: &[Vec<u8>]) -> u64 {
    // Default hasher == foldhash 0.2 (hashbrown 0.16+).
    let mut map: hashbrown::HashMap<Vec<u8>, u64> = hashbrown::HashMap::with_capacity(1024);
    for v in data {
        *map.entry_ref(v.as_slice()).or_insert(0) += 1;
    }
    map.len() as u64
}

// ---- merge ----
// Build `parts` partial tables from contiguous slices (mirrors qsv's per-chunk
// frequency tables), then time merging them all into one.
fn make_std_partials(data: &[Vec<u8>], parts: usize) -> Vec<foldhash::HashMap<Vec<u8>, u64>> {
    let chunk = data.len().div_ceil(parts);
    data.chunks(chunk)
        .map(|c| {
            let mut m: foldhash::HashMap<Vec<u8>, u64> = foldhash::HashMapExt::with_capacity(256);
            for v in c {
                *m.entry(v.clone()).or_insert(0) += 1;
            }
            m
        })
        .collect()
}

fn merge_std(mut partials: Vec<foldhash::HashMap<Vec<u8>, u64>>) -> u64 {
    let mut acc = partials.swap_remove(0);
    for p in partials {
        acc.reserve(p.len());
        for (k, v2) in p {
            match acc.entry(k) {
                StdEntry::Vacant(e) => {
                    e.insert(v2);
                }
                StdEntry::Occupied(mut e) => *e.get_mut() += v2,
            }
        }
    }
    acc.len() as u64
}

fn make_hb_partials(data: &[Vec<u8>], parts: usize) -> Vec<hashbrown::HashMap<Vec<u8>, u64>> {
    let chunk = data.len().div_ceil(parts);
    data.chunks(chunk)
        .map(|c| {
            let mut m: hashbrown::HashMap<Vec<u8>, u64> = hashbrown::HashMap::with_capacity(256);
            for v in c {
                *m.entry_ref(v.as_slice()).or_insert(0) += 1;
            }
            m
        })
        .collect()
}

fn merge_hb(mut partials: Vec<hashbrown::HashMap<Vec<u8>, u64>>) -> u64 {
    let mut acc = partials.swap_remove(0);
    for p in partials {
        acc.reserve(p.len());
        for (k, v2) in p {
            match acc.entry(k) {
                HbEntry::Vacant(e) => {
                    e.insert(v2);
                }
                HbEntry::Occupied(mut e) => *e.get_mut() += v2,
            }
        }
    }
    acc.len() as u64
}

// Datasets are generated lazily, one shape per loop iteration, so only a single
// 1M-row Vec<Vec<u8>> is resident at a time (avoids holding all shapes' peak RSS
// at once).
#[allow(clippy::type_complexity)]
fn shapes() -> Vec<(&'static str, Box<dyn Fn(usize) -> Vec<Vec<u8>>>)> {
    vec![
        ("low_card_50", Box::new(|n| gen_low_card(n, 50))),
        ("low_card_5000", Box::new(|n| gen_low_card(n, 5000))),
        ("high_card_short", Box::new(gen_high_card_short)),
        ("high_card_long", Box::new(gen_high_card_long)),
    ]
}

fn bench_build(c: &mut Criterion) {
    let n = 1_000_000usize;
    let mut group = c.benchmark_group("build");
    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(20);
    for (label, make) in shapes() {
        let data = make(n);
        assert_eq!(build_std(&data), build_hb(&data));
        group.bench_with_input(BenchmarkId::new("std", label), &data, |b, d| {
            b.iter(|| black_box(build_std(d)));
        });
        group.bench_with_input(BenchmarkId::new("hashbrown", label), &data, |b, d| {
            b.iter(|| black_box(build_hb(d)));
        });
    }
    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let n = 1_000_000usize;
    let parts = 16;
    let mut group = c.benchmark_group("merge");
    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(20);
    // Merge only exercises the two cardinality extremes.
    for (label, make) in shapes()
        .into_iter()
        .filter(|(l, _)| *l == "low_card_5000" || *l == "high_card_short")
    {
        let data = make(n);
        let std_parts = make_std_partials(&data, parts);
        let hb_parts = make_hb_partials(&data, parts);
        drop(data); // partials own their keys; free the source dataset before benching
        group.bench_with_input(BenchmarkId::new("std", label), &std_parts, |b, p| {
            b.iter_batched(
                || p.clone(),
                |p| black_box(merge_std(p)),
                criterion::BatchSize::LargeInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("hashbrown", label), &hb_parts, |b, p| {
            b.iter_batched(
                || p.clone(),
                |p| black_box(merge_hb(p)),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build, bench_merge);
criterion_main!(benches);
