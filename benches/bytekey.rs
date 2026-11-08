// Regression guard for `Frequencies<ByteKey>` (inline small keys) vs
// `Frequencies<Vec<u8>>`, using only the library's own types.
//
// Each iteration builds a table with `add_borrowed` (qsv's per-cell hot path
// for `modes`) and DROPS it inside the timed region: a `Vec<u8>` table pays
// one heap alloc + free per unique key, and that free is part of the real
// cost. `merge` mirrors qsv's indexed path (per-chunk partial tables combined
// via `Commute::merge`).
//
// Datasets are generated lazily, one shape at a time, so only one ~1M-row
// input is resident (same pattern as benches/sortcmp.rs and freqhb.rs).
//
// Shapes:
//   ids_8b       high-cardinality 8-byte IDs           (inline)
//   dates_22b    `MM/DD/YYYY HH:MM:SS AM` timestamps   (inline; > smallvec's 16)
//   low_card_13b 5,000-value categorical               (inline; few allocations)
//   urn_45b      high-cardinality 45-byte keys         (spilled; guards the
//                                                      heap-fallback overhead)
//
// RESULT (1M rows, macOS aarch64, system allocator, quiet machine;
// vec_u8 -> bytekey):
//   build+drop  ids_8b        47.9 -> 25.3 ms  (-47%)
//               dates_22b     66.4 -> 27.8 ms  (-58%)
//               low_card_13b  5.26 -> 4.72 ms  (-10%)
//               urn_45b       86.6 -> 90.9 ms  (+5%, spilled keys)
//   merge (16)  ids_8b        37.1 -> 23.6 ms  (-36%)
//               dates_22b     53.8 -> 23.6 ms  (-56%)
//               low_card_13b  0.91 -> 0.53 ms  (-41%)
//               urn_45b       64.6 -> 70.0 ms  (+8%, spilled keys)
// The spilled-key overhead did not show up end to end: qsv `stats --cardinality
// --mode` on NYC 311's 40-byte, 376k-unique Location column alone was -0.8%
// wall. Whole-file NYC 311 (jemalloc): -17% unindexed, -27% indexed, output
// byte-identical.
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use stats::{ByteKey, Commute, Frequencies};

const N: usize = 1_000_000;
const CHUNKS: usize = 16;

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s >> 16
    }
}

fn gen_ids(n: usize) -> Vec<Vec<u8>> {
    let mut next = lcg(0x1D5);
    (0..n)
        .map(|_| format!("{:08}", next() % 100_000_000).into_bytes())
        .collect()
}

fn gen_dates(n: usize) -> Vec<Vec<u8>> {
    let mut next = lcg(0xDA7E);
    (0..n)
        .map(|_| {
            let r = next();
            let (mo, d, y) = (r % 12 + 1, (r >> 4) % 28 + 1, 2010 + (r >> 9) % 11);
            let (h, mi, s) = ((r >> 13) % 12 + 1, (r >> 17) % 60, (r >> 23) % 60);
            let ampm = if (r >> 29) & 1 == 0 { "AM" } else { "PM" };
            format!("{mo:02}/{d:02}/{y} {h:02}:{mi:02}:{s:02} {ampm}").into_bytes()
        })
        .collect()
}

fn gen_low_card(n: usize) -> Vec<Vec<u8>> {
    let mut next = lcg(0xCA7);
    (0..n)
        .map(|_| format!("category_{:04}", next() % 5_000).into_bytes())
        .collect()
}

fn gen_urn(n: usize) -> Vec<Vec<u8>> {
    let mut next = lcg(0x0E11);
    (0..n)
        .map(|_| format!("urn:example:record:{:020}:payload", next()).into_bytes())
        .collect()
}

type Gen = fn(usize) -> Vec<Vec<u8>>;

fn shapes() -> [(&'static str, Gen); 4] {
    [
        ("ids_8b", gen_ids),
        ("dates_22b", gen_dates),
        ("low_card_13b", gen_low_card),
        ("urn_45b", gen_urn),
    ]
}

fn build_vec(data: &[Vec<u8>]) -> Frequencies<Vec<u8>> {
    let mut f: Frequencies<Vec<u8>> = Frequencies::new();
    for v in data {
        f.add_borrowed(v);
    }
    f
}

fn build_key(data: &[Vec<u8>]) -> Frequencies<ByteKey> {
    let mut f: Frequencies<ByteKey> = Frequencies::new();
    for v in data {
        f.add_borrowed(v);
    }
    f
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytekey_build_drop");
    group.sample_size(10);
    group.throughput(Throughput::Elements(N as u64));
    for (name, generate) in shapes() {
        let data = generate(N);
        // Equivalence check before timing.
        let (fv, fk) = (build_vec(&data), build_key(&data));
        assert_eq!(fv.len(), fk.len());
        assert_eq!(fv.modes_antimodes(), {
            let ((m, mc, mo), (a, ac, ao)) = fk.modes_antimodes();
            let m: Vec<Vec<u8>> = m.into_iter().map(Vec::from).collect();
            let a: Vec<Vec<u8>> = a.into_iter().map(Vec::from).collect();
            ((m, mc, mo), (a, ac, ao))
        });
        drop((fv, fk));

        group.bench_with_input(BenchmarkId::new("vec_u8", name), &data, |b, d| {
            b.iter(|| drop(black_box(build_vec(d))));
        });
        group.bench_with_input(BenchmarkId::new("bytekey", name), &data, |b, d| {
            b.iter(|| drop(black_box(build_key(d))));
        });
    }
    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytekey_merge");
    group.sample_size(10);
    group.throughput(Throughput::Elements(N as u64));
    for (name, generate) in shapes() {
        let data = generate(N);
        let chunk = N.div_ceil(CHUNKS);
        let pv: Vec<Frequencies<Vec<u8>>> = data.chunks(chunk).map(build_vec).collect();
        let pk: Vec<Frequencies<ByteKey>> = data.chunks(chunk).map(build_key).collect();
        drop(data);

        group.bench_with_input(BenchmarkId::new("vec_u8", name), &pv, |b, p| {
            b.iter_batched(
                || p.clone(),
                |parts| {
                    let mut it = parts.into_iter();
                    let mut acc = it.next().unwrap();
                    for f in it {
                        acc.merge(f);
                    }
                    drop(black_box(acc));
                },
                criterion::BatchSize::LargeInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("bytekey", name), &pk, |b, p| {
            b.iter_batched(
                || p.clone(),
                |parts| {
                    let mut it = parts.into_iter();
                    let mut acc = it.next().unwrap();
                    for f in it {
                        acc.merge(f);
                    }
                    drop(black_box(acc));
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build, bench_merge);
criterion_main!(benches);
