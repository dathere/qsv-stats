// Isolates the cost of the f64 sort COMPARATOR used by Unsorted<f64>.
//
// Unsorted<T> sorts Vec<Partial<T>> via Partial::cmp, which is
// `partial_cmp(..).unwrap_or(Ordering::Less)`. For f64 this is the dominant
// cost of median/quartiles/mad/percentiles (they all reuse one cached sort).
//
// This bench reproduces that comparator on a plain Vec<f64> (Partial is
// repr-equivalent / private) and compares it against f64::total_cmp, the
// canonical branchless comparator, for both the sequential and rayon parallel
// sort paths the library selects around PARALLEL_THRESHOLD (10_000).
//
// RESULT (regression guard): total_cmp is SLOWER across the board — ~15-22% on
// random data and up to ~95% on nearly-sorted data (its bit-twiddling defeats
// pdqsort's adaptive fast paths). Keep the partial_cmp().unwrap_or(Less)
// comparator; do NOT "optimize" it to total_cmp.
use std::cmp::Ordering;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rayon::slice::ParallelSliceMut;

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    }
}

fn gen_random(n: usize) -> Vec<f64> {
    let mut next = lcg(0x1234_5678_9ABC_DEF0);
    (0..n)
        .map(|_| (next() as f64) / (u64::MAX as f64) * 1_000_000.0 + 1.0)
        .collect()
}

// Many real columns are near-sorted (timestamps, ids). pdqsort is adaptive,
// so comparator cost is more exposed when the sort does less swapping.
fn gen_nearly_sorted(n: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut next = lcg(0xDEAD_BEEF_CAFE_F00D);
    // perturb ~1% of positions
    for _ in 0..(n / 100) {
        let i = (next() as usize) % n;
        let j = (next() as usize) % n;
        v.swap(i, j);
    }
    v
}

fn gen_low_cardinality(n: usize) -> Vec<f64> {
    let mut next = lcg(0x0BAD_C0DE_F00D_1337);
    (0..n).map(|_| (next() % 50) as f64).collect()
}

#[inline]
fn partial_cmp_or_less(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Less)
}

fn bench_sort(c: &mut Criterion) {
    // Span the library's sequential/parallel split (PARALLEL_THRESHOLD = 10_000).
    let sizes = [1_000usize, 50_000, 1_000_000];
    let shapes: &[(&str, fn(usize) -> Vec<f64>)] = &[
        ("random", gen_random),
        ("nearly_sorted", gen_nearly_sorted),
        ("low_card", gen_low_cardinality),
    ];

    for &n in &sizes {
        let mut group = c.benchmark_group(format!("sort_{n}"));
        group.throughput(Throughput::Elements(n as u64));
        let parallel = n >= 10_000;

        for (shape, make) in shapes {
            let data = make(n);

            // Current library comparator: partial_cmp().unwrap_or(Less)
            group.bench_with_input(BenchmarkId::new("partial_or_less", shape), &data, |b, d| {
                b.iter_batched(
                    || d.clone(),
                    |mut v| {
                        if parallel {
                            v.par_sort_unstable_by(partial_cmp_or_less);
                        } else {
                            v.sort_unstable_by(partial_cmp_or_less);
                        }
                        black_box(v)
                    },
                    criterion::BatchSize::LargeInput,
                );
            });

            // Candidate: f64::total_cmp
            group.bench_with_input(BenchmarkId::new("total_cmp", shape), &data, |b, d| {
                b.iter_batched(
                    || d.clone(),
                    |mut v| {
                        if parallel {
                            v.par_sort_unstable_by(f64::total_cmp);
                        } else {
                            v.sort_unstable_by(f64::total_cmp);
                        }
                        black_box(v)
                    },
                    criterion::BatchSize::LargeInput,
                );
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_sort);
criterion_main!(benches);
