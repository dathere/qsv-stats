use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stats::{merge_all, OnlineStats};

const N: usize = 100_000;

fn gen_f64_random(n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Strictly-positive finite range. OnlineStats has a harmonic/geometric
        // sum fast path that is permanently disabled by the first non-positive
        // sample, so we keep all samples > 0 to exercise it.
        let raw = (s as f64) / (u64::MAX as f64);
        out.push(raw * 1_000_000.0 + 1.0);
    }
    out
}

fn gen_f64_random_with_nan(n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut s: u64 = 0xFEED_FACE_DEAD_BEEF;
    for i in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if i % 100 == 0 {
            out.push(f64::NAN);
        } else {
            // Strictly-positive (NaN is silently skipped by add_f64); keeps
            // the harmonic/geometric fast path engaged.
            let raw = (s as f64) / (u64::MAX as f64);
            out.push(raw * 1_000_000.0 + 1.0);
        }
    }
    out
}

fn gen_i64_random(n: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(n);
    let mut s: u64 = 0xCAFE_F00D_1337_BEEF;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push((s as i64).rem_euclid(1_000_000));
    }
    out
}

fn bench_add_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_add_f64");
    group.throughput(Throughput::Elements(N as u64));

    let datasets: &[(&str, Vec<f64>)] = &[
        ("random", gen_f64_random(N)),
        ("random_with_nan", gen_f64_random_with_nan(N)),
    ];

    for (label, data) in datasets {
        group.bench_with_input(BenchmarkId::new("add_f64", label), data, |b, data| {
            b.iter(|| {
                let mut os = OnlineStats::new();
                for &v in data {
                    os.add_f64(black_box(v));
                }
                black_box(os)
            });
        });

        // Compare with the generic add() path (ToPrimitive dispatch).
        group.bench_with_input(BenchmarkId::new("add_generic", label), data, |b, data| {
            b.iter(|| {
                let mut os = OnlineStats::new();
                for v in data {
                    os.add(black_box(v));
                }
                black_box(os)
            });
        });
    }
    group.finish();
}

fn bench_add_i64(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_add_i64");
    group.throughput(Throughput::Elements(N as u64));
    let data = gen_i64_random(N);
    group.bench_function("random", |b| {
        b.iter(|| {
            let mut os = OnlineStats::new();
            for v in &data {
                os.add(black_box(v));
            }
            black_box(os)
        });
    });
    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_merge");
    // Each iter is one merge of two pre-aggregated halves; throughput is
    // merges/sec, not elements/sec.
    group.throughput(Throughput::Elements(1));

    // Pre-build the two halves once so the bench isolates merge cost from
    // the dominant add-loop cost.
    let half_a_data = gen_f64_random(N / 2);
    let half_b_data = gen_f64_random_with_nan(N / 2);
    let mut a = OnlineStats::new();
    for &v in &half_a_data {
        a.add_f64(v);
    }
    let mut b_half = OnlineStats::new();
    for &v in &half_b_data {
        b_half.add_f64(v);
    }

    group.bench_function("two_chunks", |bench| {
        // OnlineStats: Copy — setup is two register copies per iter.
        bench.iter_batched(
            || (a, b_half),
            |(a, b_half)| {
                let merged = merge_all(vec![a, b_half].into_iter()).unwrap_or_default();
                black_box(merged)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_add_f64, bench_add_i64, bench_merge);
criterion_main!(benches);
