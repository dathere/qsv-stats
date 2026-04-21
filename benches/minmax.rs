use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stats::MinMax;

const N: usize = 100_000;

fn gen_bytes_sorted_asc(n: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| format!("{i:016}").into_bytes()).collect()
}

fn gen_bytes_sorted_desc(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .rev()
        .map(|i| format!("{i:016}").into_bytes())
        .collect()
}

fn gen_bytes_random(n: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(n);
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push(format!("{:016x}", s).into_bytes());
    }
    out
}

fn gen_bytes_mostly_sorted(n: usize) -> Vec<Vec<u8>> {
    let mut v = gen_bytes_sorted_asc(n);
    let mut s: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let swaps = n / 100;
    for _ in 0..swaps {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let i = (s as usize) % n;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (s as usize) % n;
        v.swap(i, j);
    }
    v
}

fn gen_f64_random_with_nan(n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    for i in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if i % 100 == 0 {
            out.push(f64::NAN);
        } else {
            out.push(f64::from_bits(s) % 1_000_000.0);
        }
    }
    out
}

fn gen_i64_random(n: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(n);
    let mut s: u64 = 0xCAFE_F00D_1337_BEEF;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push(s as i64);
    }
    out
}

fn gen_string_random(n: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n);
    let mut s: u64 = 0xFEED_FACE_DEAD_BEEF;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push(format!("{:016x}", s));
    }
    out
}

fn bench_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_bytes");
    group.throughput(Throughput::Elements(N as u64));

    let datasets: &[(&str, Vec<Vec<u8>>)] = &[
        ("asc", gen_bytes_sorted_asc(N)),
        ("desc", gen_bytes_sorted_desc(N)),
        ("random", gen_bytes_random(N)),
        ("mostly_sorted", gen_bytes_mostly_sorted(N)),
    ];

    for (label, data) in datasets {
        group.bench_with_input(BenchmarkId::new("add_bytes", label), data, |b, data| {
            b.iter(|| {
                let mut mm: MinMax<Vec<u8>> = MinMax::new();
                for row in data {
                    mm.add_bytes(black_box(row));
                }
                black_box(mm)
            });
        });

        group.bench_with_input(BenchmarkId::new("add", label), data, |b, data| {
            b.iter(|| {
                let mut mm: MinMax<Vec<u8>> = MinMax::new();
                for row in data {
                    mm.add(black_box(row.clone()));
                }
                black_box(mm)
            });
        });
    }
    group.finish();
}

fn bench_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_f64");
    group.throughput(Throughput::Elements(N as u64));
    let data = gen_f64_random_with_nan(N);
    group.bench_function("random_with_nan", |b| {
        b.iter(|| {
            let mut mm: MinMax<f64> = MinMax::new();
            for v in &data {
                mm.add(black_box(*v));
            }
            black_box(mm)
        });
    });
    group.finish();
}

fn bench_i64(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_i64");
    group.throughput(Throughput::Elements(N as u64));
    let data = gen_i64_random(N);
    group.bench_function("random", |b| {
        b.iter(|| {
            let mut mm: MinMax<i64> = MinMax::new();
            for v in &data {
                mm.add(black_box(*v));
            }
            black_box(mm)
        });
    });
    group.finish();
}

fn bench_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_string");
    group.throughput(Throughput::Elements(N as u64));
    let data = gen_string_random(N);

    group.bench_function("add_random", |b| {
        b.iter(|| {
            let mut mm: MinMax<String> = MinMax::new();
            for v in &data {
                mm.add(black_box(v.clone()));
            }
            black_box(mm)
        });
    });

    group.bench_function("add_ref_random", |b| {
        b.iter(|| {
            let mut mm: MinMax<String> = MinMax::new();
            for v in &data {
                mm.add_ref(black_box(v));
            }
            black_box(mm)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_bytes, bench_f64, bench_i64, bench_string);
criterion_main!(benches);
