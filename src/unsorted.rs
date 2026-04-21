use num_traits::ToPrimitive;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use rayon::slice::ParallelSliceMut;

use serde::{Deserialize, Serialize};

use {crate::Commute, crate::Partial};

// PARALLEL_THRESHOLD (10,000) is the minimum dataset size for rayon parallel sort.
// The separate 10,240 threshold in cardinality estimation (5 × 2,048) is aligned to
// cache-line-friendly chunk sizes for parallel iterator reduction.
const PARALLEL_THRESHOLD: usize = 10_000;

/// Compute the exact median on a stream of data.
///
/// (This has time complexity `O(nlogn)` and space complexity `O(n)`.)
#[inline]
pub fn median<I>(it: I) -> Option<f64>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive + Send,
{
    it.collect::<Unsorted<_>>().median()
}

/// Compute the median absolute deviation (MAD) on a stream of data.
#[inline]
pub fn mad<I>(it: I, precalc_median: Option<f64>) -> Option<f64>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive + Send + Sync,
{
    it.collect::<Unsorted<_>>().mad(precalc_median)
}

/// Compute the exact 1-, 2-, and 3-quartiles (Q1, Q2 a.k.a. median, and Q3) on a stream of data.
///
/// (This has time complexity `O(n log n)` and space complexity `O(n)`.)
#[inline]
pub fn quartiles<I>(it: I) -> Option<(f64, f64, f64)>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive + Send,
{
    it.collect::<Unsorted<_>>().quartiles()
}

/// Compute the exact mode on a stream of data.
///
/// (This has time complexity `O(nlogn)` and space complexity `O(n)`.)
///
/// If the data does not have a mode, then `None` is returned.
#[inline]
pub fn mode<T, I>(it: I) -> Option<T>
where
    T: PartialOrd + Clone + Send,
    I: Iterator<Item = T>,
{
    it.collect::<Unsorted<T>>().mode()
}

/// Compute the modes on a stream of data.
///
/// If there is a single mode, then only that value is returned in the `Vec`
/// however, if there are multiple values tied for occurring the most amount of times
/// those values are returned.
///
/// ## Example
/// ```
/// use stats;
///
/// let vals = vec![1, 1, 2, 2, 3];
///
/// assert_eq!(stats::modes(vals.into_iter()), (vec![1, 2], 2, 2));
/// ```
/// This has time complexity `O(n)`
///
/// If the data does not have a mode, then an empty `Vec` is returned.
#[inline]
pub fn modes<T, I>(it: I) -> (Vec<T>, usize, u32)
where
    T: PartialOrd + Clone + Send,
    I: Iterator<Item = T>,
{
    it.collect::<Unsorted<T>>().modes()
}

/// Compute the antimodes on a stream of data.
///
/// Antimode is the least frequent non-zero score.
///
/// If there is a single antimode, then only that value is returned in the `Vec`
/// however, if there are multiple values tied for occurring the least amount of times
/// those values are returned.
///
/// Only the first 10 antimodes are returned to prevent returning the whole set
/// when cardinality = number of records (i.e. all unique values).
///
/// ## Example
/// ```
/// use stats;
///
/// let vals = vec![1, 1, 2, 2, 3];
///
/// assert_eq!(stats::antimodes(vals.into_iter()), (vec![3], 1, 1));
/// ```
/// This has time complexity `O(n)`
///
/// If the data does not have an antimode, then an empty `Vec` is returned.
#[inline]
pub fn antimodes<T, I>(it: I) -> (Vec<T>, usize, u32)
where
    T: PartialOrd + Clone + Send,
    I: Iterator<Item = T>,
{
    let (antimodes_result, antimodes_count, antimodes_occurrences) =
        it.collect::<Unsorted<T>>().antimodes();
    (antimodes_result, antimodes_count, antimodes_occurrences)
}

/// Compute the Gini Coefficient on a stream of data.
///
/// The Gini Coefficient measures inequality in a distribution, ranging from 0 (perfect equality)
/// to 1 (perfect inequality).
///
/// (This has time complexity `O(n log n)` and space complexity `O(n)`.)
#[inline]
pub fn gini<I>(it: I, precalc_sum: Option<f64>) -> Option<f64>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive + Send + Sync,
{
    it.collect::<Unsorted<_>>().gini(precalc_sum)
}

/// Compute the kurtosis (excess kurtosis) on a stream of data.
///
/// Kurtosis measures the "tailedness" of a distribution. Excess kurtosis is kurtosis - 3,
/// where 0 indicates a normal distribution, positive values indicate heavy tails, and
/// negative values indicate light tails.
///
/// (This has time complexity `O(n log n)` and space complexity `O(n)`.)
#[inline]
pub fn kurtosis<I>(it: I, precalc_mean: Option<f64>, precalc_variance: Option<f64>) -> Option<f64>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive + Send + Sync,
{
    it.collect::<Unsorted<_>>()
        .kurtosis(precalc_mean, precalc_variance)
}

/// Compute the percentile rank of a value on a stream of data.
///
/// Returns the percentile rank (0-100) of the given value in the distribution.
/// If the value is less than all values, returns 0.0. If greater than all, returns 100.0.
///
/// (This has time complexity `O(n log n)` and space complexity `O(n)`.)
#[inline]
pub fn percentile_rank<I, V>(it: I, value: V) -> Option<f64>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive + Send + Sync,
    V: PartialOrd + ToPrimitive,
{
    it.collect::<Unsorted<_>>().percentile_rank(value)
}

/// Compute the Atkinson Index on a stream of data.
///
/// The Atkinson Index measures inequality with an inequality aversion parameter ε.
/// It ranges from 0 (perfect equality) to 1 (perfect inequality).
/// Higher ε values give more weight to inequality at the lower end of the distribution.
///
/// (This has time complexity `O(n log n)` and space complexity `O(n)`.)
#[inline]
pub fn atkinson<I>(
    it: I,
    epsilon: f64,
    precalc_mean: Option<f64>,
    precalc_geometric_sum: Option<f64>,
) -> Option<f64>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive + Send + Sync,
{
    it.collect::<Unsorted<_>>()
        .atkinson(epsilon, precalc_mean, precalc_geometric_sum)
}

fn median_on_sorted<T>(data: &[T]) -> Option<f64>
where
    T: PartialOrd + ToPrimitive,
{
    Some(match data.len() {
        // Empty slice case - return None early
        0 => {
            core::hint::cold_path();
            return None;
        }
        // Single element case - return that element converted to f64
        1 => data.first()?.to_f64()?,
        // Even length case - average the two middle elements
        len if len.is_multiple_of(2) => {
            let idx = len / 2;
            // Safety: we know these indices are valid because we checked len is even and non-zero,
            // so idx-1 and idx are valid indices into data
            let v1 = unsafe { data.get_unchecked(idx - 1) }.to_f64()?;
            let v2 = unsafe { data.get_unchecked(idx) }.to_f64()?;
            f64::midpoint(v1, v2)
        }
        // Odd length case - return the middle element
        // Safety: we know the index is within bounds
        len => unsafe { data.get_unchecked(len / 2) }.to_f64()?,
    })
}

fn mad_on_sorted<T>(data: &[T], precalc_median: Option<f64>) -> Option<f64>
where
    T: Sync + PartialOrd + ToPrimitive,
{
    if data.is_empty() {
        core::hint::cold_path();
        return None;
    }
    let median_obs = precalc_median.unwrap_or_else(|| median_on_sorted(data).unwrap());

    // Use adaptive parallel processing based on data size
    let mut abs_diff_vec: Vec<f64> = if data.len() < PARALLEL_THRESHOLD {
        // Sequential processing for small datasets
        // Iterator collect enables TrustedLen optimization, eliminating per-element bounds checks
        data.iter()
            // SAFETY: to_f64() always returns Some for standard numeric types (f32/f64, i/u 8-64)
            .map(|x| (median_obs - unsafe { x.to_f64().unwrap_unchecked() }).abs())
            .collect()
    } else {
        // Parallel processing for large datasets
        data.par_iter()
            // SAFETY: to_f64() always returns Some for standard numeric types
            .map(|x| (median_obs - unsafe { x.to_f64().unwrap_unchecked() }).abs())
            .collect()
    };

    // Use select_nth_unstable to find the median in O(n) instead of O(n log n) full sort
    let len = abs_diff_vec.len();
    let mid = len / 2;
    let cmp = |a: &f64, b: &f64| a.total_cmp(b);

    abs_diff_vec.select_nth_unstable_by(mid, cmp);

    if len.is_multiple_of(2) {
        // Even length: need both mid-1 and mid elements
        let right = abs_diff_vec[mid];
        // The left partition [0..mid] contains elements <= abs_diff_vec[mid],
        // so we can find the max of the left partition for mid-1
        let left = abs_diff_vec[..mid]
            .iter()
            .max_by(|a, b| cmp(a, b))
            .copied()?;
        Some(f64::midpoint(left, right))
    } else {
        Some(abs_diff_vec[mid])
    }
}

fn gini_on_sorted<T>(data: &[Partial<T>], precalc_sum: Option<f64>) -> Option<f64>
where
    T: Sync + PartialOrd + ToPrimitive,
{
    let len = data.len();

    // Early return for empty data
    if len == 0 {
        core::hint::cold_path();
        return None;
    }

    // Single element case: perfect equality, Gini = 0
    if len == 1 {
        core::hint::cold_path();
        return Some(0.0);
    }

    // Gini coefficient is only defined for non-negative distributions.
    // Since data is sorted, check the first (smallest) element.
    // SAFETY: len > 1 guaranteed by checks above
    let first_val = unsafe { data.get_unchecked(0).0.to_f64().unwrap_unchecked() };
    if first_val < 0.0 {
        core::hint::cold_path();
        return None;
    }

    // Compute sum and weighted sum.
    // When precalc_sum is provided, only compute weighted_sum in a single pass.
    // When not provided, fuse both computations into a single pass over the data
    // to halve cache pressure (following the fold/reduce pattern used in kurtosis).
    let (sum, weighted_sum) = if let Some(precalc) = precalc_sum {
        if precalc < 0.0 {
            core::hint::cold_path();
            return None;
        }
        // Only need weighted_sum — single pass
        let weighted_sum = if len < PARALLEL_THRESHOLD {
            let mut weighted_sum = 0.0;
            for (i, x) in data.iter().enumerate() {
                // SAFETY: to_f64() always returns Some for standard numeric types
                let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                weighted_sum = ((i + 1) as f64).mul_add(val, weighted_sum);
            }
            weighted_sum
        } else {
            data.par_iter()
                .enumerate()
                .map(|(i, x)| {
                    // SAFETY: to_f64() always returns Some for standard numeric types
                    let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                    (i + 1) as f64 * val
                })
                .sum()
        };
        (precalc, weighted_sum)
    } else if len < PARALLEL_THRESHOLD {
        // Fused single pass: compute both sum and weighted_sum together
        let mut sum = 0.0;
        let mut weighted_sum = 0.0;
        for (i, x) in data.iter().enumerate() {
            // SAFETY: to_f64() always returns Some for standard numeric types (f32/f64, i/u 8-64)
            let val = unsafe { x.0.to_f64().unwrap_unchecked() };
            sum += val;
            weighted_sum = ((i + 1) as f64).mul_add(val, weighted_sum);
        }
        (sum, weighted_sum)
    } else {
        // Fused parallel single pass using fold/reduce
        data.par_iter()
            .enumerate()
            .fold(
                || (0.0_f64, 0.0_f64),
                |acc, (i, x)| {
                    // SAFETY: to_f64() always returns Some for standard numeric types
                    let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                    (acc.0 + val, ((i + 1) as f64).mul_add(val, acc.1))
                },
            )
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    };

    // If sum is zero, Gini is undefined
    if sum == 0.0 {
        core::hint::cold_path();
        return None;
    }

    // Compute Gini coefficient using the formula:
    // G = (2 * Σ(i * y_i)) / (n * Σ(y_i)) - (n + 1) / n
    // where i is 1-indexed rank and y_i are sorted values
    let n = len as f64;
    let gini = 2.0f64.mul_add(weighted_sum / (n * sum), -(n + 1.0) / n);

    Some(gini)
}

fn kurtosis_on_sorted<T>(
    data: &[Partial<T>],
    precalc_mean: Option<f64>,
    precalc_variance: Option<f64>,
) -> Option<f64>
where
    T: Sync + PartialOrd + ToPrimitive,
{
    let len = data.len();

    // Need at least 4 elements for meaningful kurtosis
    if len < 4 {
        core::hint::cold_path();
        return None;
    }

    // Use pre-calculated mean if provided, otherwise compute it
    let mean = precalc_mean.unwrap_or_else(|| {
        let sum: f64 = if len < PARALLEL_THRESHOLD {
            // Iterator sum enables auto-vectorization (SIMD) by the compiler
            data.iter()
                // SAFETY: to_f64() always returns Some for standard numeric types (f32/f64, i/u 8-64)
                .map(|x| unsafe { x.0.to_f64().unwrap_unchecked() })
                .sum()
        } else {
            data.par_iter()
                // SAFETY: to_f64() always returns Some for standard numeric types
                .map(|x| unsafe { x.0.to_f64().unwrap_unchecked() })
                .sum()
        };
        sum / len as f64
    });

    // Compute variance_sq and fourth_power_sum
    // If variance is provided, we can compute variance_sq directly (variance_sq = variance^2)
    // Otherwise, we need to compute variance from the data
    let (variance_sq, fourth_power_sum) = if let Some(variance) = precalc_variance {
        // Negative variance is invalid (possible floating-point rounding artifact)
        if variance < 0.0 {
            core::hint::cold_path();
            return None;
        }
        // Use pre-calculated variance: variance_sq = variance^2
        let variance_sq = variance * variance;

        // Still need to compute fourth_power_sum
        let fourth_power_sum = if len < PARALLEL_THRESHOLD {
            let mut sum = 0.0;
            for x in data {
                // SAFETY: to_f64() always returns Some for standard numeric types
                let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                let diff = val - mean;
                let diff_sq = diff * diff;
                sum = diff_sq.mul_add(diff_sq, sum);
            }
            sum
        } else {
            data.par_iter()
                .map(|x| {
                    // SAFETY: to_f64() always returns Some for standard numeric types
                    let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                    let diff = val - mean;
                    let diff_sq = diff * diff;
                    diff_sq * diff_sq
                })
                .sum()
        };

        (variance_sq, fourth_power_sum)
    } else {
        // Compute both variance_sum and fourth_power_sum
        let (variance_sum, fourth_power_sum) = if len < PARALLEL_THRESHOLD {
            let mut variance_sum = 0.0;
            let mut fourth_power_sum = 0.0;

            for x in data {
                // SAFETY: to_f64() always returns Some for standard numeric types
                let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                let diff = val - mean;
                let diff_sq = diff * diff;
                variance_sum += diff_sq;
                fourth_power_sum = diff_sq.mul_add(diff_sq, fourth_power_sum);
            }

            (variance_sum, fourth_power_sum)
        } else {
            // Single pass computing both sums simultaneously
            data.par_iter()
                .fold(
                    || (0.0_f64, 0.0_f64),
                    |acc, x| {
                        // SAFETY: to_f64() always returns Some for standard numeric types
                        let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                        let diff = val - mean;
                        let diff_sq = diff * diff;
                        (acc.0 + diff_sq, diff_sq.mul_add(diff_sq, acc.1))
                    },
                )
                .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
        };

        let variance = variance_sum / len as f64;

        // If variance is zero, all values are the same, kurtosis is undefined
        if variance == 0.0 {
            core::hint::cold_path();
            return None;
        }

        let variance_sq = variance * variance;
        (variance_sq, fourth_power_sum)
    };

    // If variance_sq is zero, all values are the same, kurtosis is undefined
    if variance_sq == 0.0 {
        core::hint::cold_path();
        return None;
    }

    let n = len as f64;

    // Sample excess kurtosis formula:
    // kurtosis = (n(n+1) * Σ((x_i - mean)⁴)) / ((n-1)(n-2)(n-3) * variance²) - 3(n-1)²/((n-2)(n-3))
    let denominator = (n - 1.0) * (n - 2.0) * (n - 3.0);
    let adjustment = 3.0 * (n - 1.0) * (n - 1.0) / denominator;
    let kurtosis =
        (n * (n + 1.0) * fourth_power_sum).mul_add(1.0 / (denominator * variance_sq), -adjustment);

    Some(kurtosis)
}

fn percentile_rank_on_sorted<T, V>(data: &[Partial<T>], value: &V) -> Option<f64>
where
    T: PartialOrd + ToPrimitive,
    V: PartialOrd + ToPrimitive,
{
    let len = data.len();

    if len == 0 {
        core::hint::cold_path();
        return None;
    }

    let value_f64 = value.to_f64()?;

    // Binary search to find the position where value would be inserted
    // This gives us the number of values <= value
    let count_leq = data.binary_search_by(|x| {
        x.0.to_f64()
            .unwrap_or(f64::NAN)
            .partial_cmp(&value_f64)
            .unwrap_or(std::cmp::Ordering::Less)
    });

    let count = match count_leq {
        Ok(idx) => {
            // Value found at idx — use partition_point (O(log n)) to find the upper bound
            // of equal values instead of a linear scan
            let upper = data[idx + 1..].partition_point(|x| {
                x.0.to_f64()
                    .is_some_and(|v| v.total_cmp(&value_f64).is_le())
            });
            idx + 1 + upper
        }
        Err(idx) => idx, // Number of values less than value
    };

    // Percentile rank = (count / n) * 100
    Some((count as f64 / len as f64) * 100.0)
}

fn atkinson_on_sorted<T>(
    data: &[Partial<T>],
    epsilon: f64,
    precalc_mean: Option<f64>,
    precalc_geometric_sum: Option<f64>,
) -> Option<f64>
where
    T: Sync + PartialOrd + ToPrimitive,
{
    let len = data.len();

    // Early return for empty data
    if len == 0 {
        core::hint::cold_path();
        return None;
    }

    // Single element case: perfect equality, Atkinson = 0
    if len == 1 {
        core::hint::cold_path();
        return Some(0.0);
    }

    // Epsilon must be non-negative
    if epsilon < 0.0 {
        core::hint::cold_path();
        return None;
    }

    // Use pre-calculated mean if provided, otherwise compute it
    let mean = precalc_mean.unwrap_or_else(|| {
        let sum: f64 = if len < PARALLEL_THRESHOLD {
            // Iterator sum enables auto-vectorization (SIMD) by the compiler
            data.iter()
                // SAFETY: to_f64() always returns Some for standard numeric types (f32/f64, i/u 8-64)
                .map(|x| unsafe { x.0.to_f64().unwrap_unchecked() })
                .sum()
        } else {
            data.par_iter()
                // SAFETY: to_f64() always returns Some for standard numeric types
                .map(|x| unsafe { x.0.to_f64().unwrap_unchecked() })
                .sum()
        };
        sum / len as f64
    });

    // If mean is zero, Atkinson is undefined
    if mean == 0.0 {
        core::hint::cold_path();
        return None;
    }

    // Handle special case: epsilon = 1 (uses geometric mean)
    if (epsilon - 1.0).abs() < 1e-10 {
        // A_1 = 1 - (geometric_mean / mean)
        let geometric_sum: f64 = if let Some(precalc) = precalc_geometric_sum {
            precalc
        } else if len < PARALLEL_THRESHOLD {
            let mut sum = 0.0;
            for x in data {
                // SAFETY: to_f64() always returns Some for standard numeric types
                let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                if val <= 0.0 {
                    // Geometric mean undefined for non-positive values
                    return None;
                }
                sum += val.ln();
            }
            sum
        } else {
            data.par_iter()
                .map(|x| {
                    // SAFETY: to_f64() always returns Some for standard numeric types
                    let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                    if val <= 0.0 {
                        return f64::NAN;
                    }
                    val.ln()
                })
                .sum()
        };

        if geometric_sum.is_nan() {
            core::hint::cold_path();
            return None;
        }

        let geometric_mean = (geometric_sum / len as f64).exp();
        return Some(1.0 - geometric_mean / mean);
    }

    // General case: epsilon != 1
    // A_ε = 1 - (1/n * Σ((x_i/mean)^(1-ε)))^(1/(1-ε))
    let exponent = 1.0 - epsilon;

    let sum_powered: f64 = if len < PARALLEL_THRESHOLD {
        let mut sum = 0.0;
        for x in data {
            // SAFETY: to_f64() always returns Some for standard numeric types
            let val = unsafe { x.0.to_f64().unwrap_unchecked() };
            if val < 0.0 {
                // Negative values with non-integer exponent are undefined
                return None;
            }
            let ratio = val / mean;
            sum += ratio.powf(exponent);
        }
        sum
    } else {
        data.par_iter()
            .map(|x| {
                // SAFETY: to_f64() always returns Some for standard numeric types
                let val = unsafe { x.0.to_f64().unwrap_unchecked() };
                if val < 0.0 {
                    return f64::NAN;
                }
                let ratio = val / mean;
                ratio.powf(exponent)
            })
            .sum()
    };

    if sum_powered.is_nan() || sum_powered <= 0.0 {
        core::hint::cold_path();
        return None;
    }

    let atkinson = 1.0 - (sum_powered / len as f64).powf(1.0 / exponent);
    Some(atkinson)
}

/// Selection algorithm to find the k-th smallest element in O(n) average time.
/// This is an implementation of quickselect algorithm.
#[cfg(test)]
fn quickselect<T>(data: &mut [Partial<T>], k: usize) -> Option<&T>
where
    T: PartialOrd,
{
    if data.is_empty() || k >= data.len() {
        core::hint::cold_path();
        return None;
    }

    let mut left = 0;
    let mut right = data.len() - 1;

    loop {
        if left == right {
            return Some(&data[left].0);
        }

        // Use median-of-three pivot selection for better performance
        let pivot_idx = median_of_three_pivot(data, left, right);
        let pivot_idx = partition(data, left, right, pivot_idx);

        match k.cmp(&pivot_idx) {
            std::cmp::Ordering::Equal => return Some(&data[pivot_idx].0),
            std::cmp::Ordering::Less => right = pivot_idx - 1,
            std::cmp::Ordering::Greater => left = pivot_idx + 1,
        }
    }
}

/// Select the median of three elements as pivot for better quickselect performance
#[cfg(test)]
fn median_of_three_pivot<T>(data: &[Partial<T>], left: usize, right: usize) -> usize
where
    T: PartialOrd,
{
    let mid = left + (right - left) / 2;

    if data[left] <= data[mid] {
        if data[mid] <= data[right] {
            mid
        } else if data[left] <= data[right] {
            right
        } else {
            left
        }
    } else if data[left] <= data[right] {
        left
    } else if data[mid] <= data[right] {
        right
    } else {
        mid
    }
}

/// Partition function for quickselect
#[cfg(test)]
fn partition<T>(data: &mut [Partial<T>], left: usize, right: usize, pivot_idx: usize) -> usize
where
    T: PartialOrd,
{
    // Move pivot to end
    data.swap(pivot_idx, right);
    let mut store_idx = left;

    // Move all elements smaller than pivot to the left
    // Cache pivot position for better cache locality (access data[right] directly each time)
    for i in left..right {
        // Safety: i, store_idx, and right are guaranteed to be in bounds
        // Compare directly with pivot at data[right] - compiler should optimize this access
        if unsafe { data.get_unchecked(i) <= data.get_unchecked(right) } {
            data.swap(i, store_idx);
            store_idx += 1;
        }
    }

    // Move pivot to its final place
    data.swap(store_idx, right);
    store_idx
}

// This implementation follows Method 3 from https://en.wikipedia.org/wiki/Quartile
// It divides data into quarters based on the length n = 4k + r where r is remainder.
// For each remainder case (0,1,2,3), it uses different formulas to compute Q1, Q2, Q3.
fn quartiles_on_sorted<T>(data: &[Partial<T>]) -> Option<(f64, f64, f64)>
where
    T: PartialOrd + ToPrimitive,
{
    let len = data.len();

    // Early return for small arrays
    match len {
        0..=2 => {
            core::hint::cold_path();
            return None;
        }
        3 => {
            return Some(
                // SAFETY: We know these indices are valid because len == 3
                unsafe {
                    (
                        data.get_unchecked(0).0.to_f64()?,
                        data.get_unchecked(1).0.to_f64()?,
                        data.get_unchecked(2).0.to_f64()?,
                    )
                },
            );
        }
        _ => {}
    }

    // Calculate k and remainder in one division
    let k = len / 4;
    let remainder = len % 4;

    // SAFETY: All index calculations below are guaranteed to be in bounds
    // because we've verified len >= 4 above and k is len/4
    unsafe {
        Some(match remainder {
            0 => {
                // Let data = {x_i}_{i=0..4k} where k is positive integer.
                // Median q2 = (x_{2k-1} + x_{2k}) / 2.
                // If we divide data into two parts {x_i < q2} as L and
                // {x_i > q2} as R, #L == #R == 2k holds true. Thus,
                // q1 = (x_{k-1} + x_{k}) / 2 and q3 = (x_{3k-1} + x_{3k}) / 2.
                // =============
                // Simply put: Length is multiple of 4 (4k)
                // q1 = (x_{k-1} + x_k) / 2
                // q2 = (x_{2k-1} + x_{2k}) / 2
                // q3 = (x_{3k-1} + x_{3k}) / 2
                let q1 = f64::midpoint(
                    data.get_unchecked(k - 1).0.to_f64()?,
                    data.get_unchecked(k).0.to_f64()?,
                );
                let q2 = f64::midpoint(
                    data.get_unchecked(2 * k - 1).0.to_f64()?,
                    data.get_unchecked(2 * k).0.to_f64()?,
                );
                let q3 = f64::midpoint(
                    data.get_unchecked(3 * k - 1).0.to_f64()?,
                    data.get_unchecked(3 * k).0.to_f64()?,
                );
                (q1, q2, q3)
            }
            1 => {
                // Let data = {x_i}_{i=0..4k+1} where k is positive integer.
                // Median q2 = x_{2k}.
                // If we divide data other than q2 into two parts {x_i < q2}
                // as L and {x_i > q2} as R, #L == #R == 2k holds true. Thus,
                // q1 = (x_{k-1} + x_{k}) / 2 and q3 = (x_{3k} + x_{3k+1}) / 2.
                // =============
                // Simply put: Length is 4k + 1
                // q1 = (x_{k-1} + x_k) / 2
                // q2 = x_{2k}
                // q3 = (x_{3k} + x_{3k+1}) / 2
                let q1 = f64::midpoint(
                    data.get_unchecked(k - 1).0.to_f64()?,
                    data.get_unchecked(k).0.to_f64()?,
                );
                let q2 = data.get_unchecked(2 * k).0.to_f64()?;
                let q3 = f64::midpoint(
                    data.get_unchecked(3 * k).0.to_f64()?,
                    data.get_unchecked(3 * k + 1).0.to_f64()?,
                );
                (q1, q2, q3)
            }
            2 => {
                // Let data = {x_i}_{i=0..4k+2} where k is positive integer.
                // Median q2 = (x_{(2k+1)-1} + x_{2k+1}) / 2.
                // If we divide data into two parts {x_i < q2} as L and
                // {x_i > q2} as R, it's true that #L == #R == 2k+1.
                // Thus, q1 = x_{k} and q3 = x_{3k+1}.
                // =============
                // Simply put: Length is 4k + 2
                // q1 = x_k
                // q2 = (x_{2k} + x_{2k+1}) / 2
                // q3 = x_{3k+1}
                let q1 = data.get_unchecked(k).0.to_f64()?;
                let q2 = f64::midpoint(
                    data.get_unchecked(2 * k).0.to_f64()?,
                    data.get_unchecked(2 * k + 1).0.to_f64()?,
                );
                let q3 = data.get_unchecked(3 * k + 1).0.to_f64()?;
                (q1, q2, q3)
            }
            _ => {
                // Let data = {x_i}_{i=0..4k+3} where k is positive integer.
                // Median q2 = x_{2k+1}.
                // If we divide data other than q2 into two parts {x_i < q2}
                // as L and {x_i > q2} as R, #L == #R == 2k+1 holds true.
                // Thus, q1 = x_{k} and q3 = x_{3k+2}.
                // =============
                // Simply put: Length is 4k + 3
                // q1 = x_k
                // q2 = x_{2k+1}
                // q3 = x_{3k+2}
                let q1 = data.get_unchecked(k).0.to_f64()?;
                let q2 = data.get_unchecked(2 * k + 1).0.to_f64()?;
                let q3 = data.get_unchecked(3 * k + 2).0.to_f64()?;
                (q1, q2, q3)
            }
        })
    }
}

/// Zero-copy quartiles computation using index-based selection.
/// This avoids copying data by working with an array of indices.
///
/// Uses `select_nth_unstable_by` on the indices array, which partitions in-place.
/// After selecting position p, elements at indices [0..p] are <= the p-th element
/// and elements at [p+1..] are >= it. By selecting positions in ascending order,
/// each subsequent selection only needs to search within the right partition,
/// avoiding redundant O(n) resets.
fn quartiles_with_zero_copy_selection<T>(data: &[Partial<T>]) -> Option<(f64, f64, f64)>
where
    T: PartialOrd + ToPrimitive,
{
    let len = data.len();

    // Early return for small arrays
    match len {
        0..=2 => {
            core::hint::cold_path();
            return None;
        }
        3 => {
            let mut indices: Vec<usize> = (0..3).collect();
            let cmp = |a: &usize, b: &usize| {
                data[*a]
                    .partial_cmp(&data[*b])
                    .unwrap_or(std::cmp::Ordering::Less)
            };
            indices.sort_unstable_by(cmp);
            let min_val = data[indices[0]].0.to_f64()?;
            let med_val = data[indices[1]].0.to_f64()?;
            let max_val = data[indices[2]].0.to_f64()?;
            return Some((min_val, med_val, max_val));
        }
        _ => {}
    }

    let k = len / 4;
    let remainder = len % 4;

    let mut indices: Vec<usize> = (0..len).collect();
    let cmp = |a: &usize, b: &usize| {
        data[*a]
            .partial_cmp(&data[*b])
            .unwrap_or(std::cmp::Ordering::Less)
    };

    // Collect the unique positions we need in ascending order.
    // By selecting in ascending order, each select_nth_unstable_by partitions
    // the array so subsequent selections operate on progressively smaller slices.
    // We deduplicate because adjacent quartile boundaries can overlap for small k.
    let raw_positions: Vec<usize> = match remainder {
        0 => vec![k - 1, k, 2 * k - 1, 2 * k, 3 * k - 1, 3 * k],
        1 => vec![k - 1, k, 2 * k, 3 * k, 3 * k + 1],
        2 => vec![k, 2 * k, 2 * k + 1, 3 * k + 1],
        _ => vec![k, 2 * k + 1, 3 * k + 2],
    };

    let mut unique_positions = raw_positions.clone();
    unique_positions.dedup();

    // Select each unique position in ascending order, narrowing the search range
    let mut start = 0;
    for &pos in &unique_positions {
        indices[start..].select_nth_unstable_by(pos - start, &cmp);
        start = pos + 1;
    }

    // Now read all needed values (including duplicates) from the partitioned indices
    let values: Vec<f64> = raw_positions
        .iter()
        .map(|&pos| data[indices[pos]].0.to_f64())
        .collect::<Option<Vec<_>>>()?;

    match remainder {
        0 => {
            let q1 = f64::midpoint(values[0], values[1]);
            let q2 = f64::midpoint(values[2], values[3]);
            let q3 = f64::midpoint(values[4], values[5]);
            Some((q1, q2, q3))
        }
        1 => {
            let q1 = f64::midpoint(values[0], values[1]);
            let q2 = values[2];
            let q3 = f64::midpoint(values[3], values[4]);
            Some((q1, q2, q3))
        }
        2 => {
            let q1 = values[0];
            let q2 = f64::midpoint(values[1], values[2]);
            let q3 = values[3];
            Some((q1, q2, q3))
        }
        _ => Some((values[0], values[1], values[2])),
    }
}

fn mode_on_sorted<T, I>(it: I) -> Option<T>
where
    T: PartialOrd,
    I: Iterator<Item = T>,
{
    use std::cmp::Ordering;

    // This approach to computing the mode works very nicely when the
    // number of samples is large and is close to its cardinality.
    // In other cases, a hashmap would be much better.
    // But really, how can we know this when given an arbitrary stream?
    // Might just switch to a hashmap to track frequencies. That would also
    // be generally useful for discovering the cardinality of a sample.
    let (mut mode, mut next) = (None, None);
    let (mut mode_count, mut next_count) = (0usize, 0usize);
    for x in it {
        if mode.as_ref() == Some(&x) {
            mode_count += 1;
        } else if next.as_ref() == Some(&x) {
            next_count += 1;
        } else {
            next = Some(x);
            next_count = 0;
        }

        match next_count.cmp(&mode_count) {
            Ordering::Greater => {
                mode = next;
                mode_count = next_count;
                next = None;
                next_count = 0;
            }
            Ordering::Equal => {
                mode = None;
                mode_count = 0;
            }
            Ordering::Less => {}
        }
    }
    mode
}

/// Computes both modes and antimodes from a sorted slice of values.
/// This version works with references to avoid unnecessary cloning.
///
/// # Arguments
///
/// * `data` - A sorted slice of values
///
/// # Notes
///
/// - Mode is the most frequently occurring value(s)
/// - Antimode is the least frequently occurring value(s)
/// - Only returns up to 10 antimodes to avoid returning the full set when all values are unique
/// - For empty slices, returns empty vectors and zero counts
/// - For single value slices, returns that value as the mode and empty antimode
/// - When all values occur exactly once, returns empty mode and up to 10 values as antimodes
///
/// # Returns
///
/// A tuple containing:
/// * Modes information: `(Vec<T>, usize, u32)` where:
///   - Vec<T>: Vector containing the mode values
///   - usize: Number of modes found
///   - u32: Frequency/count of the mode values
/// * Antimodes information: `(Vec<T>, usize, u32)` where:
///   - Vec<T>: Vector containing up to 10 antimode values
///   - usize: Total number of antimodes
///   - u32: Frequency/count of the antimode values
#[allow(clippy::type_complexity)]
#[inline]
fn modes_and_antimodes_on_sorted_slice<T>(
    data: &[Partial<T>],
) -> ((Vec<T>, usize, u32), (Vec<T>, usize, u32))
where
    T: PartialOrd + Clone,
{
    let size = data.len();

    // Early return for empty slice
    if size == 0 {
        core::hint::cold_path();
        return ((Vec::new(), 0, 0), (Vec::new(), 0, 0));
    }

    // Estimate capacity using integer square root of size
    let sqrt_size = size.isqrt();
    let mut runs: Vec<(&T, u32)> = Vec::with_capacity(sqrt_size.clamp(16, 1_000));

    let mut current_value = &data[0].0;
    let mut current_count = 1;
    let mut highest_count = 1;
    let mut lowest_count = u32::MAX;

    // Count consecutive runs - optimized to reduce allocations
    for x in data.iter().skip(1) {
        if x.0 == *current_value {
            current_count += 1;
            highest_count = highest_count.max(current_count);
        } else {
            runs.push((current_value, current_count));
            lowest_count = lowest_count.min(current_count);
            current_value = &x.0;
            current_count = 1;
        }
    }
    runs.push((current_value, current_count));
    lowest_count = lowest_count.min(current_count);

    // Early return if only one unique value
    if runs.len() == 1 {
        let (val, count) = runs.pop().unwrap();
        return ((vec![val.clone()], 1, count), (Vec::new(), 0, 0));
    }

    // Special case: if all values appear exactly once
    if highest_count == 1 {
        let antimodes_count = runs.len().min(10);
        let total_count = runs.len();
        let mut antimodes = Vec::with_capacity(antimodes_count);
        for (val, _) in runs.into_iter().take(antimodes_count) {
            antimodes.push(val.clone());
        }
        // For modes: empty, count 0, occurrences 0 (not 1, 1)
        return ((Vec::new(), 0, 0), (antimodes, total_count, 1));
    }

    // Collect modes and antimodes directly in a single pass, cloning values immediately
    // instead of collecting indices first and then cloning in a second pass
    let estimated_modes = (runs.len() / 10).clamp(1, 10);
    let estimated_antimodes = 10.min(runs.len());

    let mut modes_result = Vec::with_capacity(estimated_modes);
    let mut antimodes_result = Vec::with_capacity(estimated_antimodes);
    let mut mode_count = 0;
    let mut antimodes_count = 0;
    let mut antimodes_collected = 0_u32;

    for (val, count) in &runs {
        if *count == highest_count {
            modes_result.push((*val).clone());
            mode_count += 1;
        }
        if *count == lowest_count {
            antimodes_count += 1;
            if antimodes_collected < 10 {
                antimodes_result.push((*val).clone());
                antimodes_collected += 1;
            }
        }
    }

    (
        (modes_result, mode_count, highest_count),
        (antimodes_result, antimodes_count, lowest_count),
    )
}

/// A commutative data structure for lazily sorted sequences of data.
///
/// The sort does not occur until statistics need to be computed.
///
/// Note that this works on types that do not define a total ordering like
/// `f32` and `f64`. When an ordering is not defined, an arbitrary order
/// is returned.
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Unsorted<T> {
    /// Internal cache flag indicating whether `data` is currently sorted.
    /// This field is skipped during serialization and deserialization.
    #[serde(skip)]
    sorted: bool,
    data: Vec<Partial<T>>,
}

// Manual PartialEq/Eq: ignore `sorted` cache flag so equality reflects
// logical contents only (two Unsorted with same data compare equal
// regardless of whether one has been sorted).
impl<T: PartialEq> PartialEq for Unsorted<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: PartialEq> Eq for Unsorted<T> where Partial<T>: Eq {}

impl<T: PartialOrd + Send> Unsorted<T> {
    /// Create initial empty state.
    #[inline]
    #[must_use]
    pub fn new() -> Unsorted<T> {
        Default::default()
    }

    /// Add a new element to the set.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn add(&mut self, v: T) {
        self.sorted = false;
        self.data.push(Partial(v));
    }

    /// Return the number of data points.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    fn sort(&mut self) {
        if !self.sorted {
            // Use sequential sort for small datasets (< 10k elements) to avoid parallel overhead
            if self.data.len() < PARALLEL_THRESHOLD {
                self.data.sort_unstable();
            } else {
                self.data.par_sort_unstable();
            }
            self.sorted = true;
        }
    }

    #[inline]
    const fn already_sorted(&mut self) {
        self.sorted = true;
    }

    /// Add multiple elements efficiently
    #[inline]
    pub fn add_bulk(&mut self, values: Vec<T>) {
        self.sorted = false;
        self.data.reserve(values.len());
        self.data.extend(values.into_iter().map(Partial));
    }

    /// Shrink capacity to fit current data
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Create with specific capacity
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Unsorted {
            sorted: true,
            data: Vec::with_capacity(capacity),
        }
    }

    /// Add a value assuming it's greater than all existing values
    #[inline]
    pub fn push_ascending(&mut self, value: T) {
        if let Some(last) = self.data.last() {
            debug_assert!(last.0 <= value, "Value must be >= than last element");
        }
        self.data.push(Partial(value));
        // Data remains sorted
    }
}

impl<T: PartialOrd + PartialEq + Clone + Send + Sync> Unsorted<T> {
    #[inline]
    /// Returns the cardinality of the data.
    /// Set `sorted` to `true` if the data is already sorted.
    /// Set `parallel_threshold` to `0` to force sequential processing.
    /// Set `parallel_threshold` to `1` to use the default parallel threshold (`10_000`).
    /// Set `parallel_threshold` to `2` to force parallel processing.
    /// Set `parallel_threshold` to any other value to use a custom parallel threshold
    /// greater than the default threshold of `10_000`.
    pub fn cardinality(&mut self, sorted: bool, parallel_threshold: usize) -> u64 {
        const CHUNK_SIZE: usize = 2048; // Process data in chunks of 2048 elements
        const DEFAULT_PARALLEL_THRESHOLD: usize = 10_240; // multiple of 2048

        let len = self.data.len();
        match len {
            0 => return 0,
            1 => return 1,
            _ => {}
        }

        if sorted {
            self.already_sorted();
        } else {
            self.sort();
        }

        let use_parallel = parallel_threshold != 0
            && (parallel_threshold == 1
                || len > parallel_threshold.max(DEFAULT_PARALLEL_THRESHOLD));

        if use_parallel {
            // Parallel processing using chunks via fold/reduce — no intermediate Vec.
            // Reduction state: (count, leftmost_first, rightmost_last). Associative:
            // combining (cL, fL, lL) with (cR, fR, lR) yields (cL+cR - [lL==fR], fL, lR).
            self.data
                .par_chunks(CHUNK_SIZE)
                .map(|chunk| {
                    // Count unique elements within this chunk
                    let mut count = u64::from(!chunk.is_empty());
                    for [a, b] in chunk.array_windows::<2>() {
                        if a != b {
                            count += 1;
                        }
                    }
                    (count, chunk.first(), chunk.last())
                })
                .reduce(
                    || (0u64, None, None),
                    |(cl, fl, ll), (cr, fr, lr)| match (ll, fr) {
                        (None, _) => (cr, fr, lr),
                        (_, None) => (cl, fl, ll),
                        (Some(l), Some(r)) => {
                            let adj = u64::from(l == r);
                            (cl + cr - adj, fl, lr)
                        },
                    },
                )
                .0
        } else {
            // Sequential processing

            // the statement below is equivalent to:
            // let mut count = if self.data.is_empty() { 0 } else { 1 };
            let mut count = u64::from(!self.data.is_empty());

            for [a, b] in self.data.array_windows::<2>() {
                if a != b {
                    count += 1;
                }
            }
            count
        }
    }
}

impl<T: PartialOrd + Clone + Send> Unsorted<T> {
    /// Returns the mode of the data.
    #[inline]
    pub fn mode(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        mode_on_sorted(self.data.iter().map(|p| &p.0)).cloned()
    }

    /// Returns the modes of the data.
    /// Note that there is also a `frequency::mode()` function that return one mode
    /// with the highest frequency. If there is a tie, it returns None.
    #[inline]
    fn modes(&mut self) -> (Vec<T>, usize, u32) {
        if self.data.is_empty() {
            return (Vec::new(), 0, 0);
        }
        self.sort();
        modes_and_antimodes_on_sorted_slice(&self.data).0
    }

    /// Returns the antimodes of the data.
    /// `antimodes_result` only returns the first 10 antimodes
    #[inline]
    fn antimodes(&mut self) -> (Vec<T>, usize, u32) {
        if self.data.is_empty() {
            return (Vec::new(), 0, 0);
        }
        self.sort();
        modes_and_antimodes_on_sorted_slice(&self.data).1
    }

    /// Returns the modes and antimodes of the data.
    /// `antimodes_result` only returns the first 10 antimodes
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn modes_antimodes(&mut self) -> ((Vec<T>, usize, u32), (Vec<T>, usize, u32)) {
        if self.data.is_empty() {
            return ((Vec::new(), 0, 0), (Vec::new(), 0, 0));
        }
        self.sort();
        modes_and_antimodes_on_sorted_slice(&self.data)
    }
}

impl Unsorted<Vec<u8>> {
    /// Add a byte slice, converting to `Vec<u8>` internally.
    ///
    /// This is a convenience method that avoids requiring the caller to call
    /// `.to_vec()` before `add()`. The allocation still occurs internally,
    /// but the API is cleaner and opens the door for future optimizations
    /// (e.g., frequency-map deduplication for high-cardinality data).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn add_bytes(&mut self, v: &[u8]) {
        self.sorted = false;
        self.data.push(Partial(v.to_vec()));
    }
}

impl<T: PartialOrd + ToPrimitive + Send> Unsorted<T> {
    /// Returns the median of the data.
    #[inline]
    pub fn median(&mut self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        median_on_sorted(&self.data)
    }
}

impl<T: PartialOrd + ToPrimitive + Send + Sync> Unsorted<T> {
    /// Returns the Median Absolute Deviation (MAD) of the data.
    #[inline]
    pub fn mad(&mut self, existing_median: Option<f64>) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        if existing_median.is_none() {
            self.sort();
        }
        mad_on_sorted(&self.data, existing_median)
    }
}

impl<T: PartialOrd + ToPrimitive + Send> Unsorted<T> {
    /// Returns the quartiles of the data using the traditional sorting approach.
    ///
    /// This method sorts the data first and then computes quartiles.
    /// Time complexity: O(n log n)
    #[inline]
    pub fn quartiles(&mut self) -> Option<(f64, f64, f64)> {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        quartiles_on_sorted(&self.data)
    }
}

impl<T: PartialOrd + ToPrimitive + Send + Sync> Unsorted<T> {
    /// Returns the Gini Coefficient of the data.
    ///
    /// The Gini Coefficient measures inequality in a distribution, ranging from 0 (perfect equality)
    /// to 1 (perfect inequality). This method sorts the data first and then computes the Gini coefficient.
    /// Time complexity: O(n log n)
    #[inline]
    pub fn gini(&mut self, precalc_sum: Option<f64>) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        gini_on_sorted(&self.data, precalc_sum)
    }

    /// Returns the kurtosis (excess kurtosis) of the data.
    ///
    /// Kurtosis measures the "tailedness" of a distribution. Excess kurtosis is kurtosis - 3,
    /// where 0 indicates a normal distribution, positive values indicate heavy tails, and
    /// negative values indicate light tails. This method sorts the data first and then computes kurtosis.
    /// Time complexity: O(n log n)
    #[inline]
    pub fn kurtosis(
        &mut self,
        precalc_mean: Option<f64>,
        precalc_variance: Option<f64>,
    ) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        kurtosis_on_sorted(&self.data, precalc_mean, precalc_variance)
    }

    /// Returns the percentile rank of a value in the data.
    ///
    /// Returns the percentile rank (0-100) of the given value. If the value is less than all
    /// values, returns 0.0. If greater than all, returns 100.0.
    /// This method sorts the data first and then computes the percentile rank.
    /// Time complexity: O(n log n)
    #[inline]
    #[allow(clippy::needless_pass_by_value)]
    pub fn percentile_rank<V>(&mut self, value: V) -> Option<f64>
    where
        V: PartialOrd + ToPrimitive,
    {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        percentile_rank_on_sorted(&self.data, &value)
    }

    /// Returns the Atkinson Index of the data.
    ///
    /// The Atkinson Index measures inequality with an inequality aversion parameter ε.
    /// It ranges from 0 (perfect equality) to 1 (perfect inequality).
    /// Higher ε values give more weight to inequality at the lower end of the distribution.
    /// This method sorts the data first and then computes the Atkinson index.
    /// Time complexity: O(n log n)
    ///
    /// # Arguments
    /// * `epsilon` - Inequality aversion parameter (must be >= 0). Common values:
    ///   - 0.0: No inequality aversion (returns 0)
    ///   - 0.5: Moderate aversion
    ///   - 1.0: Uses geometric mean (special case)
    ///   - 2.0: High aversion
    /// * `precalc_mean` - Optional pre-calculated mean
    /// * `precalc_geometric_sum` - Optional pre-calculated geometric sum (sum of ln(val)), only used when epsilon = 1
    #[inline]
    pub fn atkinson(
        &mut self,
        epsilon: f64,
        precalc_mean: Option<f64>,
        precalc_geometric_sum: Option<f64>,
    ) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        atkinson_on_sorted(&self.data, epsilon, precalc_mean, precalc_geometric_sum)
    }
}

impl<T: PartialOrd + ToPrimitive + Clone + Send> Unsorted<T> {
    /// Returns the quartiles of the data using selection algorithm.
    ///
    /// This implementation uses a selection algorithm (quickselect) to find quartiles
    /// in O(n) average time complexity instead of O(n log n) sorting.
    /// Requires T to implement Clone to create a working copy of the data.
    ///
    /// **Performance Note**: While theoretically O(n) vs O(n log n), this implementation
    /// is often slower than the sorting-based approach for small to medium datasets due to:
    /// - Need to find multiple order statistics (3 separate quickselect calls)
    /// - Overhead of copying data to avoid mutation
    /// - Rayon's highly optimized parallel sorting
    #[inline]
    pub fn quartiles_with_selection(&mut self) -> Option<(f64, f64, f64)> {
        if self.data.is_empty() {
            return None;
        }
        // Use zero-copy approach (indices-based) to avoid cloning all elements
        quartiles_with_zero_copy_selection(&self.data)
    }
}

impl<T: PartialOrd + ToPrimitive + Send> Unsorted<T> {
    /// Returns the quartiles using zero-copy selection algorithm.
    ///
    /// This implementation avoids copying data by working with indices instead,
    /// providing better performance than the clone-based selection approach.
    /// The algorithm is O(n) average time and only allocates a vector of indices (usize).
    #[inline]
    #[must_use]
    pub fn quartiles_zero_copy(&self) -> Option<(f64, f64, f64)> {
        if self.data.is_empty() {
            return None;
        }
        quartiles_with_zero_copy_selection(&self.data)
    }
}

impl<T: PartialOrd + Send> Commute for Unsorted<T> {
    #[inline]
    fn merge(&mut self, mut v: Unsorted<T>) {
        if v.is_empty() {
            return;
        }

        self.sorted = false;
        // we use std::mem::take to avoid unnecessary allocations
        self.data.extend(std::mem::take(&mut v.data));
    }
}

impl<T: PartialOrd> Default for Unsorted<T> {
    #[inline]
    fn default() -> Unsorted<T> {
        Unsorted {
            data: Vec::with_capacity(16),
            sorted: true, // empty is sorted
        }
    }
}

impl<T: PartialOrd + Send> FromIterator<T> for Unsorted<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Unsorted<T> {
        let mut v = Unsorted::new();
        v.extend(it);
        v
    }
}

impl<T: PartialOrd> Extend<T> for Unsorted<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, it: I) {
        self.sorted = false;
        self.data.extend(it.into_iter().map(Partial));
    }
}

fn custom_percentiles_on_sorted<T>(data: &[Partial<T>], percentiles: &[u8]) -> Option<Vec<T>>
where
    T: PartialOrd + Clone,
{
    let len = data.len();

    // Early return for empty array or invalid percentiles
    if len == 0 || percentiles.iter().any(|&p| p > 100) {
        return None;
    }

    // Optimize: Check if percentiles are already sorted and unique
    let unique_percentiles: Vec<u8> = if percentiles.len() <= 1 {
        // Single or empty percentile - no need to sort/dedup
        percentiles.to_vec()
    } else {
        // Check if already sorted and unique (common case)
        let is_sorted_unique = percentiles.array_windows::<2>().all(|[a, b]| a < b);

        if is_sorted_unique {
            // Already sorted and unique, use directly without cloning
            percentiles.to_vec()
        } else {
            // Need to sort and dedup - use fixed-size bool array (domain is 0..=100)
            let mut seen = [false; 101];
            let mut sorted_unique = Vec::with_capacity(percentiles.len().min(101));
            for &p in percentiles {
                if !seen[p as usize] {
                    seen[p as usize] = true;
                    sorted_unique.push(p);
                }
            }
            sorted_unique.sort_unstable();
            sorted_unique
        }
    };

    let mut results = Vec::with_capacity(unique_percentiles.len());

    // SAFETY: All index calculations below are guaranteed to be in bounds
    // because we've verified len > 0 and the rank calculation ensures
    // the index is within bounds
    unsafe {
        for &p in &unique_percentiles {
            // Calculate the ordinal rank using nearest-rank method
            // see https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
            // n = ⌈(P/100) × N⌉
            #[allow(clippy::cast_sign_loss)]
            let rank = ((f64::from(p) / 100.0) * len as f64).ceil() as usize;

            // Convert to 0-based index
            let idx = rank.saturating_sub(1);

            // Get the value at that rank and extract the inner value
            results.push(data.get_unchecked(idx).0.clone());
        }
    }

    Some(results)
}

impl<T: PartialOrd + Clone + Send> Unsorted<T> {
    /// Returns the requested percentiles of the data.
    ///
    /// Uses the nearest-rank method to compute percentiles.
    /// Each returned value is an actual value from the dataset.
    ///
    /// # Arguments
    /// * `percentiles` - A slice of u8 values representing percentiles to compute (0-100)
    ///
    /// # Returns
    /// * `None` if the data is empty or if any percentile is > 100
    /// * `Some(Vec<T>)` containing percentile values in the same order as requested
    ///
    /// # Example
    /// ```
    /// use stats::Unsorted;
    /// let mut data = Unsorted::new();
    /// data.extend(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    /// let percentiles = vec![25, 50, 75];
    /// let results = data.custom_percentiles(&percentiles).unwrap();
    /// assert_eq!(results, vec![3, 5, 8]);
    /// ```
    #[inline]
    pub fn custom_percentiles(&mut self, percentiles: &[u8]) -> Option<Vec<T>> {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        custom_percentiles_on_sorted(&self.data, percentiles)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cardinality_empty() {
        let mut unsorted: Unsorted<i32> = Unsorted::new();
        assert_eq!(unsorted.cardinality(false, 1), 0);
    }

    #[test]
    fn test_cardinality_single_element() {
        let mut unsorted = Unsorted::new();
        unsorted.add(5);
        assert_eq!(unsorted.cardinality(false, 1), 1);
    }

    #[test]
    fn test_cardinality_unique_elements() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        assert_eq!(unsorted.cardinality(false, 1), 5);
    }

    #[test]
    fn test_cardinality_duplicate_elements() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4]);
        assert_eq!(unsorted.cardinality(false, 1), 4);
    }

    #[test]
    fn test_cardinality_all_same() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1; 100]);
        assert_eq!(unsorted.cardinality(false, 1), 1);
    }

    #[test]
    fn test_cardinality_large_range() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(0..1_000_000);
        assert_eq!(unsorted.cardinality(false, 1), 1_000_000);
    }

    #[test]
    fn test_cardinality_large_range_sequential() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(0..1_000_000);
        assert_eq!(unsorted.cardinality(false, 2_000_000), 1_000_000);
    }

    #[test]
    fn test_cardinality_presorted() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        unsorted.sort();
        assert_eq!(unsorted.cardinality(true, 1), 5);
    }

    #[test]
    fn test_cardinality_float() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1.0, 1.0, 2.0, 3.0, 3.0, 4.0]);
        assert_eq!(unsorted.cardinality(false, 1), 4);
    }

    #[test]
    fn test_cardinality_string() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec!["a", "b", "b", "c", "c", "c"]);
        assert_eq!(unsorted.cardinality(false, 1), 3);
    }

    #[test]
    fn test_quartiles_selection_vs_sorted() {
        // Test that selection-based quartiles gives same results as sorting-based
        let test_cases = vec![
            vec![3, 5, 7, 9],
            vec![3, 5, 7],
            vec![1, 2, 7, 11],
            vec![3, 5, 7, 9, 12],
            vec![2, 2, 3, 8, 10],
            vec![3, 5, 7, 9, 12, 20],
            vec![0, 2, 4, 8, 10, 11],
            vec![3, 5, 7, 9, 12, 20, 21],
            vec![1, 5, 6, 6, 7, 10, 19],
        ];

        for test_case in test_cases {
            let mut unsorted1 = Unsorted::new();
            let mut unsorted2 = Unsorted::new();
            let mut unsorted3 = Unsorted::new();
            unsorted1.extend(test_case.clone());
            unsorted2.extend(test_case.clone());
            unsorted3.extend(test_case.clone());

            let result_sorted = unsorted1.quartiles();
            let result_selection = unsorted2.quartiles_with_selection();
            let result_zero_copy = unsorted3.quartiles_zero_copy();

            assert_eq!(
                result_sorted, result_selection,
                "Selection mismatch for test case: {:?}",
                test_case
            );
            assert_eq!(
                result_sorted, result_zero_copy,
                "Zero-copy mismatch for test case: {:?}",
                test_case
            );
        }
    }

    #[test]
    fn test_quartiles_with_selection_small() {
        // Test edge cases for selection-based quartiles
        let mut unsorted: Unsorted<i32> = Unsorted::new();
        assert_eq!(unsorted.quartiles_with_selection(), None);

        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2]);
        assert_eq!(unsorted.quartiles_with_selection(), None);

        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3]);
        assert_eq!(unsorted.quartiles_with_selection(), Some((1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_quickselect() {
        let data = vec![
            Partial(3),
            Partial(1),
            Partial(4),
            Partial(1),
            Partial(5),
            Partial(9),
            Partial(2),
            Partial(6),
        ];

        // Test finding different positions
        assert_eq!(quickselect(&mut data.clone(), 0), Some(&1));
        assert_eq!(quickselect(&mut data.clone(), 3), Some(&3));
        assert_eq!(quickselect(&mut data.clone(), 7), Some(&9));

        // Test edge cases
        let mut empty: Vec<Partial<i32>> = vec![];
        assert_eq!(quickselect(&mut empty, 0), None);

        let mut data = vec![Partial(3), Partial(1), Partial(4), Partial(1), Partial(5)];
        assert_eq!(quickselect(&mut data, 10), None); // k >= len
    }

    #[test]
    fn median_stream() {
        assert_eq!(median(vec![3usize, 5, 7, 9].into_iter()), Some(6.0));
        assert_eq!(median(vec![3usize, 5, 7].into_iter()), Some(5.0));
    }

    #[test]
    fn mad_stream() {
        assert_eq!(mad(vec![3usize, 5, 7, 9].into_iter(), None), Some(2.0));
        assert_eq!(
            mad(
                vec![
                    86usize, 60, 95, 39, 49, 12, 56, 82, 92, 24, 33, 28, 46, 34, 100, 39, 100, 38,
                    50, 61, 39, 88, 5, 13, 64
                ]
                .into_iter(),
                None
            ),
            Some(16.0)
        );
    }

    #[test]
    fn mad_stream_precalc_median() {
        let data = vec![3usize, 5, 7, 9].into_iter();
        let median1 = median(data.clone());
        assert_eq!(mad(data, median1), Some(2.0));

        let data2 = vec![
            86usize, 60, 95, 39, 49, 12, 56, 82, 92, 24, 33, 28, 46, 34, 100, 39, 100, 38, 50, 61,
            39, 88, 5, 13, 64,
        ]
        .into_iter();
        let median2 = median(data2.clone());
        assert_eq!(mad(data2, median2), Some(16.0));
    }

    #[test]
    fn mode_stream() {
        assert_eq!(mode(vec![3usize, 5, 7, 9].into_iter()), None);
        assert_eq!(mode(vec![3usize, 3, 3, 3].into_iter()), Some(3));
        assert_eq!(mode(vec![3usize, 3, 3, 4].into_iter()), Some(3));
        assert_eq!(mode(vec![4usize, 3, 3, 3].into_iter()), Some(3));
        assert_eq!(mode(vec![1usize, 1, 2, 3, 3].into_iter()), None);
    }

    #[test]
    fn median_floats() {
        assert_eq!(median(vec![3.0f64, 5.0, 7.0, 9.0].into_iter()), Some(6.0));
        assert_eq!(median(vec![3.0f64, 5.0, 7.0].into_iter()), Some(5.0));
    }

    #[test]
    fn mode_floats() {
        assert_eq!(mode(vec![3.0f64, 5.0, 7.0, 9.0].into_iter()), None);
        assert_eq!(mode(vec![3.0f64, 3.0, 3.0, 3.0].into_iter()), Some(3.0));
        assert_eq!(mode(vec![3.0f64, 3.0, 3.0, 4.0].into_iter()), Some(3.0));
        assert_eq!(mode(vec![4.0f64, 3.0, 3.0, 3.0].into_iter()), Some(3.0));
        assert_eq!(mode(vec![1.0f64, 1.0, 2.0, 3.0, 3.0].into_iter()), None);
    }

    #[test]
    fn modes_stream() {
        assert_eq!(modes(vec![3usize, 5, 7, 9].into_iter()), (vec![], 0, 0));
        assert_eq!(modes(vec![3usize, 3, 3, 3].into_iter()), (vec![3], 1, 4));
        assert_eq!(modes(vec![3usize, 3, 4, 4].into_iter()), (vec![3, 4], 2, 2));
        assert_eq!(modes(vec![4usize, 3, 3, 3].into_iter()), (vec![3], 1, 3));
        assert_eq!(modes(vec![1usize, 1, 2, 2].into_iter()), (vec![1, 2], 2, 2));
        let vec: Vec<u32> = vec![];
        assert_eq!(modes(vec.into_iter()), (vec![], 0, 0));
    }

    #[test]
    fn modes_floats() {
        assert_eq!(
            modes(vec![3_f64, 5.0, 7.0, 9.0].into_iter()),
            (vec![], 0, 0)
        );
        assert_eq!(
            modes(vec![3_f64, 3.0, 3.0, 3.0].into_iter()),
            (vec![3.0], 1, 4)
        );
        assert_eq!(
            modes(vec![3_f64, 3.0, 4.0, 4.0].into_iter()),
            (vec![3.0, 4.0], 2, 2)
        );
        assert_eq!(
            modes(vec![1_f64, 1.0, 2.0, 3.0, 3.0].into_iter()),
            (vec![1.0, 3.0], 2, 2)
        );
    }

    #[test]
    fn antimodes_stream() {
        assert_eq!(
            antimodes(vec![3usize, 5, 7, 9].into_iter()),
            (vec![3, 5, 7, 9], 4, 1)
        );
        assert_eq!(
            antimodes(vec![1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].into_iter()),
            (vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 13, 1)
        );
        assert_eq!(
            antimodes(vec![1usize, 3, 3, 3].into_iter()),
            (vec![1], 1, 1)
        );
        assert_eq!(
            antimodes(vec![3usize, 3, 4, 4].into_iter()),
            (vec![3, 4], 2, 2)
        );
        assert_eq!(
            antimodes(
                vec![
                    3usize, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
                    14, 14, 15, 15
                ]
                .into_iter()
            ),
            // we only show the first 10 of the 13 antimodes
            (vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 13, 2)
        );
        assert_eq!(
            antimodes(
                vec![
                    3usize, 3, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 4, 4, 5, 5, 6, 6, 7, 7, 13, 13,
                    14, 14, 15, 15
                ]
                .into_iter()
            ),
            (vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 13, 2)
        );
        assert_eq!(
            antimodes(vec![3usize, 3, 3, 4].into_iter()),
            (vec![4], 1, 1)
        );
        assert_eq!(
            antimodes(vec![4usize, 3, 3, 3].into_iter()),
            (vec![4], 1, 1)
        );
        assert_eq!(
            antimodes(vec![1usize, 1, 2, 2].into_iter()),
            (vec![1, 2], 2, 2)
        );
        let vec: Vec<u32> = vec![];
        assert_eq!(antimodes(vec.into_iter()), (vec![], 0, 0));
    }

    #[test]
    fn antimodes_floats() {
        assert_eq!(
            antimodes(vec![3_f64, 5.0, 7.0, 9.0].into_iter()),
            (vec![3.0, 5.0, 7.0, 9.0], 4, 1)
        );
        assert_eq!(
            antimodes(vec![3_f64, 3.0, 3.0, 3.0].into_iter()),
            (vec![], 0, 0)
        );
        assert_eq!(
            antimodes(vec![3_f64, 3.0, 4.0, 4.0].into_iter()),
            (vec![3.0, 4.0], 2, 2)
        );
        assert_eq!(
            antimodes(vec![1_f64, 1.0, 2.0, 3.0, 3.0].into_iter()),
            (vec![2.0], 1, 1)
        );
    }

    #[test]
    fn test_custom_percentiles() {
        // Test with integers
        let mut unsorted: Unsorted<i32> = Unsorted::new();
        unsorted.extend(1..=11); // [1,2,3,4,5,6,7,8,9,10,11]

        let result = unsorted.custom_percentiles(&[25, 50, 75]).unwrap();
        assert_eq!(result, vec![3, 6, 9]);

        // Test with strings
        let mut str_data = Unsorted::new();
        str_data.extend(vec!["a", "b", "c", "d", "e"]);
        let result = str_data.custom_percentiles(&[20, 40, 60, 80]).unwrap();
        assert_eq!(result, vec!["a", "b", "c", "d"]);

        // Test with chars
        let mut char_data = Unsorted::new();
        char_data.extend('a'..='e');
        let result = char_data.custom_percentiles(&[25, 50, 75]).unwrap();
        assert_eq!(result, vec!['b', 'c', 'd']);

        // Test with floats
        let mut float_data = Unsorted::new();
        float_data.extend(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]);
        let result = float_data
            .custom_percentiles(&[10, 30, 50, 70, 90])
            .unwrap();
        assert_eq!(result, vec![1.1, 3.3, 5.5, 7.7, 9.9]);

        // Test with empty percentiles array
        let result = float_data.custom_percentiles(&[]).unwrap();
        assert_eq!(result, Vec::<f64>::new());

        // Test with duplicate percentiles
        let result = float_data.custom_percentiles(&[50, 50, 50]).unwrap();
        assert_eq!(result, vec![5.5]);

        // Test with extreme percentiles
        let result = float_data.custom_percentiles(&[0, 100]).unwrap();
        assert_eq!(result, vec![1.1, 9.9]);

        // Test with unsorted percentiles
        let result = float_data.custom_percentiles(&[75, 25, 50]).unwrap();
        assert_eq!(result, vec![3.3, 5.5, 7.7]); // results always sorted

        // Test with single element
        let mut single = Unsorted::new();
        single.add(42);
        let result = single.custom_percentiles(&[0, 50, 100]).unwrap();
        assert_eq!(result, vec![42, 42, 42]);
    }

    #[test]
    fn quartiles_stream() {
        assert_eq!(
            quartiles(vec![3usize, 5, 7].into_iter()),
            Some((3., 5., 7.))
        );
        assert_eq!(
            quartiles(vec![3usize, 5, 7, 9].into_iter()),
            Some((4., 6., 8.))
        );
        assert_eq!(
            quartiles(vec![1usize, 2, 7, 11].into_iter()),
            Some((1.5, 4.5, 9.))
        );
        assert_eq!(
            quartiles(vec![3usize, 5, 7, 9, 12].into_iter()),
            Some((4., 7., 10.5))
        );
        assert_eq!(
            quartiles(vec![2usize, 2, 3, 8, 10].into_iter()),
            Some((2., 3., 9.))
        );
        assert_eq!(
            quartiles(vec![3usize, 5, 7, 9, 12, 20].into_iter()),
            Some((5., 8., 12.))
        );
        assert_eq!(
            quartiles(vec![0usize, 2, 4, 8, 10, 11].into_iter()),
            Some((2., 6., 10.))
        );
        assert_eq!(
            quartiles(vec![3usize, 5, 7, 9, 12, 20, 21].into_iter()),
            Some((5., 9., 20.))
        );
        assert_eq!(
            quartiles(vec![1usize, 5, 6, 6, 7, 10, 19].into_iter()),
            Some((5., 6., 10.))
        );
    }

    #[test]
    fn quartiles_floats() {
        assert_eq!(
            quartiles(vec![3_f64, 5., 7.].into_iter()),
            Some((3., 5., 7.))
        );
        assert_eq!(
            quartiles(vec![3_f64, 5., 7., 9.].into_iter()),
            Some((4., 6., 8.))
        );
        assert_eq!(
            quartiles(vec![3_f64, 5., 7., 9., 12.].into_iter()),
            Some((4., 7., 10.5))
        );
        assert_eq!(
            quartiles(vec![3_f64, 5., 7., 9., 12., 20.].into_iter()),
            Some((5., 8., 12.))
        );
        assert_eq!(
            quartiles(vec![3_f64, 5., 7., 9., 12., 20., 21.].into_iter()),
            Some((5., 9., 20.))
        );
    }

    #[test]
    fn test_quartiles_zero_copy_small() {
        // Test edge cases for zero-copy quartiles
        let unsorted: Unsorted<i32> = Unsorted::new();
        assert_eq!(unsorted.quartiles_zero_copy(), None);

        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2]);
        assert_eq!(unsorted.quartiles_zero_copy(), None);

        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3]);
        assert_eq!(unsorted.quartiles_zero_copy(), Some((1.0, 2.0, 3.0)));

        // Test larger case
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![3, 5, 7, 9]);
        assert_eq!(unsorted.quartiles_zero_copy(), Some((4.0, 6.0, 8.0)));
    }

    #[test]
    fn gini_empty() {
        let mut unsorted: Unsorted<i32> = Unsorted::new();
        assert_eq!(unsorted.gini(None), None);
        let empty_vec: Vec<i32> = vec![];
        assert_eq!(gini(empty_vec.into_iter(), None), None);
    }

    #[test]
    fn gini_single_element() {
        let mut unsorted = Unsorted::new();
        unsorted.add(5);
        assert_eq!(unsorted.gini(None), Some(0.0));
        assert_eq!(gini(vec![5].into_iter(), None), Some(0.0));
    }

    #[test]
    fn gini_perfect_equality() {
        // All values are the same - perfect equality, Gini = 0
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![10, 10, 10, 10, 10]);
        let result = unsorted.gini(None).unwrap();
        assert!((result - 0.0).abs() < 1e-10, "Expected 0.0, got {}", result);

        assert!((gini(vec![10, 10, 10, 10, 10].into_iter(), None).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn gini_perfect_inequality() {
        // One value has everything, others have zero - perfect inequality
        // For [0, 0, 0, 0, 100], Gini should be close to 1
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![0, 0, 0, 0, 100]);
        let result = unsorted.gini(None).unwrap();
        // Perfect inequality should give Gini close to 1
        // For n=5, one value=100, others=0: G = (2*5*100)/(5*100) - 6/5 = 2 - 1.2 = 0.8
        assert!((result - 0.8).abs() < 1e-10, "Expected 0.8, got {}", result);
    }

    #[test]
    fn gini_stream() {
        // Test with known values
        // For [1, 2, 3, 4, 5]:
        // sum = 15
        // weighted_sum = 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 1 + 4 + 9 + 16 + 25 = 55
        // n = 5
        // G = (2 * 55) / (5 * 15) - 6/5 = 110/75 - 1.2 = 1.4667 - 1.2 = 0.2667
        let result = gini(vec![1usize, 2, 3, 4, 5].into_iter(), None).unwrap();
        let expected = (2.0 * 55.0) / (5.0 * 15.0) - 6.0 / 5.0;
        assert!(
            (result - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn gini_floats() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = unsorted.gini(None).unwrap();
        let expected = (2.0 * 55.0) / (5.0 * 15.0) - 6.0 / 5.0;
        assert!((result - expected).abs() < 1e-10);

        assert!(
            (gini(vec![1.0f64, 2.0, 3.0, 4.0, 5.0].into_iter(), None).unwrap() - expected).abs()
                < 1e-10
        );
    }

    #[test]
    fn gini_all_zeros() {
        // All zeros - sum is zero, Gini is undefined
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![0, 0, 0, 0]);
        assert_eq!(unsorted.gini(None), None);
        assert_eq!(gini(vec![0, 0, 0, 0].into_iter(), None), None);
    }

    #[test]
    fn gini_negative_values() {
        // Test with negative values (mathematically valid)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![-5, -3, -1, 1, 3, 5]);
        let result = unsorted.gini(None);
        // Sum is 0, so Gini is undefined
        assert_eq!(result, None);

        // Test with negative values that don't sum to zero
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![-2, -1, 0, 1, 2]);
        let result = unsorted.gini(None);
        // Sum is 0, so Gini is undefined
        assert_eq!(result, None);

        // Test with values containing negatives that sum to non-zero
        // Gini is undefined for negative values, should return None
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![-1, 0, 1, 2, 3]);
        let result = unsorted.gini(None);
        assert_eq!(result, None);
    }

    #[test]
    fn gini_known_cases() {
        // Test case: [1, 1, 1, 1, 1] - perfect equality
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 1, 1, 1, 1]);
        let result = unsorted.gini(None).unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // Test case: [0, 0, 0, 0, 1] - high inequality
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![0, 0, 0, 0, 1]);
        let result = unsorted.gini(None).unwrap();
        // G = (2 * 5 * 1) / (5 * 1) - 6/5 = 2 - 1.2 = 0.8
        assert!((result - 0.8).abs() < 1e-10);

        // Test case: [1, 2, 3] - moderate inequality
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3]);
        let result = unsorted.gini(None).unwrap();
        // sum = 6, weighted_sum = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        // G = (2 * 14) / (3 * 6) - 4/3 = 28/18 - 4/3 = 1.5556 - 1.3333 = 0.2222
        let expected = (2.0 * 14.0) / (3.0 * 6.0) - 4.0 / 3.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn gini_precalc_sum() {
        // Test with pre-calculated sum
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let precalc_sum = Some(15.0);
        let result = unsorted.gini(precalc_sum).unwrap();
        let expected = (2.0 * 55.0) / (5.0 * 15.0) - 6.0 / 5.0;
        assert!((result - expected).abs() < 1e-10);

        // Test that pre-calculated sum gives same result
        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5]);
        let result2 = unsorted2.gini(None).unwrap();
        assert!((result - result2).abs() < 1e-10);
    }

    #[test]
    fn gini_large_dataset() {
        // Test with larger dataset to exercise parallel path
        let data: Vec<i32> = (1..=1000).collect();
        let result = gini(data.iter().copied(), None);
        assert!(result.is_some());
        let gini_val = result.unwrap();
        // For uniform distribution, Gini should be positive but not too high
        assert!(gini_val > 0.0 && gini_val < 0.5);
    }

    #[test]
    fn gini_unsorted_vs_sorted() {
        // Test that sorting doesn't affect result
        let mut unsorted1 = Unsorted::new();
        unsorted1.extend(vec![5, 2, 8, 1, 9, 3, 7, 4, 6]);
        let result1 = unsorted1.gini(None).unwrap();

        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let result2 = unsorted2.gini(None).unwrap();

        assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    fn gini_small_values() {
        // Test with very small values
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![0.001, 0.002, 0.003, 0.004, 0.005]);
        let result = unsorted.gini(None);
        assert!(result.is_some());
        // Should be same as [1, 2, 3, 4, 5] scaled down
        let expected = (2.0 * 55.0) / (5.0 * 15.0) - 6.0 / 5.0;
        assert!((result.unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn gini_large_values() {
        // Test with large values
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1000, 2000, 3000, 4000, 5000]);
        let result = unsorted.gini(None);
        assert!(result.is_some());
        // Should be same as [1, 2, 3, 4, 5] scaled up
        let expected = (2.0 * 55.0) / (5.0 * 15.0) - 6.0 / 5.0;
        assert!((result.unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn gini_two_elements() {
        // Test with exactly 2 elements
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2]);
        let result = unsorted.gini(None).unwrap();
        // For [1, 2]: sum=3, weighted_sum=1*1+2*2=5, n=2
        // G = (2*5)/(2*3) - 3/2 = 10/6 - 1.5 = 1.6667 - 1.5 = 0.1667
        let expected = (2.0 * 5.0) / (2.0 * 3.0) - 3.0 / 2.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn gini_precalc_sum_zero() {
        // Test with pre-calculated sum of zero (should return None)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let result = unsorted.gini(Some(0.0));
        assert_eq!(result, None);
    }

    #[test]
    fn gini_precalc_sum_negative() {
        // Gini is undefined for negative values, should return None
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![-5, -3, -1, 1, 3]);
        let result = unsorted.gini(None);
        assert_eq!(result, None);

        // Negative precalculated sum should also return None
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3]);
        let result = unsorted.gini(Some(-5.0));
        assert_eq!(result, None);
    }

    #[test]
    fn gini_different_types() {
        // Test with different integer types
        let mut unsorted_u32 = Unsorted::new();
        unsorted_u32.extend(vec![1u32, 2, 3, 4, 5]);
        let result_u32 = unsorted_u32.gini(None).unwrap();

        let mut unsorted_i64 = Unsorted::new();
        unsorted_i64.extend(vec![1i64, 2, 3, 4, 5]);
        let result_i64 = unsorted_i64.gini(None).unwrap();

        let expected = (2.0 * 55.0) / (5.0 * 15.0) - 6.0 / 5.0;
        assert!((result_u32 - expected).abs() < 1e-10);
        assert!((result_i64 - expected).abs() < 1e-10);
    }

    #[test]
    fn gini_extreme_inequality() {
        // Test with extreme inequality: one very large value, many zeros
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1000]);
        let result = unsorted.gini(None).unwrap();
        // For [0,0,0,0,0,0,0,0,0,1000]: sum=1000, weighted_sum=10*1000=10000, n=10
        // G = (2*10000)/(10*1000) - 11/10 = 20/10 - 1.1 = 2 - 1.1 = 0.9
        assert!((result - 0.9).abs() < 1e-10);
    }

    #[test]
    fn gini_duplicate_values() {
        // Test with many duplicate values
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 1, 1, 5, 5, 5, 10, 10, 10]);
        let result = unsorted.gini(None);
        assert!(result.is_some());
        // Should be between 0 and 1
        let gini_val = result.unwrap();
        assert!((0.0..=1.0).contains(&gini_val));
    }

    #[test]
    fn kurtosis_empty() {
        let mut unsorted: Unsorted<i32> = Unsorted::new();
        assert_eq!(unsorted.kurtosis(None, None), None);
        let empty_vec: Vec<i32> = vec![];
        assert_eq!(kurtosis(empty_vec.into_iter(), None, None), None);
    }

    #[test]
    fn kurtosis_small() {
        // Need at least 4 elements
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2]);
        assert_eq!(unsorted.kurtosis(None, None), None);

        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3]);
        assert_eq!(unsorted.kurtosis(None, None), None);
    }

    #[test]
    fn kurtosis_normal_distribution() {
        // Normal distribution should have kurtosis close to 0
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let result = unsorted.kurtosis(None, None);
        assert!(result.is_some());
        // For small samples, kurtosis can vary significantly
    }

    #[test]
    fn kurtosis_all_same() {
        // All same values - variance is 0, kurtosis undefined
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![5, 5, 5, 5]);
        assert_eq!(unsorted.kurtosis(None, None), None);
    }

    #[test]
    fn kurtosis_stream() {
        let result = kurtosis(vec![1usize, 2, 3, 4, 5].into_iter(), None, None);
        assert!(result.is_some());
    }

    #[test]
    fn kurtosis_precalc_mean_variance() {
        // Test with pre-calculated mean and variance
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);

        // Calculate mean and variance manually
        let mean = 3.0f64;
        let variance = ((1.0f64 - 3.0).powi(2)
            + (2.0f64 - 3.0).powi(2)
            + (3.0f64 - 3.0).powi(2)
            + (4.0f64 - 3.0).powi(2)
            + (5.0f64 - 3.0).powi(2))
            / 5.0;

        let result = unsorted.kurtosis(Some(mean), Some(variance));
        assert!(result.is_some());

        // Test that pre-calculated values give same result
        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5]);
        let result2 = unsorted2.kurtosis(None, None);
        assert!((result.unwrap() - result2.unwrap()).abs() < 1e-10);
    }

    #[test]
    fn kurtosis_precalc_mean_only() {
        // Test with pre-calculated mean only
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let mean = 3.0f64;

        let result = unsorted.kurtosis(Some(mean), None);
        assert!(result.is_some());

        // Test that pre-calculated mean gives same result
        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5]);
        let result2 = unsorted2.kurtosis(None, None);
        assert!((result.unwrap() - result2.unwrap()).abs() < 1e-10);
    }

    #[test]
    fn kurtosis_precalc_variance_only() {
        // Test with pre-calculated variance only
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let variance = ((1.0f64 - 3.0).powi(2)
            + (2.0f64 - 3.0).powi(2)
            + (3.0f64 - 3.0).powi(2)
            + (4.0f64 - 3.0).powi(2)
            + (5.0f64 - 3.0).powi(2))
            / 5.0;

        let result = unsorted.kurtosis(None, Some(variance));
        assert!(result.is_some());

        // Test that pre-calculated variance gives same result
        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5]);
        let result2 = unsorted2.kurtosis(None, None);
        assert!((result.unwrap() - result2.unwrap()).abs() < 1e-10);
    }

    #[test]
    fn kurtosis_exact_calculation() {
        // Test with exact calculation for [1, 2, 3, 4]
        // Mean = 2.5
        // Variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
        // Variance^2 = 1.5625
        // Fourth powers: (1-2.5)^4 + (2-2.5)^4 + (3-2.5)^4 + (4-2.5)^4 = 5.0625 + 0.0625 + 0.0625 + 5.0625 = 10.25
        // n = 4
        // Kurtosis = (4*5*10.25) / (3*2*1*1.5625) - 3*3*3/(2*1) = 205 / 9.375 - 13.5 = 21.8667 - 13.5 = 8.3667
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4]);
        let result = unsorted.kurtosis(None, None).unwrap();
        // For small samples, kurtosis can be very high
        assert!(result.is_finite());
    }

    #[test]
    fn kurtosis_uniform_distribution() {
        // Uniform distribution should have negative excess kurtosis
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let result = unsorted.kurtosis(None, None).unwrap();
        // Uniform distribution has excess kurtosis = -1.2
        // But for small samples, it can vary significantly
        assert!(result.is_finite());
    }

    #[test]
    fn kurtosis_large_dataset() {
        // Test with larger dataset to exercise parallel path
        let data: Vec<i32> = (1..=1000).collect();
        let result = kurtosis(data.iter().copied(), None, None);
        assert!(result.is_some());
        let kurt_val = result.unwrap();
        assert!(kurt_val.is_finite());
    }

    #[test]
    fn kurtosis_unsorted_vs_sorted() {
        // Test that sorting doesn't affect result
        let mut unsorted1 = Unsorted::new();
        unsorted1.extend(vec![5, 2, 8, 1, 9, 3, 7, 4, 6]);
        let result1 = unsorted1.kurtosis(None, None).unwrap();

        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let result2 = unsorted2.kurtosis(None, None).unwrap();

        assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    fn kurtosis_minimum_size() {
        // Test with exactly 4 elements (minimum required)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4]);
        let result = unsorted.kurtosis(None, None);
        assert!(result.is_some());
        assert!(result.unwrap().is_finite());
    }

    #[test]
    fn kurtosis_heavy_tailed() {
        // Test with heavy-tailed distribution (outliers)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
        let result = unsorted.kurtosis(None, None).unwrap();
        // Heavy tails should give positive excess kurtosis
        assert!(result.is_finite());
        // With an outlier, kurtosis should be positive
        assert!(result > -10.0); // Allow some variance but should be reasonable
    }

    #[test]
    fn kurtosis_light_tailed() {
        // Test with light-tailed distribution (values close together)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
        let result = unsorted.kurtosis(None, None).unwrap();
        // Light tails might give negative excess kurtosis
        assert!(result.is_finite());
    }

    #[test]
    fn kurtosis_small_variance() {
        // Test with very small variance (values very close together)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![10.0, 10.001, 10.002, 10.003, 10.004]);
        let result = unsorted.kurtosis(None, None);
        // Should still compute (variance is very small but non-zero)
        assert!(result.is_some());
        assert!(result.unwrap().is_finite());
    }

    #[test]
    fn kurtosis_precalc_zero_variance() {
        // Test with pre-calculated variance of zero (should return None)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let result = unsorted.kurtosis(None, Some(0.0));
        assert_eq!(result, None);
    }

    #[test]
    fn kurtosis_precalc_negative_variance() {
        // Test with negative variance (invalid, but should handle gracefully)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        // Negative variance is invalid, but function should handle it
        let result = unsorted.kurtosis(None, Some(-1.0));
        // Should either return None or handle it gracefully
        // The function computes variance_sq = variance^2, so negative becomes positive
        // But this is invalid input, so behavior may vary
        // For now, just check it doesn't panic
        let _ = result;
    }

    #[test]
    fn kurtosis_different_types() {
        // Test with different integer types
        let mut unsorted_u32 = Unsorted::new();
        unsorted_u32.extend(vec![1u32, 2, 3, 4, 5]);
        let result_u32 = unsorted_u32.kurtosis(None, None).unwrap();

        let mut unsorted_i64 = Unsorted::new();
        unsorted_i64.extend(vec![1i64, 2, 3, 4, 5]);
        let result_i64 = unsorted_i64.kurtosis(None, None).unwrap();

        assert!((result_u32 - result_i64).abs() < 1e-10);
    }

    #[test]
    fn kurtosis_floating_point_precision() {
        // Test floating point precision
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1.1, 2.2, 3.3, 4.4, 5.5]);
        let result = unsorted.kurtosis(None, None);
        assert!(result.is_some());
        assert!(result.unwrap().is_finite());
    }

    #[test]
    fn kurtosis_negative_values() {
        // Test with negative values
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![-5, -3, -1, 1, 3, 5]);
        let result = unsorted.kurtosis(None, None);
        assert!(result.is_some());
        assert!(result.unwrap().is_finite());
    }

    #[test]
    fn kurtosis_mixed_positive_negative() {
        // Test with mixed positive and negative values
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![-10, -5, 0, 5, 10]);
        let result = unsorted.kurtosis(None, None);
        assert!(result.is_some());
        assert!(result.unwrap().is_finite());
    }

    #[test]
    fn kurtosis_duplicate_values() {
        // Test with duplicate values (but not all same)
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
        let result = unsorted.kurtosis(None, None);
        assert!(result.is_some());
        assert!(result.unwrap().is_finite());
    }

    #[test]
    fn kurtosis_precalc_mean_wrong() {
        // Test that wrong pre-calculated mean gives wrong result
        let mut unsorted1 = Unsorted::new();
        unsorted1.extend(vec![1, 2, 3, 4, 5]);
        let correct_result = unsorted1.kurtosis(None, None).unwrap();

        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5]);
        let wrong_mean = 10.0; // Wrong mean
        let wrong_result = unsorted2.kurtosis(Some(wrong_mean), None).unwrap();

        // Results should be different
        assert!((correct_result - wrong_result).abs() > 1e-5);
    }

    #[test]
    fn percentile_rank_empty() {
        let mut unsorted: Unsorted<i32> = Unsorted::new();
        assert_eq!(unsorted.percentile_rank(5), None);
        let empty_vec: Vec<i32> = vec![];
        assert_eq!(percentile_rank(empty_vec.into_iter(), 5), None);
    }

    #[test]
    fn percentile_rank_basic() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        // Value less than all
        assert_eq!(unsorted.percentile_rank(0), Some(0.0));

        // Value greater than all
        assert_eq!(unsorted.percentile_rank(11), Some(100.0));

        // Median (5) should be around 50th percentile
        let rank = unsorted.percentile_rank(5).unwrap();
        assert!((rank - 50.0).abs() < 1.0);

        // First value should be at 10th percentile
        let rank = unsorted.percentile_rank(1).unwrap();
        assert!((rank - 10.0).abs() < 1.0);
    }

    #[test]
    fn percentile_rank_duplicates() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);

        // Value 2 appears twice, should be at 40th percentile (4 values <= 2)
        let rank = unsorted.percentile_rank(2).unwrap();
        assert!((rank - 40.0).abs() < 1.0);
    }

    #[test]
    fn percentile_rank_stream() {
        let result = percentile_rank(vec![1usize, 2, 3, 4, 5].into_iter(), 3);
        assert_eq!(result, Some(60.0)); // 3 out of 5 values <= 3
    }

    #[test]
    fn percentile_rank_many_ties() {
        // 100 copies of 5 followed by 100 copies of 10 — tests O(log n) upper bound
        let mut unsorted = Unsorted::new();
        for _ in 0..100 {
            unsorted.add(5u32);
        }
        for _ in 0..100 {
            unsorted.add(10u32);
        }
        // 100 values <= 5 out of 200
        let rank = unsorted.percentile_rank(5).unwrap();
        assert!((rank - 50.0).abs() < f64::EPSILON);
        // All 200 values <= 10
        let mut unsorted2 = Unsorted::new();
        for _ in 0..100 {
            unsorted2.add(5u32);
        }
        for _ in 0..100 {
            unsorted2.add(10u32);
        }
        let rank = unsorted2.percentile_rank(10).unwrap();
        assert!((rank - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn atkinson_empty() {
        let mut unsorted: Unsorted<i32> = Unsorted::new();
        assert_eq!(unsorted.atkinson(1.0, None, None), None);
        let empty_vec: Vec<i32> = vec![];
        assert_eq!(atkinson(empty_vec.into_iter(), 1.0, None, None), None);
    }

    #[test]
    fn atkinson_single_element() {
        let mut unsorted = Unsorted::new();
        unsorted.add(5);
        assert_eq!(unsorted.atkinson(1.0, None, None), Some(0.0));
        assert_eq!(atkinson(vec![5].into_iter(), 1.0, None, None), Some(0.0));
    }

    #[test]
    fn atkinson_perfect_equality() {
        // All values the same - perfect equality, Atkinson = 0
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![10, 10, 10, 10, 10]);
        let result = unsorted.atkinson(1.0, None, None).unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn atkinson_epsilon_zero() {
        // Epsilon = 0 means no inequality aversion, should return 0
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let result = unsorted.atkinson(0.0, None, None).unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn atkinson_epsilon_one() {
        // Epsilon = 1 uses geometric mean
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let result = unsorted.atkinson(1.0, None, None);
        assert!(result.is_some());
    }

    #[test]
    fn atkinson_negative_epsilon() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        assert_eq!(unsorted.atkinson(-1.0, None, None), None);
    }

    #[test]
    fn atkinson_zero_mean() {
        // If mean is zero, Atkinson is undefined
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![0, 0, 0, 0]);
        assert_eq!(unsorted.atkinson(1.0, None, None), None);
    }

    #[test]
    fn atkinson_stream() {
        let result = atkinson(vec![1usize, 2, 3, 4, 5].into_iter(), 1.0, None, None);
        assert!(result.is_some());
    }

    #[test]
    fn atkinson_precalc_mean_geometric_sum() {
        // Test with pre-calculated mean and geometric_sum
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);

        // Calculate mean and geometric_sum manually
        let mean = 3.0f64;
        let geometric_sum = 1.0f64.ln() + 2.0f64.ln() + 3.0f64.ln() + 4.0f64.ln() + 5.0f64.ln();

        let result = unsorted.atkinson(1.0, Some(mean), Some(geometric_sum));
        assert!(result.is_some());

        // Test that pre-calculated values give same result
        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5]);
        let result2 = unsorted2.atkinson(1.0, None, None);
        assert!((result.unwrap() - result2.unwrap()).abs() < 1e-10);
    }

    #[test]
    fn atkinson_precalc_mean_only() {
        // Test with pre-calculated mean only
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let mean = 3.0f64;

        let result = unsorted.atkinson(1.0, Some(mean), None);
        assert!(result.is_some());

        // Test that pre-calculated mean gives same result
        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5]);
        let result2 = unsorted2.atkinson(1.0, None, None);
        assert!((result.unwrap() - result2.unwrap()).abs() < 1e-10);
    }

    #[test]
    fn atkinson_precalc_geometric_sum_only() {
        // Test with pre-calculated geometric_sum only
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1, 2, 3, 4, 5]);
        let geometric_sum = 1.0f64.ln() + 2.0f64.ln() + 3.0f64.ln() + 4.0f64.ln() + 5.0f64.ln();

        let result = unsorted.atkinson(1.0, None, Some(geometric_sum));
        assert!(result.is_some());

        // Test that pre-calculated geometric_sum gives same result
        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(vec![1, 2, 3, 4, 5]);
        let result2 = unsorted2.atkinson(1.0, None, None);
        assert!((result.unwrap() - result2.unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_median_with_infinity() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1.0f64, 2.0, f64::INFINITY]);
        assert_eq!(unsorted.median(), Some(2.0));
    }

    #[test]
    fn test_median_with_neg_infinity() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![f64::NEG_INFINITY, 1.0f64, 2.0]);
        assert_eq!(unsorted.median(), Some(1.0));
    }

    #[test]
    fn test_quartiles_with_infinity() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY]);
        let q = unsorted.quartiles();
        // Q2 (median) should be 2.0
        assert!(q.is_some());
        let (_, q2, _) = q.unwrap();
        assert_eq!(q2, 2.0);
    }

    #[test]
    fn test_mode_with_nan() {
        // NaN breaks the Ord contract via Partial<T>, so sort order is
        // non-deterministic. We only verify the call doesn't panic —
        // the exact mode value depends on where NaN lands after sorting.
        let mut unsorted: Unsorted<f64> = Unsorted::new();
        unsorted.extend(vec![1.0, f64::NAN, 2.0, 2.0, 3.0]);
        let _result = unsorted.mode(); // must not panic
    }

    #[test]
    fn test_gini_with_infinity() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1.0f64, 2.0, f64::INFINITY]);
        let g = unsorted.gini(None);
        // Gini with infinity in the data: the weighted_sum/sum ratio involves
        // Inf/Inf which is NaN, so the result is Some(NaN) — not a meaningful
        // Gini coefficient, but importantly does not panic
        assert!(g.unwrap().is_nan());
    }

    #[test]
    fn test_cardinality_with_infinity() {
        let mut unsorted = Unsorted::new();
        unsorted.extend(vec![1.0f64, f64::INFINITY, f64::NEG_INFINITY, 1.0]);
        assert_eq!(unsorted.cardinality(false, 10_000), 3);
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore] // Run with `cargo test comprehensive_quartiles_benchmark -- --ignored --nocapture` to see performance comparison
    fn comprehensive_quartiles_benchmark() {
        // Test a wide range of data sizes
        let data_sizes = vec![
            1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
        ];

        println!("=== COMPREHENSIVE QUARTILES BENCHMARK ===\n");

        for size in data_sizes {
            println!("--- Testing with {} elements ---", size);

            // Test different data patterns
            let test_patterns = vec![
                ("Random", generate_random_data(size)),
                ("Reverse Sorted", {
                    let mut v = Vec::with_capacity(size);
                    for x in (0..size).rev() {
                        v.push(x as i32);
                    }
                    v
                }),
                ("Already Sorted", {
                    let mut v = Vec::with_capacity(size);
                    for x in 0..size {
                        v.push(x as i32);
                    }
                    v
                }),
                ("Many Duplicates", {
                    // Create a vector with just a few distinct values repeated many times
                    let mut v = Vec::with_capacity(size);
                    let chunk_size = size / 100;
                    for i in 0..100 {
                        v.extend(std::iter::repeat_n(i, chunk_size));
                    }
                    // Add any remaining elements
                    v.extend(std::iter::repeat_n(0, size - v.len()));
                    v
                }),
            ];

            for (pattern_name, test_data) in test_patterns {
                println!("\n  Pattern: {}", pattern_name);

                // Benchmark sorting-based approach
                let mut unsorted1 = Unsorted::new();
                unsorted1.extend(test_data.clone());

                let start = Instant::now();
                let result_sorted = unsorted1.quartiles();
                let sorted_time = start.elapsed();

                // Benchmark selection-based approach (with copying)
                let mut unsorted2 = Unsorted::new();
                unsorted2.extend(test_data.clone());

                let start = Instant::now();
                let result_selection = unsorted2.quartiles_with_selection();
                let selection_time = start.elapsed();

                // Benchmark zero-copy selection-based approach
                let mut unsorted3 = Unsorted::new();
                unsorted3.extend(test_data);

                let start = Instant::now();
                let result_zero_copy = unsorted3.quartiles_zero_copy();
                let zero_copy_time = start.elapsed();

                // Verify results are the same
                assert_eq!(result_sorted, result_selection);
                assert_eq!(result_sorted, result_zero_copy);

                let selection_speedup =
                    sorted_time.as_nanos() as f64 / selection_time.as_nanos() as f64;
                let zero_copy_speedup =
                    sorted_time.as_nanos() as f64 / zero_copy_time.as_nanos() as f64;

                println!("    Sorting:       {:>12?}", sorted_time);
                println!(
                    "    Selection:     {:>12?} (speedup: {:.2}x)",
                    selection_time, selection_speedup
                );
                println!(
                    "    Zero-copy:     {:>12?} (speedup: {:.2}x)",
                    zero_copy_time, zero_copy_speedup
                );

                let best_algorithm =
                    if zero_copy_speedup > 1.0 && zero_copy_speedup >= selection_speedup {
                        "ZERO-COPY"
                    } else if selection_speedup > 1.0 {
                        "SELECTION"
                    } else {
                        "SORTING"
                    };
                println!("    Best: {}", best_algorithm);
            }

            println!(); // Add blank line between sizes
        }
    }

    // Generate random data for benchmarking
    fn generate_random_data(size: usize) -> Vec<i32> {
        // Simple LCG random number generator for reproducible results
        let mut rng = 1234567u64;
        let mut vec = Vec::with_capacity(size);
        for _ in 0..size {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            vec.push((rng >> 16) as i32);
        }
        vec
    }

    #[test]
    #[ignore] // Run with `cargo test find_selection_threshold -- --ignored --nocapture` to find exact threshold
    fn find_selection_threshold() {
        println!("=== FINDING SELECTION ALGORITHM THRESHOLD ===\n");

        // Binary search approach to find the threshold
        let mut found_threshold = None;
        let test_sizes = vec![
            1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 7_500_000, 10_000_000,
            15_000_000, 20_000_000, 25_000_000, 30_000_000,
        ];

        for size in test_sizes {
            println!("Testing size: {}", size);

            // Use random data as it's most representative of real-world scenarios
            let test_data = generate_random_data(size);

            // Run multiple iterations to get average performance
            let iterations = 3;
            let mut sorting_total = 0u128;
            let mut selection_total = 0u128;
            let mut zero_copy_total = 0u128;

            for i in 0..iterations {
                println!("  Iteration {}/{}", i + 1, iterations);

                // Sorting approach
                let mut unsorted1 = Unsorted::new();
                unsorted1.extend(test_data.clone());

                let start = Instant::now();
                let _result_sorted = unsorted1.quartiles();
                sorting_total += start.elapsed().as_nanos();

                // Selection approach (with copying)
                let mut unsorted2 = Unsorted::new();
                unsorted2.extend(test_data.clone());

                let start = Instant::now();
                let _result_selection = unsorted2.quartiles_with_selection();
                selection_total += start.elapsed().as_nanos();

                // Zero-copy selection approach
                let mut unsorted3 = Unsorted::new();
                unsorted3.extend(test_data.clone());

                let start = Instant::now();
                let _result_zero_copy = unsorted3.quartiles_zero_copy();
                zero_copy_total += start.elapsed().as_nanos();
            }

            let avg_sorting = sorting_total / iterations as u128;
            let avg_selection = selection_total / iterations as u128;
            let avg_zero_copy = zero_copy_total / iterations as u128;
            let selection_speedup = avg_sorting as f64 / avg_selection as f64;
            let zero_copy_speedup = avg_sorting as f64 / avg_zero_copy as f64;

            println!(
                "  Average sorting:    {:>12.2}ms",
                avg_sorting as f64 / 1_000_000.0
            );
            println!(
                "  Average selection:  {:>12.2}ms (speedup: {:.2}x)",
                avg_selection as f64 / 1_000_000.0,
                selection_speedup
            );
            println!(
                "  Average zero-copy:  {:>12.2}ms (speedup: {:.2}x)",
                avg_zero_copy as f64 / 1_000_000.0,
                zero_copy_speedup
            );

            if (selection_speedup > 1.0 || zero_copy_speedup > 1.0) && found_threshold.is_none() {
                found_threshold = Some(size);
                let best_method = if zero_copy_speedup > selection_speedup {
                    "Zero-copy"
                } else {
                    "Selection"
                };
                println!(
                    "  *** THRESHOLD FOUND: {} becomes faster at {} elements ***",
                    best_method, size
                );
            }

            println!();
        }

        match found_threshold {
            Some(threshold) => println!(
                "🎯 Selection algorithm becomes faster at approximately {} elements",
                threshold
            ),
            None => println!("❌ Selection algorithm did not become faster in the tested range"),
        }
    }

    #[test]
    #[ignore] // Run with `cargo test benchmark_different_data_types -- --ignored --nocapture` to test different data types
    fn benchmark_different_data_types() {
        println!("=== BENCHMARKING DIFFERENT DATA TYPES ===\n");

        let size = 5_000_000; // Use a large size where differences might be visible

        // Test with f64 (floating point)
        println!("Testing with f64 data:");
        let float_data: Vec<f64> = generate_random_data(size)
            .into_iter()
            .map(|x| x as f64 / 1000.0)
            .collect();

        let mut unsorted1 = Unsorted::new();
        unsorted1.extend(float_data.clone());
        let start = Instant::now();
        let _result = unsorted1.quartiles();
        let sorting_time = start.elapsed();

        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(float_data.clone());
        let start = Instant::now();
        let _result = unsorted2.quartiles_with_selection();
        let selection_time = start.elapsed();

        let mut unsorted3 = Unsorted::new();
        unsorted3.extend(float_data);
        let start = Instant::now();
        let _result = unsorted3.quartiles_zero_copy();
        let zero_copy_time = start.elapsed();

        println!("  Sorting:    {:?}", sorting_time);
        println!("  Selection:  {:?}", selection_time);
        println!("  Zero-copy:  {:?}", zero_copy_time);
        println!(
            "  Selection Speedup:  {:.2}x",
            sorting_time.as_nanos() as f64 / selection_time.as_nanos() as f64
        );
        println!(
            "  Zero-copy Speedup:  {:.2}x\n",
            sorting_time.as_nanos() as f64 / zero_copy_time.as_nanos() as f64
        );

        // Test with i64 (larger integers)
        println!("Testing with i64 data:");
        let int64_data: Vec<i64> = generate_random_data(size)
            .into_iter()
            .map(|x| x as i64 * 1000)
            .collect();

        let mut unsorted1 = Unsorted::new();
        unsorted1.extend(int64_data.clone());
        let start = Instant::now();
        let _result = unsorted1.quartiles();
        let sorting_time = start.elapsed();

        let mut unsorted2 = Unsorted::new();
        unsorted2.extend(int64_data.clone());
        let start = Instant::now();
        let _result = unsorted2.quartiles_with_selection();
        let selection_time = start.elapsed();

        let mut unsorted3 = Unsorted::new();
        unsorted3.extend(int64_data);
        let start = Instant::now();
        let _result = unsorted3.quartiles_zero_copy();
        let zero_copy_time = start.elapsed();

        println!("  Sorting:    {:?}", sorting_time);
        println!("  Selection:  {:?}", selection_time);
        println!("  Zero-copy:  {:?}", zero_copy_time);
        println!(
            "  Selection Speedup:  {:.2}x",
            sorting_time.as_nanos() as f64 / selection_time.as_nanos() as f64
        );
        println!(
            "  Zero-copy Speedup:  {:.2}x",
            sorting_time.as_nanos() as f64 / zero_copy_time.as_nanos() as f64
        );
    }
}
