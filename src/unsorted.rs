use num_traits::ToPrimitive;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use rayon::slice::ParallelSliceMut;

use serde::{Deserialize, Serialize};

use {crate::Commute, crate::Partial};

const PARALLEL_THRESHOLD: usize = 10_000;

/// Compute the exact median on a stream of data.
///
/// (This has time complexity `O(nlogn)` and space complexity `O(n)`.)
pub fn median<I>(it: I) -> Option<f64>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive,
{
    it.collect::<Unsorted<_>>().median()
}

/// Compute the median absolute deviation (MAD) on a stream of data.
pub fn mad<I>(it: I, precalc_median: Option<f64>) -> Option<f64>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive,
{
    it.collect::<Unsorted<_>>().mad(precalc_median)
}

/// Compute the exact 1-, 2-, and 3-quartiles (Q1, Q2 a.k.a. median, and Q3) on a stream of data.
///
/// (This has time complexity `O(n log n)` and space complexity `O(n)`.)
pub fn quartiles<I>(it: I) -> Option<(f64, f64, f64)>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd + ToPrimitive,
{
    it.collect::<Unsorted<_>>().quartiles()
}

/// Compute the exact mode on a stream of data.
///
/// (This has time complexity `O(nlogn)` and space complexity `O(n)`.)
///
/// If the data does not have a mode, then `None` is returned.
pub fn mode<T, I>(it: I) -> Option<T>
where
    T: PartialOrd + Clone,
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
pub fn modes<T, I>(it: I) -> (Vec<T>, usize, u32)
where
    T: PartialOrd + Clone,
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
pub fn antimodes<T, I>(it: I) -> (Vec<T>, usize, u32)
where
    T: PartialOrd + Clone,
    I: Iterator<Item = T>,
{
    let (antimodes_result, antimodes_count, antimodes_occurrences) =
        it.collect::<Unsorted<T>>().antimodes();
    (antimodes_result, antimodes_count, antimodes_occurrences)
}

fn median_on_sorted<T>(data: &[T]) -> Option<f64>
where
    T: PartialOrd + ToPrimitive,
{
    Some(match data.len() {
        // Empty slice case - return None early
        0 => return None,
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
        return None;
    }
    let median_obs = precalc_median.unwrap_or_else(|| median_on_sorted(data).unwrap());

    // Use adaptive parallel processing based on data size
    let mut abs_diff_vec = if data.len() < PARALLEL_THRESHOLD {
        // Sequential processing for small datasets
        let mut vec = Vec::with_capacity(data.len());
        for x in data {
            vec.push((median_obs - unsafe { x.to_f64().unwrap_unchecked() }).abs());
        }
        vec
    } else {
        // Parallel processing for large datasets
        data.par_iter()
            .map(|x| (median_obs - unsafe { x.to_f64().unwrap_unchecked() }).abs())
            .collect()
    };

    // Use adaptive sorting based on size
    if abs_diff_vec.len() < PARALLEL_THRESHOLD {
        abs_diff_vec.sort_unstable_by(|a, b| unsafe { a.partial_cmp(b).unwrap_unchecked() });
    } else {
        abs_diff_vec.par_sort_unstable_by(|a, b| unsafe { a.partial_cmp(b).unwrap_unchecked() });
    }
    median_on_sorted(&abs_diff_vec)
}

/// Selection algorithm to find the k-th smallest element in O(n) average time.
/// This is an implementation of quickselect algorithm.
fn quickselect<T>(data: &mut [Partial<T>], k: usize) -> Option<&T>
where
    T: PartialOrd,
{
    if data.is_empty() || k >= data.len() {
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

/// Zero-copy selection algorithm that works with indices instead of copying data.
/// This avoids the overhead of cloning data elements.
fn quickselect_by_index<'a, T>(
    data: &'a [Partial<T>],
    indices: &mut [usize],
    k: usize,
) -> Option<&'a T>
where
    T: PartialOrd,
{
    if data.is_empty() || indices.is_empty() || k >= indices.len() {
        return None;
    }

    let mut left = 0;
    let mut right = indices.len() - 1;

    loop {
        if left == right {
            return Some(&data[indices[left]].0);
        }

        // Use median-of-three pivot selection for better performance
        let pivot_idx = median_of_three_pivot_by_index(data, indices, left, right);
        let pivot_idx = partition_by_index(data, indices, left, right, pivot_idx);

        match k.cmp(&pivot_idx) {
            std::cmp::Ordering::Equal => return Some(&data[indices[pivot_idx]].0),
            std::cmp::Ordering::Less => right = pivot_idx - 1,
            std::cmp::Ordering::Greater => left = pivot_idx + 1,
        }
    }
}

/// Select the median of three elements as pivot for better quickselect performance
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

/// Select the median of three elements as pivot using indices
fn median_of_three_pivot_by_index<T>(
    data: &[Partial<T>],
    indices: &[usize],
    left: usize,
    right: usize,
) -> usize
where
    T: PartialOrd,
{
    let mid = left + (right - left) / 2;

    if data[indices[left]] <= data[indices[mid]] {
        if data[indices[mid]] <= data[indices[right]] {
            mid
        } else if data[indices[left]] <= data[indices[right]] {
            right
        } else {
            left
        }
    } else if data[indices[left]] <= data[indices[right]] {
        left
    } else if data[indices[mid]] <= data[indices[right]] {
        right
    } else {
        mid
    }
}

/// Partition function for quickselect
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

/// Partition function for quickselect using indices
fn partition_by_index<T>(
    data: &[Partial<T>],
    indices: &mut [usize],
    left: usize,
    right: usize,
    pivot_idx: usize,
) -> usize
where
    T: PartialOrd,
{
    // Move pivot to end
    indices.swap(pivot_idx, right);
    let mut store_idx = left;

    // Cache pivot index and value for better cache locality
    // This reduces indirection: indices[right] -> data[indices[right]]
    // Safety: right is guaranteed to be in bounds
    let pivot_idx_cached = unsafe { *indices.get_unchecked(right) };
    let pivot_val = unsafe { data.get_unchecked(pivot_idx_cached) };

    // Move all elements smaller than pivot to the left
    let mut elem_idx: usize;
    for i in left..right {
        // Safety: i and store_idx are guaranteed to be in bounds
        elem_idx = unsafe { *indices.get_unchecked(i) };
        if unsafe { data.get_unchecked(elem_idx) <= pivot_val } {
            indices.swap(i, store_idx);
            store_idx += 1;
        }
    }

    // Move pivot to its final place
    indices.swap(store_idx, right);
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
        0..=2 => return None,
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

/// Compute quartiles using selection algorithm in O(n) time instead of O(n log n) sorting.
/// This implementation follows Method 3 from `<https://en.wikipedia.org/wiki/Quartile>`
fn quartiles_with_selection<T>(data: &mut [Partial<T>]) -> Option<(f64, f64, f64)>
where
    T: PartialOrd + ToPrimitive,
{
    let len = data.len();

    // Early return for small arrays
    match len {
        0..=2 => return None,
        3 => {
            // For 3 elements, we need to find the sorted order using selection
            let min_val = quickselect(data, 0)?.to_f64()?;
            let med_val = quickselect(data, 1)?.to_f64()?;
            let max_val = quickselect(data, 2)?.to_f64()?;
            return Some((min_val, med_val, max_val));
        }
        _ => {}
    }

    // Calculate k and remainder in one division
    let k = len / 4;
    let remainder = len % 4;

    // Use selection algorithm to find the required order statistics
    match remainder {
        0 => {
            // Length is multiple of 4 (4k)
            // Q1 = (x_{k-1} + x_k) / 2
            // Q2 = (x_{2k-1} + x_{2k}) / 2
            // Q3 = (x_{3k-1} + x_{3k}) / 2
            let q1_low = quickselect(data, k - 1)?.to_f64()?;
            let q1_high = quickselect(data, k)?.to_f64()?;
            let q1 = f64::midpoint(q1_low, q1_high);

            let q2_low = quickselect(data, 2 * k - 1)?.to_f64()?;
            let q2_high = quickselect(data, 2 * k)?.to_f64()?;
            let q2 = f64::midpoint(q2_low, q2_high);

            let q3_low = quickselect(data, 3 * k - 1)?.to_f64()?;
            let q3_high = quickselect(data, 3 * k)?.to_f64()?;
            let q3 = f64::midpoint(q3_low, q3_high);

            Some((q1, q2, q3))
        }
        1 => {
            // Length is 4k + 1
            // Q1 = (x_{k-1} + x_k) / 2
            // Q2 = x_{2k}
            // Q3 = (x_{3k} + x_{3k+1}) / 2
            let q1_low = quickselect(data, k - 1)?.to_f64()?;
            let q1_high = quickselect(data, k)?.to_f64()?;
            let q1 = f64::midpoint(q1_low, q1_high);

            let q2 = quickselect(data, 2 * k)?.to_f64()?;

            let q3_low = quickselect(data, 3 * k)?.to_f64()?;
            let q3_high = quickselect(data, 3 * k + 1)?.to_f64()?;
            let q3 = f64::midpoint(q3_low, q3_high);

            Some((q1, q2, q3))
        }
        2 => {
            // Length is 4k + 2
            // Q1 = x_k
            // Q2 = (x_{2k} + x_{2k+1}) / 2
            // Q3 = x_{3k+1}
            let q1 = quickselect(data, k)?.to_f64()?;

            let q2_low = quickselect(data, 2 * k)?.to_f64()?;
            let q2_high = quickselect(data, 2 * k + 1)?.to_f64()?;
            let q2 = f64::midpoint(q2_low, q2_high);

            let q3 = quickselect(data, 3 * k + 1)?.to_f64()?;

            Some((q1, q2, q3))
        }
        _ => {
            // Length is 4k + 3
            // Q1 = x_k
            // Q2 = x_{2k+1}
            // Q3 = x_{3k+2}
            let q1 = quickselect(data, k)?.to_f64()?;
            let q2 = quickselect(data, 2 * k + 1)?.to_f64()?;
            let q3 = quickselect(data, 3 * k + 2)?.to_f64()?;

            Some((q1, q2, q3))
        }
    }
}

/// Zero-copy quartiles computation using index-based selection.
/// This avoids copying data by working with an array of indices.
fn quartiles_with_zero_copy_selection<T>(data: &[Partial<T>]) -> Option<(f64, f64, f64)>
where
    T: PartialOrd + ToPrimitive,
{
    let len = data.len();

    // Early return for small arrays
    match len {
        0..=2 => return None,
        3 => {
            // For 3 elements, create indices and find sorted order
            let mut indices = vec![0, 1, 2];
            let min_val = quickselect_by_index(data, &mut indices, 0)?.to_f64()?;
            let med_val = quickselect_by_index(data, &mut indices, 1)?.to_f64()?;
            let max_val = quickselect_by_index(data, &mut indices, 2)?.to_f64()?;
            return Some((min_val, med_val, max_val));
        }
        _ => {}
    }

    // Create indices array once
    let mut indices: Vec<usize> = (0..len).collect();

    // Calculate k and remainder in one division
    let k = len / 4;
    let remainder = len % 4;

    // Use zero-copy selection algorithm to find the required order statistics
    match remainder {
        0 => {
            // Length is multiple of 4 (4k)
            let q1_low = quickselect_by_index(data, &mut indices, k - 1)?.to_f64()?;
            let q1_high = quickselect_by_index(data, &mut indices, k)?.to_f64()?;
            let q1 = f64::midpoint(q1_low, q1_high);

            let q2_low = quickselect_by_index(data, &mut indices, 2 * k - 1)?.to_f64()?;
            let q2_high = quickselect_by_index(data, &mut indices, 2 * k)?.to_f64()?;
            let q2 = f64::midpoint(q2_low, q2_high);

            let q3_low = quickselect_by_index(data, &mut indices, 3 * k - 1)?.to_f64()?;
            let q3_high = quickselect_by_index(data, &mut indices, 3 * k)?.to_f64()?;
            let q3 = f64::midpoint(q3_low, q3_high);

            Some((q1, q2, q3))
        }
        1 => {
            // Length is 4k + 1
            let q1_low = quickselect_by_index(data, &mut indices, k - 1)?.to_f64()?;
            let q1_high = quickselect_by_index(data, &mut indices, k)?.to_f64()?;
            let q1 = f64::midpoint(q1_low, q1_high);

            let q2 = quickselect_by_index(data, &mut indices, 2 * k)?.to_f64()?;

            let q3_low = quickselect_by_index(data, &mut indices, 3 * k)?.to_f64()?;
            let q3_high = quickselect_by_index(data, &mut indices, 3 * k + 1)?.to_f64()?;
            let q3 = f64::midpoint(q3_low, q3_high);

            Some((q1, q2, q3))
        }
        2 => {
            // Length is 4k + 2
            let q1 = quickselect_by_index(data, &mut indices, k)?.to_f64()?;

            let q2_low = quickselect_by_index(data, &mut indices, 2 * k)?.to_f64()?;
            let q2_high = quickselect_by_index(data, &mut indices, 2 * k + 1)?.to_f64()?;
            let q2 = f64::midpoint(q2_low, q2_high);

            let q3 = quickselect_by_index(data, &mut indices, 3 * k + 1)?.to_f64()?;

            Some((q1, q2, q3))
        }
        _ => {
            // Length is 4k + 3
            let q1 = quickselect_by_index(data, &mut indices, k)?.to_f64()?;
            let q2 = quickselect_by_index(data, &mut indices, 2 * k + 1)?.to_f64()?;
            let q3 = quickselect_by_index(data, &mut indices, 3 * k + 2)?.to_f64()?;

            Some((q1, q2, q3))
        }
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

/// Computes both modes and antimodes from a sorted iterator of values.
///
/// # Arguments
///
/// * `it` - A sorted iterator of values
/// * `size` - The total number of elements in the iterator
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
///
/// # Notes
///
/// - Mode is the most frequently occurring value(s)
/// - Antimode is the least frequently occurring value(s)
/// - Only returns up to 10 antimodes to avoid returning the full set when all values are unique
/// - For empty iterators, returns empty vectors and zero counts
/// - For single value iterators, returns that value as the mode and empty antimode
/// - When all values occur exactly once, returns empty mode and up to 10 values as antimodes
///
/// # Type Parameters
///
/// * `T`: The value type that implements `PartialOrd` + `Clone`
/// * `I`: The iterator type
#[allow(clippy::type_complexity)]
#[inline]
fn modes_and_antimodes_on_sorted<T, I>(
    mut it: I,
    size: usize,
) -> ((Vec<T>, usize, u32), (Vec<T>, usize, u32))
where
    T: PartialOrd + Clone,
    I: Iterator<Item = T>,
{
    // Early return for empty iterator
    let Some(first) = it.next() else {
        return ((Vec::new(), 0, 0), (Vec::new(), 0, 0));
    };

    // Estimate capacity using square root of size
    #[allow(clippy::cast_sign_loss)]
    let mut runs: Vec<(T, u32)> =
        Vec::with_capacity(((size as f64).sqrt() as usize).clamp(16, 1_000));

    let mut current_value = first;
    let mut current_count = 1;
    let mut highest_count = 1;
    let mut lowest_count = u32::MAX;

    // Count consecutive runs - optimized to reduce allocations
    for x in it {
        if x == current_value {
            current_count += 1;
            highest_count = highest_count.max(current_count);
        } else {
            runs.push((current_value, current_count));
            lowest_count = lowest_count.min(current_count);
            current_value = x;
            current_count = 1;
        }
    }
    runs.push((current_value, current_count));
    lowest_count = lowest_count.min(current_count);

    // Early return if only one unique value
    if runs.len() == 1 {
        let (val, count) = runs.pop().unwrap();
        return ((vec![val], 1, count), (Vec::new(), 0, 0));
    }

    // Special case: if all values appear exactly once
    if highest_count == 1 {
        let antimodes_count = runs.len().min(10);
        let total_count = runs.len();
        let mut antimodes = Vec::with_capacity(antimodes_count);
        for (val, _) in runs.into_iter().take(antimodes_count) {
            antimodes.push(val);
        }
        // For modes: empty, count 0, occurrences 0 (not 1, 1)
        return ((Vec::new(), 0, 0), (antimodes, total_count, 1));
    }

    // Estimate capacities based on the number of runs
    // For modes: typically 1-3 modes, rarely more than 10% of runs
    // For antimodes: we only collect up to 10, but need to count all
    let estimated_modes = (runs.len() / 10).clamp(1, 10);
    let estimated_antimodes = 10.min(runs.len());

    // Collect indices first to avoid unnecessary cloning
    let mut modes_indices = Vec::with_capacity(estimated_modes);
    let mut antimodes_indices = Vec::with_capacity(estimated_antimodes);
    let mut mode_count = 0;
    let mut antimodes_count = 0;
    let mut antimodes_collected = 0_u32;

    // Count and collect mode/antimode indices simultaneously
    for (idx, (_, count)) in runs.iter().enumerate() {
        if *count == highest_count {
            modes_indices.push(idx);
            mode_count += 1;
        }
        if *count == lowest_count {
            antimodes_count += 1;
            if antimodes_collected < 10 {
                antimodes_indices.push(idx);
                antimodes_collected += 1;
            }
        }
    }

    // Extract values only for the indices we need, avoiding unnecessary clones
    let modes_result: Vec<T> = modes_indices
        .into_iter()
        .map(|idx| runs[idx].0.clone())
        .collect();
    let antimodes_result: Vec<T> = antimodes_indices
        .into_iter()
        .map(|idx| runs[idx].0.clone())
        .collect();

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
#[derive(Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Unsorted<T> {
    sorted: bool,
    data: Vec<Partial<T>>,
}

impl<T: PartialOrd> Unsorted<T> {
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

impl<T: PartialOrd + PartialEq + Clone> Unsorted<T> {
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
            // Parallel processing using chunks
            let chunks = self.data.par_chunks(CHUNK_SIZE);

            // Process each chunk and combine results
            let chunk_results: Vec<u64> = chunks
                .map(|chunk| {
                    // Count unique elements within each chunk
                    let mut count = 1; // Start at 1 for first element
                    for window in chunk.windows(2) {
                        // safety: windows(2) guarantees window has length 2
                        if unsafe { window.get_unchecked(0) != window.get_unchecked(1) } {
                            count += 1;
                        }
                    }
                    count
                })
                .collect();

            // Combine results from chunks, checking boundaries between chunks
            // Pre-compute chunk boundaries to avoid repeated multiplications
            let mut total = 0;
            let mut curr_chunk_start_idx = 0;

            for (i, &count) in chunk_results.iter().enumerate() {
                total += count;

                // Check boundary between chunks
                if i > 0 {
                    // Pre-compute indices once to avoid repeated multiplication
                    // Safety: These indices are guaranteed to be in bounds since we're iterating
                    // over chunk_results which was created from valid chunks of self.data
                    unsafe {
                        let prev_chunk_end_idx = curr_chunk_start_idx - 1;
                        let prev_chunk_end = self.data.get_unchecked(prev_chunk_end_idx);
                        let curr_chunk_start = self.data.get_unchecked(curr_chunk_start_idx);
                        if prev_chunk_end == curr_chunk_start {
                            total -= 1;
                        }
                    }
                }

                // Update for next iteration
                curr_chunk_start_idx += CHUNK_SIZE;
            }

            total
        } else {
            // Sequential processing

            // the statement below is equivalent to:
            // let mut count = if self.data.is_empty() { 0 } else { 1 };
            let mut count = u64::from(!self.data.is_empty());

            for window in self.data.windows(2) {
                // safety: windows(2) guarantees window has length 2
                if unsafe { window.get_unchecked(0) != window.get_unchecked(1) } {
                    count += 1;
                }
            }
            count
        }
    }
}

impl<T: PartialOrd + Clone> Unsorted<T> {
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
        modes_and_antimodes_on_sorted(self.data.iter().map(|p| p.0.clone()), self.len()).0
    }

    /// Returns the antimodes of the data.
    /// `antimodes_result` only returns the first 10 antimodes
    #[inline]
    fn antimodes(&mut self) -> (Vec<T>, usize, u32) {
        if self.data.is_empty() {
            return (Vec::new(), 0, 0);
        }
        self.sort();
        modes_and_antimodes_on_sorted(self.data.iter().map(|p| p.0.clone()), self.len()).1
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
        modes_and_antimodes_on_sorted(self.data.iter().map(|p| p.0.clone()), self.len())
    }
}

impl<T: PartialOrd + ToPrimitive> Unsorted<T> {
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

impl<T: PartialOrd + ToPrimitive> Unsorted<T> {
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

impl<T: PartialOrd + ToPrimitive> Unsorted<T> {
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

impl<T: PartialOrd + ToPrimitive + Clone> Unsorted<T> {
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
        // Create a copy using collect to avoid mutating the original for selection
        let mut data_copy: Vec<Partial<T>> =
            self.data.iter().map(|x| Partial(x.0.clone())).collect();
        quartiles_with_selection(&mut data_copy)
    }
}

impl<T: PartialOrd + ToPrimitive> Unsorted<T> {
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

impl<T: PartialOrd> Commute for Unsorted<T> {
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
            data: Vec::with_capacity(10_000),
            sorted: true, // empty is sorted
        }
    }
}

impl<T: PartialOrd> FromIterator<T> for Unsorted<T> {
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

    // Create a sorted vector of unique percentiles
    let mut unique_percentiles = percentiles.to_vec();
    unique_percentiles.sort_unstable();
    unique_percentiles.dedup();

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

impl<T: PartialOrd + Clone> Unsorted<T> {
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
                        v.extend(std::iter::repeat(i as i32).take(chunk_size));
                    }
                    // Add any remaining elements
                    v.extend(std::iter::repeat(0).take(size - v.len()));
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
