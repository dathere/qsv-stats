use num_traits::ToPrimitive;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;
use rayon::slice::ParallelSliceMut;

use serde::{Deserialize, Serialize};

use {crate::Commute, crate::Partial};

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
/// (This has time complexity `O(nlogn)` and space complexity `O(n)`.)
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
        len if len % 2 == 0 => {
            let idx = len / 2;
            // Safety: we know these indices are valid because we checked len is even and non-zero,
            // so idx-1 and idx are valid indices into data
            let v1 = unsafe { data.get_unchecked(idx - 1) }.to_f64()?;
            let v2 = unsafe { data.get_unchecked(idx) }.to_f64()?;
            (v1 + v2) / 2.0
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
    let median_obs =
        precalc_median.map_or_else(|| median_on_sorted(data).unwrap(), |precalc| precalc);

    let mut abs_diff_vec: Vec<f64> = data
        .par_iter()
        .map(|x| (median_obs - unsafe { x.to_f64().unwrap_unchecked() }).abs())
        .collect();

    abs_diff_vec.par_sort_unstable_by(|a, b| unsafe { a.partial_cmp(b).unwrap_unchecked() });
    median_on_sorted(&abs_diff_vec)
}

fn quartiles_on_sorted<T>(data: &[T]) -> Option<(f64, f64, f64)>
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
                        data.get_unchecked(0).to_f64()?,
                        data.get_unchecked(1).to_f64()?,
                        data.get_unchecked(2).to_f64()?,
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
                // Length is multiple of 4 (4k)
                // Q1 = (x_{k-1} + x_k) / 2
                // Q2 = (x_{2k-1} + x_{2k}) / 2
                // Q3 = (x_{3k-1} + x_{3k}) / 2
                let q1 =
                    (data.get_unchecked(k - 1).to_f64()? + data.get_unchecked(k).to_f64()?) / 2.0;
                let q2 = (data.get_unchecked(2 * k - 1).to_f64()?
                    + data.get_unchecked(2 * k).to_f64()?)
                    / 2.0;
                let q3 = (data.get_unchecked(3 * k - 1).to_f64()?
                    + data.get_unchecked(3 * k).to_f64()?)
                    / 2.0;
                (q1, q2, q3)
            }
            1 => {
                // Let data = {x_i}_{i=0..4k+1} where k is positive integer.
                // Median q2 = x_{2k}.
                // If we divide data other than q2 into two parts {x_i < q2}
                // as L and {x_i > q2} as R, #L == #R == 2k holds true. Thus,
                // q1 = (x_{k-1} + x_{k}) / 2 and q3 = (x_{3k} + x_{3k+1}) / 2.
                // =============
                // Length is 4k + 1
                // Q1 = (x_{k-1} + x_k) / 2
                // Q2 = x_{2k}
                // Q3 = (x_{3k} + x_{3k+1}) / 2
                let q1 =
                    (data.get_unchecked(k - 1).to_f64()? + data.get_unchecked(k).to_f64()?) / 2.0;
                let q2 = data.get_unchecked(2 * k).to_f64()?;
                let q3 = (data.get_unchecked(3 * k).to_f64()?
                    + data.get_unchecked(3 * k + 1).to_f64()?)
                    / 2.0;
                (q1, q2, q3)
            }
            2 => {
                // Let data = {x_i}_{i=0..4k+2} where k is positive integer.
                // Median q2 = (x_{(2k+1)-1} + x_{2k+1}) / 2.
                // If we divide data into two parts {x_i < q2} as L and
                // {x_i > q2} as R, it's true that #L == #R == 2k+1.
                // Thus, q1 = x_{k} and q3 = x_{3k+1}.
                // =============
                // Length is 4k + 2
                // Q1 = x_k
                // Q2 = (x_{2k} + x_{2k+1}) / 2
                // Q3 = x_{3k+1}
                let q1 = data.get_unchecked(k).to_f64()?;
                let q2 = (data.get_unchecked(2 * k).to_f64()?
                    + data.get_unchecked(2 * k + 1).to_f64()?)
                    / 2.0;
                let q3 = data.get_unchecked(3 * k + 1).to_f64()?;
                (q1, q2, q3)
            }
            _ => {
                // Let data = {x_i}_{i=0..4k+3} where k is positive integer.
                // Median q2 = x_{2k+1}.
                // If we divide data other than q2 into two parts {x_i < q2}
                // as L and {x_i > q2} as R, #L == #R == 2k+1 holds true.
                // Thus, q1 = x_{k} and q3 = x_{3k+2}.
                // =============
                // Length is 4k + 3
                // Q1 = x_k
                // Q2 = x_{2k+1}
                // Q3 = x_{3k+2}
                let q1 = data.get_unchecked(k).to_f64()?;
                let q2 = data.get_unchecked(2 * k + 1).to_f64()?;
                let q3 = data.get_unchecked(3 * k + 2).to_f64()?;
                (q1, q2, q3)
            }
        })
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
/// * `T`: The value type that implements PartialOrd + Clone
/// * `I`: The iterator type
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
    let mut runs: Vec<(T, u32)> = Vec::with_capacity(
        ((size as f64).sqrt() as usize).clamp(16, 1_000), // Min 16, max 1000
    );

    let mut current_value = first;
    let mut current_count = 1;
    let mut highest_count = 1;
    let mut lowest_count = u32::MAX;

    // Count consecutive runs
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
        let mut antimodes = Vec::with_capacity(10.min(runs.len()));
        let total_count = runs.len();
        for (val, _) in runs.into_iter().take(10) {
            antimodes.push(val);
        }
        return ((Vec::new(), 0, 0), (antimodes, total_count, 1));
    }

    // Collect modes and antimodes in a single pass
    let mut modes_result = Vec::with_capacity(10);
    let mut antimodes_result = Vec::with_capacity(10);
    let mut mode_count = 0;
    let mut antimodes_count = 0;

    for (val, count) in &runs {
        if *count == highest_count {
            modes_result.push(val.clone());
            mode_count += 1;
        }
        if *count == lowest_count {
            antimodes_count += 1;
            if antimodes_result.len() < 10 {
                antimodes_result.push(val.clone());
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
    #[inline]
    pub fn add(&mut self, v: T) {
        self.sorted = false;
        self.data.push(Partial(v));
    }

    /// Return the number of data points.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    fn sort(&mut self) {
        if !self.sorted {
            self.data.par_sort_unstable();
            self.sorted = true;
        }
    }

    #[inline]
    const fn already_sorted(&mut self) {
        self.sorted = true;
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
            let mut total = 0;
            for (i, &count) in chunk_results.iter().enumerate() {
                total += count;

                // Check boundary between chunks
                if i > 0 {
                    // safety: When i > 0:
                    // - (i * CHUNK_SIZE) - 1 is valid because it points to the last element of the previous chunk
                    // - i * CHUNK_SIZE is valid because it points to the first element of the current chunk
                    // These indices are guaranteed to be in bounds since we're iterating over chunk_results
                    // which was created from valid chunks of self.data
                    unsafe {
                        let prev_chunk_end = self.data.get_unchecked((i * CHUNK_SIZE) - 1);
                        let curr_chunk_start = self.data.get_unchecked(i * CHUNK_SIZE);
                        if prev_chunk_end == curr_chunk_start {
                            total -= 1;
                        }
                    }
                }
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
        mode_on_sorted(self.data.iter()).map(|p| p.0.clone())
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
    /// Returns the quartiles of the data.
    #[inline]
    pub fn quartiles(&mut self) -> Option<(f64, f64, f64)> {
        if self.data.is_empty() {
            return None;
        }
        self.sort();
        quartiles_on_sorted(&self.data)
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
}
