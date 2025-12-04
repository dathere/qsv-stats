use std::fmt;

use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::Commute;

/// Compute the standard deviation of a stream in constant space.
#[inline]
pub fn stddev<I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: ToPrimitive,
{
    x.into_iter().collect::<OnlineStats>().stddev()
}

/// Compute the variance of a stream in constant space.
#[inline]
pub fn variance<I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: ToPrimitive,
{
    x.into_iter().collect::<OnlineStats>().variance()
}

/// Compute the mean of a stream in constant space.
#[inline]
pub fn mean<I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: ToPrimitive,
{
    x.into_iter().collect::<OnlineStats>().mean()
}

/// Online state for computing mean, variance and standard deviation.
///
/// Optimized memory layout for better cache performance:
/// - Grouped related fields together in hot, warm and cold paths.
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OnlineStats {
    // Hot path - always accessed together (24 bytes)
    size: u64, // 8 bytes - always accessed
    mean: f64, // 8 bytes - always accessed
    q: f64,    // 8 bytes - always accessed

    // Warm path - fast path for positive numbers (25 bytes)
    hg_sums: bool,      // 1 byte - checked before sums
    harmonic_sum: f64,  // 8 bytes - warm path
    geometric_sum: f64, // 8 bytes - warm path
    n_positive: u64,    // 8 bytes - warm path

    // Cold path - slow path for zeros/negatives (16 bytes)
    n_zero: u64,     // 8 bytes - cold path
    n_negative: u64, // 8 bytes - cold path
}

impl OnlineStats {
    /// Create initial state.
    ///
    /// Population size, variance and mean are set to `0`.
    #[must_use]
    pub fn new() -> OnlineStats {
        Default::default()
    }

    /// Initializes `OnlineStats` from a sample.
    #[must_use]
    pub fn from_slice<T: ToPrimitive>(samples: &[T]) -> OnlineStats {
        // safety: OnlineStats is only for numbers
        samples
            .iter()
            .map(|n| unsafe { n.to_f64().unwrap_unchecked() })
            .collect()
    }

    /// Return the current mean.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        if self.is_empty() { f64::NAN } else { self.mean }
    }

    /// Return the current standard deviation.
    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Return the current variance.
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.q / (self.size as f64)
    }

    /// Return the current harmonic mean.
    #[must_use]
    pub fn harmonic_mean(&self) -> f64 {
        if self.is_empty() || self.n_zero > 0 || self.n_negative > 0 {
            f64::NAN
        } else {
            (self.size as f64) / self.harmonic_sum
        }
    }

    /// Return the current geometric mean.
    #[must_use]
    pub fn geometric_mean(&self) -> f64 {
        if self.is_empty()
            || self.n_negative > 0
            || self.geometric_sum.is_infinite()
            || self.geometric_sum.is_nan()
        {
            f64::NAN
        } else if self.n_zero > 0 {
            0.0
        } else {
            (self.geometric_sum / (self.size as f64)).exp()
        }
    }

    /// Return the number of negative, zero and positive counts.
    ///
    /// Returns a tuple `(negative_count, zero_count, positive_count)` where:
    /// - `negative_count`: number of values less than 0
    /// - `zero_count`: number of values equal to 0 (including +0.0 but not -0.0)
    /// - `positive_count`: number of values greater than 0
    ///
    /// # Example
    ///
    /// ```
    /// use stats::OnlineStats;
    ///
    /// let mut stats = OnlineStats::new();
    /// stats.extend(vec![-2, -1, 0, 0, 1, 2, 3]);
    ///
    /// let (neg, zero, pos) = stats.n_counts();
    /// assert_eq!(neg, 2);   // -2, -1
    /// assert_eq!(zero, 2);  // 0, 0
    /// assert_eq!(pos, 3);   // 1, 2, 3
    /// ```
    #[must_use]
    pub const fn n_counts(&self) -> (u64, u64, u64) {
        (self.n_negative, self.n_zero, self.n_positive)
    }

    // TODO: Calculate kurtosis
    // also see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    /// Add a new sample.
    #[inline]
    pub fn add<T: ToPrimitive>(&mut self, sample: &T) {
        // safety: we only add samples for numbers, so safe to unwrap
        let sample = unsafe { sample.to_f64().unwrap_unchecked() };

        // Taken from: https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
        // See also: https://api.semanticscholar.org/CorpusID:120126049
        self.size += 1;
        let delta = sample - self.mean;

        // FMA: equivalent to: self.mean += delta * (1.0 / (self.size as f64));
        self.mean = delta.mul_add(1.0 / (self.size as f64), self.mean);

        // FMA: equivalent to: self.q += delta * (sample - self.mean);
        self.q = delta.mul_add(sample - self.mean, self.q);

        // Optimized path for positive numbers (most common case)
        if sample > 0.0 && self.hg_sums {
            // Fast path: compute harmonic & geometric sums directly
            // use FMA. equivalent to: self.harmonic_sum += 1.0 / sample
            self.harmonic_sum = (1.0 / sample).mul_add(1.0, self.harmonic_sum);
            // use FMA. equivalent to: self.geometric_sum += ln(sample)
            self.geometric_sum = sample.ln().mul_add(1.0, self.geometric_sum);
            self.n_positive += 1;
            return;
        }

        // Handle special cases (zero and negative numbers)
        if sample <= 0.0 {
            if sample.is_sign_negative() {
                self.n_negative += 1;
            } else {
                self.n_zero += 1;
            }
            self.hg_sums = self.n_negative == 0 && self.n_zero == 0;
        } else {
            self.n_positive += 1;
        }
    }

    /// Add a new f64 sample.
    /// Skipping the `ToPrimitive` conversion.
    #[inline]
    pub fn add_f64(&mut self, sample: f64) {
        self.size += 1;
        let delta = sample - self.mean;

        self.mean = delta.mul_add(1.0 / (self.size as f64), self.mean);
        self.q = delta.mul_add(sample - self.mean, self.q);

        if sample > 0.0 && self.hg_sums {
            self.harmonic_sum = (1.0 / sample).mul_add(1.0, self.harmonic_sum);
            self.geometric_sum = sample.ln().mul_add(1.0, self.geometric_sum);
            self.n_positive += 1;
            return;
        }

        if sample <= 0.0 {
            if sample.is_sign_negative() {
                self.n_negative += 1;
            } else {
                self.n_zero += 1;
            }
            self.hg_sums = self.n_negative == 0 && self.n_zero == 0;
        } else {
            self.n_positive += 1;
        }
    }

    /// Add a new NULL value to the population.
    /// This increases the population size by `1`.
    #[inline]
    pub fn add_null(&mut self) {
        self.add_f64(0.0);
    }

    /// Returns the number of data points.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.size as usize
    }

    /// Returns if empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Commute for OnlineStats {
    #[inline]
    fn merge(&mut self, v: OnlineStats) {
        if v.is_empty() {
            return;
        }

        // Taken from: https://en.wikipedia.org/wiki/Standard_deviation#Combining_standard_deviations
        let (s1, s2) = (self.size as f64, v.size as f64);
        let total = s1 + s2;
        let meandiffsq = (self.mean - v.mean) * (self.mean - v.mean);

        self.size += v.size;

        //self.mean = ((s1 * self.mean) + (s2 * v.mean)) / (s1 + s2);
        // below is the fused multiply add version of the statement above
        // its more performant as we're taking advantage of a CPU instruction
        self.mean = s1.mul_add(self.mean, s2 * v.mean) / total;

        // self.q += v.q + meandiffsq * s1 * s2 / (s1 + s2);
        // below is the fused multiply add version of the statement above
        self.q += v.q + f64::mul_add(meandiffsq, s1 * s2 / total, 0.0);

        self.harmonic_sum += v.harmonic_sum;
        self.geometric_sum += v.geometric_sum;

        self.n_zero += v.n_zero;
        self.n_negative += v.n_negative;
        self.n_positive += v.n_positive;
    }
}

impl Default for OnlineStats {
    fn default() -> OnlineStats {
        OnlineStats {
            size: 0,
            mean: 0.0,
            q: 0.0,
            harmonic_sum: 0.0,
            geometric_sum: 0.0,
            n_zero: 0,
            n_negative: 0,
            n_positive: 0,
            hg_sums: true,
        }
    }
}

impl fmt::Debug for OnlineStats {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.10} +/- {:.10}", self.mean(), self.stddev())
    }
}

impl<T: ToPrimitive> FromIterator<T> for OnlineStats {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> OnlineStats {
        let mut v = OnlineStats::new();
        v.extend(it);
        v
    }
}

impl<T: ToPrimitive> Extend<T> for OnlineStats {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, it: I) {
        for sample in it {
            self.add(&sample);
        }
    }
}

#[cfg(test)]
mod test {
    use super::{OnlineStats, mean, stddev, variance};
    use {crate::Commute, crate::merge_all};

    #[test]
    fn online() {
        // TODO: Convert this to a quickcheck test.
        let expected = OnlineStats::from_slice(&[1usize, 2, 3, 2, 4, 6]);

        let var1 = OnlineStats::from_slice(&[1usize, 2, 3]);
        let var2 = OnlineStats::from_slice(&[2usize, 4, 6]);
        let mut got = var1;
        got.merge(var2);
        assert_eq!(expected.stddev(), got.stddev());
        assert_eq!(expected.mean(), got.mean());
        assert_eq!(expected.variance(), got.variance());
    }

    #[test]
    fn online_empty() {
        let expected = OnlineStats::new();
        assert!(expected.is_empty());
    }

    #[test]
    fn online_many() {
        // TODO: Convert this to a quickcheck test.
        let expected = OnlineStats::from_slice(&[1usize, 2, 3, 2, 4, 6, 3, 6, 9]);

        let vars = vec![
            OnlineStats::from_slice(&[1usize, 2, 3]),
            OnlineStats::from_slice(&[2usize, 4, 6]),
            OnlineStats::from_slice(&[3usize, 6, 9]),
        ];
        assert_eq!(
            expected.stddev(),
            merge_all(vars.clone().into_iter()).unwrap().stddev()
        );
        assert_eq!(
            expected.mean(),
            merge_all(vars.clone().into_iter()).unwrap().mean()
        );
        assert_eq!(
            expected.variance(),
            merge_all(vars.into_iter()).unwrap().variance()
        );
    }

    #[test]
    fn test_means() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![2.0f64, 4.0, 8.0]);

        // Arithmetic mean = (2 + 4 + 8) / 3 = 4.666...
        assert!((stats.mean() - 4.666666666667).abs() < 1e-10);

        // Harmonic mean = 3 / (1/2 + 1/4 + 1/8) = 3.428571429
        assert_eq!("3.42857143", format!("{:.8}", stats.harmonic_mean()));

        // Geometric mean = (2 * 4 * 8)^(1/3) = 4.0
        assert!((stats.geometric_mean() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_means_with_negative() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![-2.0f64, 2.0]);

        // Arithmetic mean = (-2 + 2) / 2 = 0
        assert!(stats.mean().abs() < 1e-10);

        // Geometric mean is NaN for negative numbers
        assert!(stats.geometric_mean().is_nan());

        // Harmonic mean is undefined when values have different signs
        assert!(stats.harmonic_mean().is_nan());
    }

    #[test]
    fn test_means_with_zero() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![0.0f64, 4.0, 8.0]);

        // Arithmetic mean = (0 + 4 + 8) / 3 = 4
        assert!((stats.mean() - 4.0).abs() < 1e-10);

        // Geometric mean = (0 * 4 * 8)^(1/3) = 0
        assert!(stats.geometric_mean().abs() < 1e-10);

        // Harmonic mean is undefined when any value is 0
        assert!(stats.harmonic_mean().is_nan());
    }

    #[test]
    fn test_means_with_zero_and_negative_values() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![-10i32, -5, 0, 5, 10]);

        // Arithmetic mean = (-10 + -5 + 0 + 5 + 10) / 5 = 0
        assert!(stats.mean().abs() < 1e-10);

        // Geometric mean is NaN due to negative values
        assert!(stats.geometric_mean().is_nan());

        // Harmonic mean is NaN due to zero value
        assert!(stats.harmonic_mean().is_nan());
    }

    #[test]
    fn test_means_single_value() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![5.0f64]);

        // All means should equal the single value
        assert!((stats.mean() - 5.0).abs() < 1e-10);
        assert!((stats.geometric_mean() - 5.0).abs() < 1e-10);
        assert!((stats.harmonic_mean() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_means_empty() {
        let stats = OnlineStats::new();

        // All means should be NaN for empty stats
        assert!(stats.mean().is_nan());
        assert!(stats.geometric_mean().is_nan());
        assert!(stats.harmonic_mean().is_nan());
    }

    // Tests for wrapper functions: stddev(), variance(), mean()

    #[test]
    fn test_mean_wrapper_basic() {
        // Test with f64 values
        let result = mean(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        assert!((result - 3.0).abs() < 1e-10);

        // Test with i32 values
        let result = mean(vec![1i32, 2, 3, 4, 5]);
        assert!((result - 3.0).abs() < 1e-10);

        // Test with u32 values
        let result = mean(vec![10u32, 20, 30]);
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_wrapper_empty() {
        let result = mean(Vec::<f64>::new());
        assert!(result.is_nan());
    }

    #[test]
    fn test_mean_wrapper_single_element() {
        assert!((mean(vec![42.0f64]) - 42.0).abs() < 1e-10);
        assert!((mean(vec![100i32]) - 100.0).abs() < 1e-10);
        assert!((mean(vec![0u8]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_wrapper_negative_values() {
        let result = mean(vec![-5.0f64, 5.0]);
        assert!(result.abs() < 1e-10); // Mean should be 0

        let result = mean(vec![-10i32, -20, -30]);
        assert!((result - (-20.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mean_wrapper_various_numeric_types() {
        // Test with different numeric types
        assert!((mean(vec![1u8, 2, 3]) - 2.0).abs() < 1e-10);
        assert!((mean(vec![1u16, 2, 3]) - 2.0).abs() < 1e-10);
        assert!((mean(vec![1u64, 2, 3]) - 2.0).abs() < 1e-10);
        assert!((mean(vec![1i8, 2, 3]) - 2.0).abs() < 1e-10);
        assert!((mean(vec![1i16, 2, 3]) - 2.0).abs() < 1e-10);
        assert!((mean(vec![1i64, 2, 3]) - 2.0).abs() < 1e-10);
        assert!((mean(vec![1.0f32, 2.0, 3.0]) - 2.0).abs() < 1e-6);
        assert!((mean(vec![1usize, 2, 3]) - 2.0).abs() < 1e-10);
        assert!((mean(vec![1isize, 2, 3]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_wrapper_basic() {
        // Variance of [1, 2, 3, 4, 5] = 2.0 (population variance)
        let result = variance(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        assert!((result - 2.0).abs() < 1e-10);

        // Test with i32 values
        let result = variance(vec![1i32, 2, 3, 4, 5]);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_wrapper_empty() {
        let result = variance(Vec::<f64>::new());
        assert!(result.is_nan());
    }

    #[test]
    fn test_variance_wrapper_single_element() {
        // Variance of a single element is 0
        assert!(variance(vec![42.0f64]).abs() < 1e-10);
        assert!(variance(vec![100i32]).abs() < 1e-10);
    }

    #[test]
    fn test_variance_wrapper_identical_values() {
        // Variance of identical values is 0
        let result = variance(vec![5.0f64, 5.0, 5.0, 5.0]);
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_variance_wrapper_various_numeric_types() {
        // Test with different numeric types - variance of [1, 2, 3] = 2/3
        let expected = 2.0 / 3.0;
        assert!((variance(vec![1u8, 2, 3]) - expected).abs() < 1e-10);
        assert!((variance(vec![1u16, 2, 3]) - expected).abs() < 1e-10);
        assert!((variance(vec![1i32, 2, 3]) - expected).abs() < 1e-10);
        assert!((variance(vec![1i64, 2, 3]) - expected).abs() < 1e-10);
        assert!((variance(vec![1usize, 2, 3]) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_stddev_wrapper_basic() {
        // Standard deviation of [1, 2, 3, 4, 5] = sqrt(2.0)
        let result = stddev(vec![1.0f64, 2.0, 3.0, 4.0, 5.0]);
        assert!((result - 2.0f64.sqrt()).abs() < 1e-10);

        // Test with i32 values
        let result = stddev(vec![1i32, 2, 3, 4, 5]);
        assert!((result - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_stddev_wrapper_empty() {
        let result = stddev(Vec::<f64>::new());
        assert!(result.is_nan());
    }

    #[test]
    fn test_stddev_wrapper_single_element() {
        // Standard deviation of a single element is 0
        assert!(stddev(vec![42.0f64]).abs() < 1e-10);
        assert!(stddev(vec![100i32]).abs() < 1e-10);
    }

    #[test]
    fn test_stddev_wrapper_identical_values() {
        // Standard deviation of identical values is 0
        let result = stddev(vec![5.0f64, 5.0, 5.0, 5.0]);
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_stddev_wrapper_various_numeric_types() {
        // Test with different numeric types - stddev of [1, 2, 3] = sqrt(2/3)
        let expected = (2.0f64 / 3.0).sqrt();
        assert!((stddev(vec![1u8, 2, 3]) - expected).abs() < 1e-10);
        assert!((stddev(vec![1u16, 2, 3]) - expected).abs() < 1e-10);
        assert!((stddev(vec![1i32, 2, 3]) - expected).abs() < 1e-10);
        assert!((stddev(vec![1i64, 2, 3]) - expected).abs() < 1e-10);
        assert!((stddev(vec![1usize, 2, 3]) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_wrapper_functions_consistency() {
        // Verify that wrapper functions produce same results as OnlineStats methods
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = OnlineStats::from_slice(&data);

        assert!((mean(data.clone()) - stats.mean()).abs() < 1e-10);
        assert!((variance(data.clone()) - stats.variance()).abs() < 1e-10);
        assert!((stddev(data) - stats.stddev()).abs() < 1e-10);
    }

    #[test]
    fn test_wrapper_functions_with_iterators() {
        // Test that wrappers work with various iterator types
        let arr = [1, 2, 3, 4, 5];

        // Array iterator
        assert!((mean(arr) - 3.0).abs() < 1e-10);

        // Range iterator
        assert!((mean(1..=5) - 3.0).abs() < 1e-10);

        // Mapped iterator
        let result = mean((1..=5).map(|x| x * 2));
        assert!((result - 6.0).abs() < 1e-10);
    }

    // Tests for n_counts functionality

    #[test]
    fn test_n_counts_basic() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![-5, -3, 0, 0, 2, 4, 6]);

        let (neg, zero, pos) = stats.n_counts();
        assert_eq!(neg, 2, "Should have 2 negative values");
        assert_eq!(zero, 2, "Should have 2 zero values");
        assert_eq!(pos, 3, "Should have 3 positive values");
    }

    #[test]
    fn test_n_counts_all_positive() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![1.0, 2.0, 3.0, 4.0]);

        let (neg, zero, pos) = stats.n_counts();
        assert_eq!(neg, 0);
        assert_eq!(zero, 0);
        assert_eq!(pos, 4);
    }

    #[test]
    fn test_n_counts_all_negative() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![-1.0, -2.0, -3.0]);

        let (neg, zero, pos) = stats.n_counts();
        assert_eq!(neg, 3);
        assert_eq!(zero, 0);
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_n_counts_all_zeros() {
        let mut stats = OnlineStats::new();
        stats.extend(vec![0.0, 0.0, 0.0]);

        let (neg, zero, pos) = stats.n_counts();
        assert_eq!(neg, 0);
        assert_eq!(zero, 3);
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_n_counts_with_merge() {
        let mut stats1 = OnlineStats::new();
        stats1.extend(vec![-2, 0, 3]);

        let mut stats2 = OnlineStats::new();
        stats2.extend(vec![-1, 5, 7]);

        stats1.merge(stats2);

        let (neg, zero, pos) = stats1.n_counts();
        assert_eq!(neg, 2, "Should have 2 negative values after merge");
        assert_eq!(zero, 1, "Should have 1 zero value after merge");
        assert_eq!(pos, 3, "Should have 3 positive values after merge");
    }

    #[test]
    fn test_n_counts_empty() {
        let stats = OnlineStats::new();

        let (neg, zero, pos) = stats.n_counts();
        assert_eq!(neg, 0);
        assert_eq!(zero, 0);
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_n_counts_negative_zero() {
        let mut stats = OnlineStats::new();
        // -0.0 is counted as negative per IEEE 754 (has negative sign bit)
        // +0.0 is counted as zero
        stats.extend(vec![-0.0f64, 0.0]);

        let (neg, zero, pos) = stats.n_counts();
        assert_eq!(neg, 1, "-0.0 has negative sign bit");
        assert_eq!(zero, 1, "+0.0 is zero");
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_n_counts_floats_boundary() {
        let mut stats = OnlineStats::new();
        // Test with very small positive and negative numbers
        stats.extend(vec![-0.0001f64, 0.0, 0.0001]);

        let (neg, zero, pos) = stats.n_counts();
        assert_eq!(neg, 1);
        assert_eq!(zero, 1);
        assert_eq!(pos, 1);
    }
}
