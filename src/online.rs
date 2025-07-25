use std::fmt;

use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::Commute;

/// Compute the standard deviation of a stream in constant space.
#[inline]
pub fn stddev<'a, I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: Into<&'a f64>,
{
    let it = x.into_iter();
    stddev(it)
}

/// Compute the variance of a stream in constant space.
#[inline]
pub fn variance<'a, I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: Into<&'a f64>,
{
    let it = x.into_iter();
    variance(it)
}

/// Compute the mean of a stream in constant space.
#[inline]
pub fn mean<'a, I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: Into<&'a f64>,
{
    let it = x.into_iter();
    mean(it)
}

/// Online state for computing mean, variance and standard deviation.
///
/// Optimized memory layout for better cache performance:
/// - 64-byte alignment for cache line efficiency
/// - Grouped related fields together
/// - Uses bit flags to reduce padding overhead
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
#[repr(C, align(64))]
pub struct OnlineStats {
    // Core statistics (40 bytes)
    size: u64,          // 8 bytes
    mean: f64,          // 8 bytes
    q: f64,             // 8 bytes
    harmonic_sum: f64,  // 8 bytes
    geometric_sum: f64, // 8 bytes

    // flags (2 bytes)
    has_zero: bool,     // 1 byte
    has_negative: bool, // 1 byte
    hg_sums: bool,      // 1 byte

    // Padding to reach 64 bytes for cache alignment
    _padding: [u8; 21], // 21 bytes padding
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
        if self.is_empty() || self.has_zero || self.has_negative {
            f64::NAN
        } else {
            (self.size as f64) / self.harmonic_sum
        }
    }

    /// Return the current geometric mean.
    #[must_use]
    pub fn geometric_mean(&self) -> f64 {
        if self.is_empty()
            || self.has_negative
            || self.geometric_sum.is_infinite()
            || self.geometric_sum.is_nan()
        {
            f64::NAN
        } else if self.has_zero {
            0.0
        } else {
            (self.geometric_sum / (self.size as f64)).exp()
        }
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
            return;
        }

        // Handle special cases (zero and negative numbers)
        if sample <= 0.0 {
            if sample.is_sign_negative() {
                self.has_negative = true;
            } else {
                self.has_zero = true;
            }
            self.hg_sums = !self.has_negative && !self.has_zero;
        }
    }

    /// Add a new f64 sample.
    /// Skipping the ToPrimitive conversion.
    #[inline]
    pub fn add_f64(&mut self, sample: f64) {
        self.size += 1;
        let delta = sample - self.mean;

        self.mean = delta.mul_add(1.0 / (self.size as f64), self.mean);
        self.q = delta.mul_add(sample - self.mean, self.q);

        if sample > 0.0 && self.hg_sums {
            self.harmonic_sum = (1.0 / sample).mul_add(1.0, self.harmonic_sum);
            self.geometric_sum = sample.ln().mul_add(1.0, self.geometric_sum);
            return;
        }

        if sample <= 0.0 {
            if sample.is_sign_negative() {
                self.has_negative = true;
            } else {
                self.has_zero = true;
            }
            self.hg_sums = !self.has_negative && !self.has_zero;
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

        self.has_zero |= v.has_zero;
        self.has_negative |= v.has_negative;
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
            has_zero: false,
            has_negative: false,
            hg_sums: true,
            _padding: [0; 21],
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
    use super::OnlineStats;
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
}
