use std::default::Default;
use std::fmt;
use std::iter::{FromIterator, IntoIterator};

use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::Commute;

/// Compute the standard deviation of a stream in constant space.
pub fn stddev<'a, I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: Into<&'a f64>,
{
    let it = x.into_iter();
    stddev(it)
}

/// Compute the variance of a stream in constant space.
pub fn variance<'a, I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: Into<&'a f64>,
{
    let it = x.into_iter();
    variance(it)
}

/// Compute the mean of a stream in constant space.
pub fn mean<'a, I, T>(x: I) -> f64
where
    I: IntoIterator<Item = T>,
    T: Into<&'a f64>,
{
    let it = x.into_iter();
    mean(it)
}

/// Online state for computing mean, variance and standard deviation.
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OnlineStats {
    size: u64,
    mean: f64,
    q: f64,
}

impl OnlineStats {
    /// Create initial state.
    ///
    /// Population size, variance and mean are set to `0`.
    #[must_use]
    pub fn new() -> OnlineStats {
        Default::default()
    }

    /// Initializes variance from a sample.
    #[must_use]
    pub fn from_slice<T: ToPrimitive>(samples: &[T]) -> OnlineStats {
        samples
            .iter()
            .map(|n| unsafe { n.to_f64().unwrap_unchecked() })
            .collect()
    }

    /// Return the current mean.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Return the current standard deviation.
    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Return the current variance.
    // TODO: look into alternate algorithms for calculating variance
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.q / (self.size as f64)
    }

    // TODO: Calculate kurtosis
    // also see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    /// Add a new sample.
    #[inline]
    #[allow(clippy::needless_pass_by_value)]
    pub fn add<T: ToPrimitive>(&mut self, sample: T) {
        let sample = unsafe { sample.to_f64().unwrap_unchecked() };
        // Taken from: https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
        // See also: https://api.semanticscholar.org/CorpusID:120126049
        let oldmean = self.mean;
        self.size += 1;
        let delta = sample - oldmean;
        self.mean += delta / (self.size as f64);
        let delta2 = sample - self.mean;
        self.q += delta * delta2;
    }

    /// Add a new NULL value to the population.
    ///
    /// This increases the population size by `1`.
    #[inline]
    pub fn add_null(&mut self) {
        self.add(0usize);
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
        // Taken from: https://en.wikipedia.org/wiki/Standard_deviation#Combining_standard_deviations
        let (s1, s2) = (self.size as f64, v.size as f64);
        let meandiffsq = (self.mean - v.mean) * (self.mean - v.mean);

        self.size += v.size;

        //self.mean = ((s1 * self.mean) + (s2 * v.mean)) / (s1 + s2);
        /*
        below is the fused multiply add version of the statement above
        its more performant as we're taking advantage of a CPU instruction
        Note that fma appears to have issues on macOS per the flaky CI tests
        and it appears that clippy::suboptimal_flops lint that suggested
        this made a false-positive recommendation
        https://github.com/rust-lang/rust-clippy/issues/10003
        leaving on for now, as qsv is primarily optimized for Linux targets */
        self.mean = s1.mul_add(self.mean, s2 * v.mean) / (s1 + s2);

        self.q += v.q + meandiffsq * s1 * s2 / (s1 + s2);
    }
}

impl Default for OnlineStats {
    fn default() -> OnlineStats {
        OnlineStats {
            size: 0,
            mean: 0.0,
            q: 0.0,
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
            self.add(sample);
        }
    }
}

#[cfg(test)]
mod test {
    use super::OnlineStats;
    use {crate::merge_all, crate::Commute};

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
}
