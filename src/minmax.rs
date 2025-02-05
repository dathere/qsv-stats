use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

use crate::Commute;

/// Represents the current sort order of the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum SortOrder {
    Unsorted,
    Ascending,
    Descending,
}

/// A commutative data structure for tracking minimum and maximum values
/// and detecting sort order in a stream of data.
#[derive(Clone, Copy, Deserialize, Serialize, Eq, PartialEq)]
pub struct MinMax<T> {
    len: u32,
    sort_order: SortOrder,
    min: Option<T>,
    max: Option<T>,
    first_value: Option<T>, // Tracks the first value added
    last_value: Option<T>,  // Tracks the last value added
    ascending_pairs: u32,   // Track number of ascending pairs
    descending_pairs: u32,  // Track number of descending pairs
}

impl<T: PartialOrd + Clone> MinMax<T> {
    /// Create an empty state where min and max values do not exist.
    #[must_use]
    pub fn new() -> MinMax<T> {
        Default::default()
    }

    /// Add a sample to the data and track min/max, the sort order & "sortiness".
    #[inline]
    pub fn add(&mut self, sample: T) {
        match self.len {
            2.. => {
                // all samples after the second, update sort order & sortiness
                // Compare with last value to update sort order and pair counts
                // we have it as the first match arm for performance reasons
                if let Some(ref last) = self.last_value {
                    match sample.partial_cmp(last) {
                        Some(Ordering::Greater) => {
                            self.ascending_pairs += 1;
                            if self.sort_order == SortOrder::Descending {
                                self.sort_order = SortOrder::Unsorted;
                            }
                        }
                        Some(Ordering::Equal) => self.ascending_pairs += 1,
                        Some(Ordering::Less) => {
                            self.descending_pairs += 1;
                            if self.sort_order == SortOrder::Ascending {
                                self.sort_order = SortOrder::Unsorted;
                            }
                        }
                        None => self.sort_order = SortOrder::Unsorted,
                    }
                }
            }
            0 => {
                // first sample, initialize everything
                self.first_value = Some(sample.clone());
                self.min = Some(sample.clone());
                self.max = Some(sample);
                self.sort_order = SortOrder::Unsorted;
                self.len = 1;
                return;
            }
            1 => {
                // second sample, establish initial sort order
                if let Some(ref first) = self.first_value {
                    match sample.partial_cmp(first) {
                        Some(Ordering::Greater | Ordering::Equal) => {
                            self.ascending_pairs = 1;
                            self.sort_order = SortOrder::Ascending;
                        }
                        Some(Ordering::Less) => {
                            self.descending_pairs = 1;
                            self.sort_order = SortOrder::Descending;
                        }
                        None => self.sort_order = SortOrder::Unsorted,
                    }
                }
            }
        }

        // Update min/max
        if self.min.as_ref().is_none_or(|v| &sample < v) {
            self.min = Some(sample.clone());
        } else if self.max.as_ref().is_none_or(|v| &sample > v) {
            self.max = Some(sample.clone());
        }

        // Update last value and number of samples
        self.last_value = Some(sample);
        self.len += 1;
    }

    /// Returns the minimum of the data set.
    ///
    /// `None` is returned if and only if the number of samples is `0`.
    #[inline]
    #[must_use]
    pub const fn min(&self) -> Option<&T> {
        self.min.as_ref()
    }

    /// Returns the maximum of the data set.
    ///
    /// `None` is returned if and only if the number of samples is `0`.
    #[inline]
    #[must_use]
    pub const fn max(&self) -> Option<&T> {
        self.max.as_ref()
    }

    /// Returns the number of data points.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns true if there are no data points.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the current sort order of the data.
    #[inline]
    #[must_use]
    pub const fn sort_order(&self) -> SortOrder {
        self.sort_order
    }

    /// Calculates a "sortiness" score for the data, indicating how close it is to being sorted.
    ///
    /// Returns a value between -1.0 and 1.0:
    /// * 1.0 indicates perfectly ascending order
    /// * -1.0 indicates perfectly descending order
    /// * Values in between indicate the general tendency towards ascending or descending order
    /// * 0.0 indicates either no clear ordering or empty/single-element collections
    ///
    /// # Examples
    /// ```
    /// use stats::MinMax;
    ///
    /// let mut asc: MinMax<i32> = vec![1, 2, 3, 4, 5].into_iter().collect();
    /// assert_eq!(asc.sortiness(), 1.0);
    ///
    /// let mut desc: MinMax<i32> = vec![5, 4, 3, 2, 1].into_iter().collect();
    /// assert_eq!(desc.sortiness(), -1.0);
    ///
    /// let mut mostly_asc: MinMax<i32> = vec![1, 2, 4, 3, 5].into_iter().collect();
    /// assert!(mostly_asc.sortiness() > 0.0); // Positive but less than 1.0
    /// ```
    #[inline]
    #[must_use]
    pub fn sortiness(&self) -> f64 {
        match self.len {
            0 | 1 => 0.0,
            _ => {
                let total_pairs = self.ascending_pairs + self.descending_pairs;
                if total_pairs == 0 {
                    0.0
                } else {
                    (self.ascending_pairs as f64 - self.descending_pairs as f64)
                        / total_pairs as f64
                }
            }
        }
    }
}

impl<T: PartialOrd + Clone> Commute for MinMax<T> {
    #[inline]
    fn merge(&mut self, v: MinMax<T>) {
        self.len += v.len;
        if self.min.is_none() || (v.min.is_some() && v.min < self.min) {
            self.min = v.min;
        }
        if self.max.is_none() || (v.max.is_some() && v.max > self.max) {
            self.max = v.max;
        }

        // Merge sort order logic
        if self.sort_order == SortOrder::Unsorted
            || v.sort_order == SortOrder::Unsorted
            || self.sort_order != v.sort_order
        {
            self.sort_order = SortOrder::Unsorted;
        }

        // Merge pair counts
        self.ascending_pairs += v.ascending_pairs;
        self.descending_pairs += v.descending_pairs;

        // Handle merging of first_value and last_value
        if self.len > 1 && v.len > 0 {
            if self.first_value.is_none() {
                self.first_value.clone_from(&v.first_value);
            }
            // Add an additional pair count for the merge point
            if let (Some(ref last), Some(ref v_first)) = (&self.last_value, &v.first_value) {
                match v_first.partial_cmp(last) {
                    Some(Ordering::Greater | Ordering::Equal) => self.ascending_pairs += 1,
                    Some(Ordering::Less) => self.descending_pairs += 1,
                    None => {}
                }
            }
            self.last_value = v.last_value;
        }
    }
}

impl<T: PartialOrd> Default for MinMax<T> {
    #[inline]
    fn default() -> MinMax<T> {
        MinMax {
            len: 0,
            sort_order: SortOrder::Unsorted, // Start with Unsorted by default
            min: None,
            max: None,
            first_value: None,
            last_value: None,
            ascending_pairs: 0,
            descending_pairs: 0,
        }
    }
}

#[cfg(debug_assertions)]
impl<T: fmt::Debug> fmt::Debug for MinMax<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match (&self.min, &self.max) {
            (Some(min), Some(max)) => {
                write!(f, "[{min:?}, {max:?}], sort_order: {:?}", self.sort_order)
            }
            (&None, &None) => write!(f, "N/A"),
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for SortOrder {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SortOrder::Unsorted => write!(f, "Unsorted"),
            SortOrder::Ascending => write!(f, "Ascending"),
            SortOrder::Descending => write!(f, "Descending"),
        }
    }
}

impl<T: PartialOrd + Clone> FromIterator<T> for MinMax<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> MinMax<T> {
        let mut v = MinMax::new();
        v.extend(it);
        v
    }
}

impl<T: PartialOrd + Clone> Extend<T> for MinMax<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, it: I) {
        for sample in it {
            self.add(sample);
        }
    }
}

#[cfg(test)]
mod test {
    use super::{MinMax, SortOrder};
    use crate::Commute;

    #[test]
    fn minmax() {
        let minmax: MinMax<u32> = vec![1u32, 4, 2, 3, 10].into_iter().collect();
        assert_eq!(minmax.min(), Some(&1u32));
        assert_eq!(minmax.max(), Some(&10u32));
        assert_eq!(minmax.sort_order(), SortOrder::Unsorted);
    }

    #[test]
    fn minmax_sorted_ascending() {
        let minmax: MinMax<u32> = vec![1u32, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(minmax.min(), Some(&1u32));
        assert_eq!(minmax.max(), Some(&5u32));
        assert_eq!(minmax.sort_order(), SortOrder::Ascending);
    }

    #[test]
    fn minmax_sorted_descending() {
        let minmax: MinMax<u32> = vec![5u32, 4, 3, 2, 1].into_iter().collect();
        assert_eq!(minmax.min(), Some(&1u32));
        assert_eq!(minmax.max(), Some(&5u32));
        assert_eq!(minmax.sort_order(), SortOrder::Descending);
    }

    #[test]
    fn minmax_empty() {
        let minmax: MinMax<u32> = MinMax::new();
        assert!(minmax.is_empty());
        assert_eq!(minmax.sort_order(), SortOrder::Unsorted);
    }

    #[test]
    fn minmax_merge_empty() {
        let mut mx1: MinMax<u32> = vec![1, 4, 2, 3, 10].into_iter().collect();
        assert_eq!(mx1.min(), Some(&1u32));
        assert_eq!(mx1.max(), Some(&10u32));
        assert_eq!(mx1.sort_order(), SortOrder::Unsorted);

        mx1.merge(MinMax::default());
        assert_eq!(mx1.min(), Some(&1u32));
        assert_eq!(mx1.max(), Some(&10u32));
        assert_eq!(mx1.sort_order(), SortOrder::Unsorted);
    }

    #[test]
    fn minmax_merge_diffsorts() {
        let mut mx1: MinMax<u32> = vec![1, 2, 2, 2, 3, 3, 4, 10].into_iter().collect();
        assert_eq!(mx1.min(), Some(&1u32));
        assert_eq!(mx1.max(), Some(&10u32));
        assert_eq!(mx1.sort_order(), SortOrder::Ascending);

        let mx2: MinMax<u32> = vec![5, 4, 3, 2, 1].into_iter().collect();
        assert_eq!(mx2.min(), Some(&1u32));
        assert_eq!(mx2.max(), Some(&5u32));
        assert_eq!(mx2.sort_order(), SortOrder::Descending);
        mx1.merge(mx2);
        assert_eq!(mx1.min(), Some(&1u32));
        assert_eq!(mx1.max(), Some(&10u32));
        assert_eq!(mx1.sort_order(), SortOrder::Unsorted);
    }

    #[test]
    fn minmax_merge_asc_sorts() {
        let mut mx1: MinMax<u32> = vec![2, 2, 2, 5, 10].into_iter().collect();
        assert_eq!(mx1.min(), Some(&2u32));
        assert_eq!(mx1.max(), Some(&10u32));
        assert_eq!(mx1.sort_order(), SortOrder::Ascending);

        let mx2: MinMax<u32> = vec![11, 14, 23, 32, 41].into_iter().collect();
        assert_eq!(mx2.min(), Some(&11u32));
        assert_eq!(mx2.max(), Some(&41u32));
        assert_eq!(mx2.sort_order(), SortOrder::Ascending);
        mx1.merge(mx2);
        assert_eq!(mx1.min(), Some(&2u32));
        assert_eq!(mx1.max(), Some(&41u32));
        assert_eq!(mx1.sort_order(), SortOrder::Ascending);
    }

    #[test]
    fn test_sortiness() {
        // Test empty
        let minmax: MinMax<u32> = MinMax::new();
        assert_eq!(minmax.sortiness(), 0.0);

        // Test single element
        let minmax: MinMax<u32> = vec![1].into_iter().collect();
        assert_eq!(minmax.sortiness(), 0.0);

        // Test perfectly ascending
        let minmax: MinMax<u32> = vec![1, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(minmax.sortiness(), 1.0);

        // Test perfectly descending
        let minmax: MinMax<u32> = vec![5, 4, 3, 2, 1].into_iter().collect();
        assert_eq!(minmax.sortiness(), -1.0);

        // Test all equal
        let minmax: MinMax<u32> = vec![1, 1, 1, 1].into_iter().collect();
        assert_eq!(minmax.sortiness(), 1.0); // Equal pairs are considered ascending

        // Test mostly ascending
        let minmax: MinMax<u32> = vec![1, 2, 4, 3, 5].into_iter().collect();
        assert!(minmax.sortiness() > 0.0 && minmax.sortiness() < 1.0);
        assert_eq!(minmax.sortiness(), 0.5); // 2 ascending pairs, 1 descending pair

        // Test mostly descending
        let minmax: MinMax<u32> = vec![5, 4, 3, 4, 2].into_iter().collect();
        assert!(minmax.sortiness() < 0.0 && minmax.sortiness() > -1.0);
        assert_eq!(minmax.sortiness(), -0.5); // 1 ascending pair, 3 descending pairs
    }

    #[test]
    fn test_sortiness_merge() {
        let mut mx1: MinMax<u32> = vec![1, 2, 3].into_iter().collect();
        let mx2: MinMax<u32> = vec![4, 5, 6].into_iter().collect();
        assert_eq!(mx1.sortiness(), 1.0);
        assert_eq!(mx2.sortiness(), 1.0);

        mx1.merge(mx2);
        assert_eq!(mx1.sortiness(), 1.0); // Should remain perfectly sorted after merge

        let mut mx3: MinMax<u32> = vec![1, 2, 3].into_iter().collect();
        let mx4: MinMax<u32> = vec![2, 1, 0].into_iter().collect();
        mx3.merge(mx4);
        assert_eq!(mx3, vec![1, 2, 3, 2, 1, 0].into_iter().collect());
        assert!(mx3.sortiness() < 1.0); // Should show mixed sorting after merge
        assert_eq!(mx3.sortiness(), -0.2);
    }
}
