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
    len: u64,
    sort_order: SortOrder,
    min: Option<T>,
    max: Option<T>,
    first_value: Option<T>, // Tracks the first value added
    last_value: Option<T>,  // Tracks the last value added
}

impl<T: PartialOrd + Clone> MinMax<T> {
    /// Create an empty state where min and max values do not exist.
    #[must_use]
    pub fn new() -> MinMax<T> {
        Default::default()
    }

    /// Add a sample to the data and update the sort order.
    #[inline]
    pub fn add(&mut self, sample: T) {
        self.len += 1;

        if self.len > 2 {
            // Third or more value, update sort order based on last value
            if let Some(ref last) = self.last_value {
                match self.sort_order {
                    SortOrder::Unsorted => {}
                    SortOrder::Ascending => {
                        if sample < *last {
                            self.sort_order = SortOrder::Unsorted;
                        }
                    }
                    SortOrder::Descending => {
                        if sample > *last {
                            self.sort_order = SortOrder::Unsorted;
                        }
                    }
                }
            }
            self.last_value = Some(sample.clone());
        } else if self.len == 1 {
            // First value added
            self.first_value = Some(sample.clone());
            self.last_value = Some(sample.clone());
            self.min = Some(sample.clone());
            self.max = Some(sample);
            self.sort_order = SortOrder::Unsorted;
            return;
        } else {
            // Second value (self.len == 2), determine initial sort order
            self.last_value = Some(sample.clone());
            if let Some(ref first) = self.first_value {
                self.sort_order = match sample.partial_cmp(first) {
                    Some(Ordering::Greater | Ordering::Equal) => SortOrder::Ascending,
                    Some(Ordering::Less) => SortOrder::Descending,
                    None => SortOrder::Unsorted,
                };
            }
        }

        if self.min.as_ref().map_or(true, |v| &sample < v) {
            self.min = Some(sample.clone());
        }
        if self.max.as_ref().map_or(true, |v| &sample > v) {
            self.max = Some(sample);
        }
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
}

impl<T: PartialOrd> Commute for MinMax<T> {
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

        // Handle merging of first_value and last_value
        if self.len > 1 && v.len > 0 {
            if self.first_value.is_none() {
                self.first_value = v.first_value;
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
}
