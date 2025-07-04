use foldhash::{HashMap, HashMapExt};
use std::collections::hash_map::{Entry, Keys};
use std::hash::Hash;

use rayon::prelude::*;

use crate::Commute;
/// A commutative data structure for exact frequency counts.
#[derive(Clone)]
pub struct Frequencies<T> {
    data: HashMap<T, u64>,
}

#[cfg(debug_assertions)]
impl<T: std::fmt::Debug + Eq + Hash> std::fmt::Debug for Frequencies<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl<T: Eq + Hash> Frequencies<T> {
    /// Create a new frequency table with no samples.
    #[must_use]
    pub fn new() -> Frequencies<T> {
        Default::default()
    }

    // Add constructor with configurable capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Frequencies {
            data: HashMap::with_capacity(capacity),
        }
    }

    /// Add a value to the frequency table.
    #[inline]
    pub fn add(&mut self, v: T) {
        *self.data.entry(v).or_insert(0) += 1;
    }

    /// Return the number of occurrences of `v` in the data.
    #[inline]
    #[must_use]
    pub fn count(&self, v: &T) -> u64 {
        self.data.get(v).copied().unwrap_or(0)
    }

    /// Return the cardinality (number of unique elements) in the data.
    #[inline]
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        self.len() as u64
    }

    /// Returns the mode if one exists.
    /// Note that there is also a `modes()` function that returns all
    /// modes (plural) in `unsorted::modes()`. It returns all modes
    /// with the same frequency.
    #[inline]
    #[must_use]
    pub fn mode(&self) -> Option<&T> {
        let (counts, _) = self.most_frequent();
        if counts.is_empty() {
            return None;
        }
        // If there is a tie for the most frequent element, return None.
        if counts.len() >= 2 && counts[0].1 == counts[1].1 {
            None
        } else {
            Some(counts[0].0)
        }
    }

    /// Return a `Vec` of elements, their corresponding counts in
    /// descending order, and the total count.
    #[inline]
    #[must_use]
    pub fn most_frequent(&self) -> (Vec<(&T, u64)>, u64) {
        let len = self.data.len();
        let mut counts = Vec::with_capacity(len);
        let mut total_count = 0_u64;

        for (k, &v) in &self.data {
            total_count += v;
            counts.push((k, v));
        }
        counts.sort_unstable_by(|&(_, c1), &(_, c2)| c2.cmp(&c1));
        (counts, total_count)
    }

    /// Return a `Vec` of elements, their corresponding counts in
    /// ascending order, and the total count.
    #[inline]
    #[must_use]
    pub fn least_frequent(&self) -> (Vec<(&T, u64)>, u64) {
        let mut total_count = 0_u64;
        let mut counts: Vec<_> = self
            .data
            .iter()
            .map(|(k, &v)| {
                total_count += v;
                (k, v)
            })
            .collect();
        counts.sort_unstable_by(|&(_, c1), &(_, c2)| c1.cmp(&c2));
        (counts, total_count)
    }

    /// Return a `Vec` of elements, their corresponding counts in order
    /// based on the `least` parameter, and the total count. Uses parallel sort.
    #[inline]
    #[must_use]
    pub fn par_frequent(&self, least: bool) -> (Vec<(&T, u64)>, u64)
    where
        for<'a> (&'a T, u64): Send,
        T: Ord,
    {
        let mut total_count = 0_u64;
        let mut counts: Vec<_> = self
            .data
            .iter()
            .map(|(k, &v)| {
                total_count += v;
                (k, v)
            })
            .collect();
        // sort by counts asc/desc
        // if counts are equal, sort by values lexicographically
        // We need to do this because otherwise the values are not guaranteed to be in order for equal counts
        if least {
            // return counts in ascending order
            counts.par_sort_unstable_by(|&(v1, c1), &(v2, c2)| {
                let cmp = c1.cmp(&c2);
                if cmp == std::cmp::Ordering::Equal {
                    v1.cmp(v2)
                } else {
                    cmp
                }
            });
        } else {
            // return counts in descending order
            counts
                .par_sort_unstable_by(|&(v1, c1), &(v2, c2)| c2.cmp(&c1).then_with(|| v1.cmp(v2)));
        }
        (counts, total_count)
    }

    /// Returns the cardinality of the data.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if there is no frequency/cardinality data.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return an iterator over the unique values of the data.
    #[must_use]
    pub fn unique_values(&self) -> UniqueValues<'_, T> {
        UniqueValues {
            data_keys: self.data.keys(),
        }
    }

    /// Get the top N most frequent items without sorting the entire vector
    /// This is much faster than `most_frequent()` when you only need a few items
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<(&T, u64)>
    where
        T: Ord,
    {
        use std::collections::BinaryHeap;

        // We use a min-heap of size n to keep track of the largest elements
        let mut heap = BinaryHeap::with_capacity(n + 1);

        for (item, count) in &self.data {
            // Negate count because BinaryHeap is a max-heap
            // and we want to remove smallest elements
            heap.push(std::cmp::Reverse((*count, item)));

            // Keep heap size at n
            if heap.len() > n {
                heap.pop();
            }
        }

        // Convert to sorted vector
        heap.into_sorted_vec()
            .into_iter()
            .map(|std::cmp::Reverse((count, item))| (item, count))
            .collect()
    }

    /// Similar to `top_n` but for least frequent items
    #[must_use]
    pub fn bottom_n(&self, n: usize) -> Vec<(&T, u64)>
    where
        T: Ord,
    {
        use std::collections::BinaryHeap;

        let mut heap = BinaryHeap::with_capacity(n + 1);

        for (item, count) in &self.data {
            heap.push((*count, item));
            if heap.len() > n {
                heap.pop();
            }
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|(count, item)| (item, count))
            .collect()
    }

    /// Get items with exactly n occurrences
    #[must_use]
    pub fn items_with_count(&self, n: u64) -> Vec<&T> {
        self.data
            .iter()
            .filter(|&(_, &count)| count == n)
            .map(|(item, _)| item)
            .collect()
    }

    /// Get the sum of all counts
    #[must_use]
    pub fn total_count(&self) -> u64 {
        self.data.values().sum()
    }

    /// Check if any item occurs exactly n times
    #[must_use]
    pub fn has_count(&self, n: u64) -> bool {
        self.data.values().any(|&count| count == n)
    }

    /// Add specialized method for single increment
    #[inline]
    pub fn increment_by(&mut self, v: T, count: u64) {
        match self.data.entry(v) {
            Entry::Vacant(entry) => {
                entry.insert(count);
            }
            Entry::Occupied(mut entry) => {
                *entry.get_mut() += count;
            }
        }
    }
}

impl<T: Eq + Hash> Commute for Frequencies<T> {
    #[inline]
    fn merge(&mut self, v: Frequencies<T>) {
        // Reserve additional capacity to avoid reallocations
        self.data.reserve(v.data.len());

        for (k, v2) in v.data {
            match self.data.entry(k) {
                Entry::Vacant(v1) => {
                    v1.insert(v2);
                }
                Entry::Occupied(mut v1) => {
                    *v1.get_mut() += v2;
                }
            }
        }
    }
}

impl<T: Eq + Hash> Default for Frequencies<T> {
    #[inline]
    fn default() -> Frequencies<T> {
        Frequencies {
            data: HashMap::with_capacity(1_000),
        }
    }
}

impl<T: Eq + Hash> FromIterator<T> for Frequencies<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Frequencies<T> {
        let mut v = Frequencies::new();
        v.extend(it);
        v
    }
}

impl<T: Eq + Hash> Extend<T> for Frequencies<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, it: I) {
        for sample in it {
            self.add(sample);
        }
    }
}

/// An iterator over unique values in a frequencies count.
pub struct UniqueValues<'a, K> {
    data_keys: Keys<'a, K, u64>,
}

impl<'a, K> Iterator for UniqueValues<'a, K> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.data_keys.next()
    }
}

#[cfg(test)]
mod test {
    use super::Frequencies;
    use std::iter::FromIterator;

    #[test]
    fn ranked() {
        let mut counts = Frequencies::new();
        counts.extend(vec![1usize, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4]);
        let (most_count, most_total) = counts.most_frequent();
        assert_eq!(most_count[0], (&2, 5));
        assert_eq!(most_total, 11);
        let (least_count, least_total) = counts.least_frequent();
        assert_eq!(least_count[0], (&3, 1));
        assert_eq!(least_total, 11);
        assert_eq!(
            counts.least_frequent(),
            (vec![(&3, 1), (&1, 2), (&4, 3), (&2, 5)], 11)
        );
    }

    #[test]
    fn ranked2() {
        let mut counts = Frequencies::new();
        counts.extend(vec![1usize, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4]);
        let (most_count, most_total) = counts.par_frequent(false);
        assert_eq!(most_count[0], (&2, 5));
        assert_eq!(most_total, 11);
        let (least_count, least_total) = counts.par_frequent(true);
        assert_eq!(least_count[0], (&3, 1));
        assert_eq!(least_total, 11);
    }

    #[test]
    fn unique_values() {
        let freqs = Frequencies::from_iter(vec![8, 6, 5, 1, 1, 2, 2, 2, 3, 4, 7, 4, 4]);
        let mut unique: Vec<isize> = freqs.unique_values().copied().collect();
        unique.sort_unstable();
        assert_eq!(unique, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_top_n() {
        let mut freq = Frequencies::new();
        freq.extend(vec![1, 1, 1, 2, 2, 3, 4, 4, 4, 4]);

        let top_2 = freq.top_n(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0], (&4, 4)); // Most frequent
        assert_eq!(top_2[1], (&1, 3)); // Second most frequent

        let bottom_2 = freq.bottom_n(2);
        assert_eq!(bottom_2.len(), 2);
        assert_eq!(bottom_2[0], (&3, 1)); // Least frequent
        assert_eq!(bottom_2[1], (&2, 2)); // Second least frequent
    }

    #[test]
    fn test_count_methods() {
        let mut freq = Frequencies::new();
        freq.extend(vec![1, 1, 1, 2, 2, 3, 4, 4, 4, 4]);

        // Test total_count()
        assert_eq!(freq.total_count(), 10);

        // Test has_count()
        assert!(freq.has_count(3)); // 1 appears 3 times
        assert!(freq.has_count(4)); // 4 appears 4 times
        assert!(freq.has_count(1)); // 3 appears 1 time
        assert!(!freq.has_count(5)); // No element appears 5 times

        // Test items_with_count()
        let items_with_3 = freq.items_with_count(3);
        assert_eq!(items_with_3, vec![&1]); // Only 1 appears 3 times

        let items_with_2 = freq.items_with_count(2);
        assert_eq!(items_with_2, vec![&2]); // Only 2 appears 2 times

        let items_with_1 = freq.items_with_count(1);
        assert_eq!(items_with_1, vec![&3]); // Only 3 appears 1 time

        let items_with_4 = freq.items_with_count(4);
        assert_eq!(items_with_4, vec![&4]); // Only 4 appears 4 times

        let items_with_5 = freq.items_with_count(5);
        assert!(items_with_5.is_empty()); // No elements appear 5 times
    }
}
