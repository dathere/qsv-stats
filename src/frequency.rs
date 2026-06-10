use std::hash::Hash;

use hashbrown::HashMap;
use hashbrown::hash_map::{Entry, EntryRef, Keys};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Commute;
use crate::unsorted::modes_antimodes_from_runs;

const PARALLEL_THRESHOLD: usize = 10_000;
/// A commutative data structure for exact frequency counts.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize + Eq + Hash",
    deserialize = "T: Deserialize<'de> + Eq + Hash"
))]
pub struct Frequencies<T> {
    data: HashMap<T, u64>,
}

// Manual impl: the derive would bound on `T: PartialEq` only, but
// `HashMap: PartialEq` requires `T: Eq + Hash`.
impl<T: Eq + Hash> PartialEq for Frequencies<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
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
    #[allow(clippy::inline_always)]
    #[inline(always)]
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

    /// Collect counts and total in a single pass, reused by `most/least_frequent`.
    fn collect_counts(&self) -> (Vec<(&T, u64)>, u64) {
        let mut total_count = 0u64;
        let counts: Vec<(&T, u64)> = self
            .data
            .iter()
            .map(|(k, &v)| {
                total_count += v;
                (k, v)
            })
            .collect();
        (counts, total_count)
    }

    /// Return a `Vec` of elements, their corresponding counts in
    /// descending order, and the total count.
    #[inline]
    #[must_use]
    pub fn most_frequent(&self) -> (Vec<(&T, u64)>, u64) {
        let (mut counts, total_count) = self.collect_counts();
        counts.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
        (counts, total_count)
    }

    /// Return a `Vec` of elements, their corresponding counts in
    /// ascending order, and the total count.
    #[inline]
    #[must_use]
    pub fn least_frequent(&self) -> (Vec<(&T, u64)>, u64) {
        let (mut counts, total_count) = self.collect_counts();
        counts.sort_unstable_by_key(|&(_, c)| c);
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
        let (mut counts, total_count) = self.collect_counts();
        // sort by counts asc/desc
        // if counts are equal, sort by values lexicographically
        // We need to do this because otherwise the values are not guaranteed to be in order for
        // equal counts
        if least {
            // return counts in ascending order
            let sort_fn =
                |&(v1, c1): &(&T, u64), &(v2, c2): &(&T, u64)| c1.cmp(&c2).then_with(|| v1.cmp(v2));
            if counts.len() < PARALLEL_THRESHOLD {
                counts.sort_unstable_by(sort_fn);
            } else {
                counts.par_sort_unstable_by(sort_fn);
            }
        } else {
            // return counts in descending order
            let sort_fn =
                |&(v1, c1): &(&T, u64), &(v2, c2): &(&T, u64)| c2.cmp(&c1).then_with(|| v1.cmp(v2));
            if counts.len() < PARALLEL_THRESHOLD {
                counts.sort_unstable_by(sort_fn);
            } else {
                counts.par_sort_unstable_by(sort_fn);
            }
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

        // Ties are broken by value *ascending* so the result matches
        // `most_frequent()` / `par_frequent(false)` (count desc, value asc).
        // We achieve this by ranking each entry on `(count, Reverse(item))`:
        // a larger count wins, and on a count tie the lexicographically smaller
        // value wins (its `Reverse(item)` is larger).
        //
        // Min-heap (via the outer Reverse) of the n best entries seen so far.
        // peek() returns the worst entry currently in the top-N — replace it
        // only when a strictly better candidate comes in. Avoids a push+pop per
        // rejected element on high-cardinality inputs with small n.
        let mut heap = BinaryHeap::with_capacity(n);

        for (item, count) in &self.data {
            let candidate = (*count, std::cmp::Reverse(item));
            if heap.len() < n {
                heap.push(std::cmp::Reverse(candidate));
            } else if let Some(top) = heap.peek()
                && candidate > top.0
            {
                heap.pop();
                heap.push(std::cmp::Reverse(candidate));
            }
        }

        // Convert to sorted vector (count desc, value asc)
        heap.into_sorted_vec()
            .into_iter()
            .map(|std::cmp::Reverse((count, std::cmp::Reverse(item)))| (item, count))
            .collect()
    }

    /// Similar to `top_n` but for least frequent items
    #[must_use]
    pub fn bottom_n(&self, n: usize) -> Vec<(&T, u64)>
    where
        T: Ord,
    {
        use std::collections::BinaryHeap;

        // Max-heap of the n smallest elements seen so far. Mirror of top_n.
        let mut heap = BinaryHeap::with_capacity(n);

        for (item, count) in &self.data {
            let candidate = (*count, item);
            if heap.len() < n {
                heap.push(candidate);
            } else if let Some(top) = heap.peek()
                && candidate < *top
            {
                heap.pop();
                heap.push(candidate);
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
    #[allow(clippy::inline_always)]
    #[inline(always)]
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

impl<T: Eq + Hash + Ord + Clone + Send + Sync> Frequencies<T> {
    /// Returns the modes and antimodes of the data.
    ///
    /// Produces results identical to [`crate::Unsorted::modes_antimodes`] for the
    /// same multiset of samples: the frequency map's `(value, count)` pairs,
    /// sorted ascending by value, describe the exact same run sequence that
    /// `Unsorted` derives from its fully sorted sample buffer. Both paths are
    /// routed through the same `modes_antimodes_from_runs` core.
    ///
    /// Unlike `Unsorted`, this only sorts the *unique* values (cardinality),
    /// not every sample - O(c log c) instead of O(n log n) - and the frequency
    /// map itself stores one entry per unique value instead of one per sample.
    ///
    /// Returns `((modes, modes_count, mode_occurrences),
    /// (antimodes, antimodes_count, antimode_occurrences))`.
    /// Only the first 10 antimodes are returned.
    #[allow(clippy::type_complexity)]
    #[must_use]
    pub fn modes_antimodes(&self) -> ((Vec<T>, usize, u32), (Vec<T>, usize, u32)) {
        let mut runs: Vec<(&T, u32)> = self
            .data
            .iter()
            .map(|(k, &c)| (k, u32::try_from(c).unwrap_or(u32::MAX)))
            .collect();

        // sort ascending by value - same ordering Unsorted uses for its samples
        if runs.len() > PARALLEL_THRESHOLD {
            runs.par_sort_unstable_by(|a, b| a.0.cmp(b.0));
        } else {
            runs.sort_unstable_by(|a, b| a.0.cmp(b.0));
        }

        let mut highest_count = 1_u32;
        let mut lowest_count = u32::MAX;
        for &(_, c) in &runs {
            highest_count = highest_count.max(c);
            lowest_count = lowest_count.min(c);
        }

        modes_antimodes_from_runs(runs, highest_count, lowest_count)
    }
}

impl Frequencies<Vec<u8>> {
    /// Increment count for a byte slice key, avoiding allocation when key exists.
    /// Uses hashbrown's `entry_ref(&[u8])`, which probes once with the borrowed
    /// key and only allocates (`[u8]::to_owned()` -> `Vec<u8>`) on the vacant
    /// branch. For low-cardinality columns (the common case), this eliminates
    /// ~99% of allocations; for new keys it is a single hash+probe (std's
    /// HashMap has no stable raw-entry API, so the old path hashed twice).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn add_borrowed(&mut self, v: &[u8]) {
        *self.data.entry_ref(v).or_insert(0) += 1;
    }

    /// Increment by a count for a byte slice key, avoiding allocation when key exists.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn increment_by_borrowed(&mut self, v: &[u8], count: u64) {
        *self.data.entry_ref(v).or_insert(0) += count;
    }

    /// Increment the count for `v`, enforcing a cardinality cap.
    ///
    /// Existing keys always increment (the map doesn't grow). A NEW key that
    /// would grow the map past `cap` unique entries is rejected: the map is
    /// left unchanged and `false` is returned, so the caller can drop the
    /// tracker. `cap == 0` means unbounded.
    ///
    /// Like [`Self::add_borrowed`], this single-probes via `entry_ref` and
    /// only allocates an owned key on the (admitted) vacant branch.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn add_borrowed_capped(&mut self, v: &[u8], cap: u64) -> bool {
        let len = self.data.len() as u64;
        match self.data.entry_ref(v) {
            EntryRef::Occupied(mut e) => {
                *e.get_mut() += 1;
                true
            }
            EntryRef::Vacant(e) => {
                if cap > 0 && len >= cap {
                    false
                } else {
                    e.insert(1);
                    true
                }
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
            data: HashMap::with_capacity(64),
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
        let iter = it.into_iter();
        // Reserve capacity if size hint is available and reliable
        if let (lower, Some(upper)) = iter.size_hint()
            && lower == upper
        {
            // Exact size known - reserve capacity for new entries
            // We don't know how many will be new vs existing, so reserve conservatively
            self.data.reserve(lower.saturating_sub(self.data.len()));
        }
        for sample in iter {
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
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.data_keys.next()
    }

    // Forward the exact size hint from the underlying `Keys` iterator so that
    // `collect`/`extend` can preallocate exactly instead of reallocating.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.data_keys.size_hint()
    }
}

impl<K> ExactSizeIterator for UniqueValues<'_, K> {
    #[inline]
    fn len(&self) -> usize {
        self.data_keys.len()
    }
}

impl<K> std::iter::FusedIterator for UniqueValues<'_, K> {}

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

    // top_n/bottom_n must select the SAME set and order as
    // par_frequent(..) truncated to n, including the tie-break at the
    // n/n+1 boundary (count primary, value ascending on ties). This is the
    // invariant qsv's `frequency --limit N` fast path relies on.
    #[test]
    fn top_n_matches_par_frequent_all_ties() {
        // All counts equal: every comparison is a pure value tie-break.
        let mut freq = Frequencies::new();
        freq.extend(vec![1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        for n in 0..=10usize {
            let (full_desc, _) = freq.par_frequent(false);
            let expected_desc: Vec<(&usize, u64)> = full_desc.into_iter().take(n).collect();
            assert_eq!(freq.top_n(n), expected_desc, "top_n({n}) all-ties mismatch");

            let (full_asc, _) = freq.par_frequent(true);
            let expected_asc: Vec<(&usize, u64)> = full_asc.into_iter().take(n).collect();
            assert_eq!(
                freq.bottom_n(n),
                expected_asc,
                "bottom_n({n}) all-ties mismatch"
            );
        }
    }

    #[test]
    fn top_n_matches_par_frequent_boundary_ties() {
        // Counts deliberately tie across the cutoff: values 10..19 all count 2,
        // values 20..24 count 5, values 30..34 count 1.
        let mut freq = Frequencies::new();
        for v in 10..20usize {
            freq.extend(vec![v, v]); // count 2
        }
        for v in 20..25usize {
            freq.extend(vec![v; 5]); // count 5
        }
        for v in 30..35usize {
            freq.extend(vec![v]); // count 1
        }
        for n in 0..=freq.len() + 2 {
            let (full_desc, _) = freq.par_frequent(false);
            let expected_desc: Vec<(&usize, u64)> = full_desc.into_iter().take(n).collect();
            assert_eq!(
                freq.top_n(n),
                expected_desc,
                "top_n({n}) boundary-tie mismatch"
            );

            let (full_asc, _) = freq.par_frequent(true);
            let expected_asc: Vec<(&usize, u64)> = full_asc.into_iter().take(n).collect();
            assert_eq!(
                freq.bottom_n(n),
                expected_asc,
                "bottom_n({n}) boundary-tie mismatch"
            );
        }
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

    #[test]
    fn add_borrowed_inserts_new_key() {
        let mut freq = Frequencies::<Vec<u8>>::new();
        freq.add_borrowed(b"hello");
        assert_eq!(freq.count(&b"hello".to_vec()), 1);
        assert_eq!(freq.cardinality(), 1);
    }

    #[test]
    fn add_borrowed_increments_existing_key() {
        let mut freq = Frequencies::<Vec<u8>>::new();
        freq.add_borrowed(b"hello");
        freq.add_borrowed(b"hello");
        freq.add_borrowed(b"hello");
        assert_eq!(freq.count(&b"hello".to_vec()), 3);
        assert_eq!(freq.cardinality(), 1);

        // Also test increment_by_borrowed
        freq.increment_by_borrowed(b"world", 5);
        assert_eq!(freq.count(&b"world".to_vec()), 5);
        freq.increment_by_borrowed(b"world", 3);
        assert_eq!(freq.count(&b"world".to_vec()), 8);
    }

    #[test]
    fn borrowed_owned_interop_for_same_key() {
        let mut freq = Frequencies::<Vec<u8>>::new();
        // Insert via owned add
        freq.add(b"key".to_vec());
        // Increment via borrowed add
        freq.add_borrowed(b"key");
        freq.increment_by_borrowed(b"key", 3);
        // All methods should see the same accumulated count
        assert_eq!(freq.count(&b"key".to_vec()), 5);
        assert_eq!(freq.cardinality(), 1);
    }

    /// Property test: `Frequencies::modes_antimodes` must produce results
    /// identical to `Unsorted::modes_antimodes` for the same multiset of
    /// samples, and the cardinalities must match. This is the behavior
    /// preservation proof for replacing the `Unsorted<Vec<u8>>` modes tracker
    /// with a `Frequencies<Vec<u8>>` counted-runs map.
    #[test]
    fn modes_antimodes_matches_unsorted() {
        // simple deterministic LCG so the test needs no rand dependency
        let mut seed = 0xDEAD_BEEF_u64;
        let mut next = move |bound: u64| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 33) % bound
        };

        for case in 0..200 {
            // vary size and cardinality to hit the special cases:
            // empty, single unique, all unique, ties, skewed
            let n_samples = next(50) as usize;
            let key_space = 1 + next(30);

            let mut unsorted = crate::Unsorted::<Vec<u8>>::default();
            let mut freqs = Frequencies::<Vec<u8>>::new();
            for _ in 0..n_samples {
                let key = format!("k{:02}", next(key_space)).into_bytes();
                unsorted.add_bytes(&key);
                freqs.add_borrowed(&key);
            }

            assert_eq!(
                unsorted.cardinality(false, 1),
                freqs.cardinality(),
                "cardinality mismatch in case {case}"
            );
            assert_eq!(
                unsorted.modes_antimodes(),
                freqs.modes_antimodes(),
                "modes/antimodes mismatch in case {case} (n={n_samples}, k={key_space})"
            );
        }

        // explicit edge cases
        // empty
        let mut u = crate::Unsorted::<Vec<u8>>::default();
        let f = Frequencies::<Vec<u8>>::new();
        assert_eq!(u.modes_antimodes(), f.modes_antimodes());

        // single unique value, multiple occurrences
        let mut u = crate::Unsorted::<Vec<u8>>::default();
        let mut f = Frequencies::<Vec<u8>>::new();
        for _ in 0..5 {
            u.add_bytes(b"only");
            f.add_borrowed(b"only");
        }
        assert_eq!(u.modes_antimodes(), f.modes_antimodes());

        // all values unique (highest_count == 1), more than 10 antimodes
        let mut u = crate::Unsorted::<Vec<u8>>::default();
        let mut f = Frequencies::<Vec<u8>>::new();
        for i in 0..15 {
            let key = format!("u{i:02}").into_bytes();
            u.add_bytes(&key);
            f.add_borrowed(&key);
        }
        assert_eq!(u.modes_antimodes(), f.modes_antimodes());
    }
}
