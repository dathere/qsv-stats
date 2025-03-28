use std::collections::BinaryHeap;
// use foldhash::{HashMap, HashMapExt};
use std::default::Default;
use std::iter::{FromIterator, IntoIterator};

use num_traits::ToPrimitive;

use {crate::Commute, crate::Partial};

pub fn median_on_sorted<T>(data: &[T]) -> Option<f64>
        where T: PartialOrd + ToPrimitive {
    let len = data.len();
    match len {
        0 => None,
        // safety: // Use direct indexing since we know len == 1
        1 => data[0].to_f64(),
        len if len % 2 == 0 => {
            let idx = len / 2;
            // SAFETY: We know idx and idx-1 are valid because len >= 2
            unsafe {
                let v1 = data.get_unchecked(idx - 1).to_f64()?;
                let v2 = data.get_unchecked(idx).to_f64()?;
                Some((v1 + v2) / 2.0)
            }
        }
        len => {
            // SAFETY: We know len/2 is valid because len >= 3
            unsafe { data.get_unchecked(len / 2).to_f64() }
        }
    }
}

pub fn mode_on_sorted<T, I>(it: I) -> Option<T>
        where T: PartialOrd, I: Iterator<Item=T> {
    // This approach to computing the mode works very nicely when the
    // number of samples is large and is close to its cardinality.
    // In other cases, a hashmap would be much better.
    // But really, how can we know this when given an arbitrary stream?
    // Might just switch to a hashmap to track frequencies. That would also
    // be generally useful for discovering the cardinality of a sample.
    let (mut mode, mut next) = (None, None);
    let (mut mode_count, mut next_count) = (0usize, 0usize);
    for x in it {
        if mode.as_ref().map(|y| y == &x).unwrap_or(false) {
            mode_count += 1;
        } else if next.as_ref().map(|y| y == &x).unwrap_or(false) {
            next_count += 1;
        } else {
            next = Some(x);
            next_count = 0;
        }

        if next_count > mode_count {
            mode = next;
            mode_count = next_count;
            next = None;
            next_count = 0;
        } else if next_count == mode_count {
            mode = None;
            mode_count = 0usize;
        }
    }
    mode
}

// pub fn mode_hashmap<T, I>(it: I) -> Option<T>
// where
//     T: Eq + Hash + Clone,
//     I: Iterator<Item = T>,
// {
//     let mut counts = HashMap::new();
//     let mut mode = None;
//     let mut mode_count = 0;
//     for x in it {
//         let count = counts.entry(x.clone()).or_insert(0);
//         *count += 1;
//         if *count > mode_count {
//             mode = Some(x.clone());
//             mode_count = *count;
//         }
//     }
//     mode
// }

/// A commutative data structure for sorted sequences of data.
///
/// Note that this works on types that do not define a total ordering like
/// `f32` and `f64`. Then an ordering is not defined, an arbitrary order
/// is returned.
#[derive(Clone, Deserialize, Serialize, Eq, PartialEq, Debug)]
pub struct Sorted<T> {
    data: BinaryHeap<Partial<T>>,
}

impl<T: PartialOrd> Sorted<T> {
    /// Create initial empty state.
    #[inline]
    pub fn new() -> Sorted<T> {
        Default::default()
    }

    /// Add a new element to the set.
    #[inline]
    pub fn add(&mut self, v: T) {
        self.data.push(Partial(v))
    }

    /// Returns the number of data points.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T: PartialOrd + Clone> Sorted<T> {
    /// Returns the mode of the data.
    #[inline]
    pub fn mode(&self) -> Option<T> {
        let p = mode_on_sorted(self.data.clone().into_sorted_vec().into_iter());
        p.map(|p| p.0)
    }

    // #[inline]
    // pub fn mode2(&self) -> Option<T> {
    //     let p = mode_hashmap(self.data.clone().into_iter());
    //     p.map(|p| p.0)
    // }
}

impl<T: PartialOrd + ToPrimitive + Clone> Sorted<T> {
    /// Returns the median of the data.
    #[inline]
    pub fn median(&self) -> Option<f64> {
        // Grr. The only way to avoid the alloc here is to take `self` by
        // value. Could return `(f64, Sorted<T>)`, but that seems a bit weird.
        //
        // NOTE: Can `std::mem::swap` help us here?
        let data = self.data.clone().into_sorted_vec();
        median_on_sorted(&*data)
    }
}

impl<T: PartialOrd> Commute for Sorted<T> {
    #[inline]
    fn merge(&mut self, v: Sorted<T>) {
        // should this be `into_sorted_vec`?
        self.data.extend(v.data.into_vec().into_iter());
    }
}

impl<T: PartialOrd> Default for Sorted<T> {
    #[inline]
    fn default() -> Sorted<T> { Sorted { data: BinaryHeap::new() } }
}

impl<T: PartialOrd> FromIterator<T> for Sorted<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item=T>>(it: I) -> Sorted<T> {
        let mut v = Sorted::new();
        v.extend(it);
        v
    }
}

impl<T: PartialOrd> Extend<T> for Sorted<T> {
    #[inline]
    fn extend<I: IntoIterator<Item=T>>(&mut self, it: I) {
        self.data.extend(it.into_iter().map(Partial))
    }
}

#[cfg(test)]
mod test {
    use num::ToPrimitive;
    use super::Sorted;

    fn median<T, I>(it: I) -> Option<f64>
       where T: PartialOrd + ToPrimitive + Clone, I: Iterator<Item=T> {
        it.collect::<Sorted<T>>().median()
    }

    fn mode<T, I>(it: I) -> Option<T>
       where T: PartialOrd + Clone, I: Iterator<Item=T> {
        it.collect::<Sorted<T>>().mode()
    }

    fn mode2<T, I>(it: I) -> Option<T>
    where T: PartialOrd + Clone, I: Iterator<Item=T> {
     it.collect::<Sorted<T>>().mode2()
 }

    #[test]
    fn median_stream() {
        assert_eq!(median(vec![3usize, 5, 7, 9].into_iter()), Some(6.0));
        assert_eq!(median(vec![3usize, 5, 7].into_iter()), Some(5.0));
    }

    #[test]
    fn mode_stream() {
        assert_eq!(mode(vec![3usize, 5, 7, 9].into_iter()), None);
        assert_eq!(mode(vec![3usize, 3, 3, 3].into_iter()), Some(3));
        assert_eq!(mode(vec![3usize, 3, 3, 4].into_iter()), Some(3));
        assert_eq!(mode(vec![4usize, 3, 3, 3].into_iter()), Some(3));
        assert_eq!(mode(vec![1usize, 1, 2, 3, 3].into_iter()), None);
    }

    // #[test]
    // fn mode2_stream() {
    //     assert_eq!(mode2(vec![3usize, 5, 7, 9].into_iter()), None);
    //     assert_eq!(mode2(vec![3usize, 3, 3, 3].into_iter()), Some(3));
    //     assert_eq!(mode2(vec![3usize, 3, 3, 4].into_iter()), Some(3));
    //     assert_eq!(mode2(vec![4usize, 3, 3, 3].into_iter()), Some(3));
    //     assert_eq!(mode2(vec![1usize, 1, 2, 3, 3].into_iter()), None);
    // }

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

    // #[test]
    // fn mode2_floats() {
    //     assert_eq!(mode2(vec![3.0f64, 5.0, 7.0, 9.0].into_iter()), None);
    //     assert_eq!(mode2(vec![3.0f64, 3.0, 3.0, 3.0].into_iter()), Some(3.0));
    //     assert_eq!(mode2(vec![3.0f64, 3.0, 3.0, 4.0].into_iter()), Some(3.0));
    //     assert_eq!(mode2(vec![4.0f64, 3.0, 3.0, 3.0].into_iter()), Some(3.0));
    //     assert_eq!(mode2(vec![1.0f64, 1.0, 2.0, 3.0, 3.0].into_iter()), None);
    // }
}
