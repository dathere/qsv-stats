#![allow(unconditional_recursion)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::use_self)]

use num_traits::ToPrimitive;
use std::cmp::Ordering;
use std::hash;

use serde::{Deserialize, Serialize};

pub use frequency::{Frequencies, UniqueValues};
pub use minmax::MinMax;
pub use online::{OnlineStats, mean, stddev, variance};
pub use unsorted::{Unsorted, antimodes, mad, median, mode, modes, quartiles};

/// Partial wraps a type that satisfies `PartialOrd` and implements `Ord`.
///
/// This allows types like `f64` to be used in data structures that require
/// `Ord`. When an ordering is not defined, an arbitrary order is returned.
#[allow(clippy::non_send_fields_in_send_ty)]
#[allow(clippy::derive_ord_xor_partial_ord)]
#[derive(Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
struct Partial<T>(pub T);

impl<T: PartialEq> Eq for Partial<T> {}
unsafe impl<T> Send for Partial<T> {}
unsafe impl<T> Sync for Partial<T> {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl<T: PartialOrd> Ord for Partial<T> {
    #[inline]
    fn cmp(&self, other: &Partial<T>) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}

impl<T: ToPrimitive> ToPrimitive for Partial<T> {
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        self.0.to_isize()
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        self.0.to_i8()
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        self.0.to_i16()
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        self.0.to_i32()
    }
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    #[inline]
    fn to_usize(&self) -> Option<usize> {
        self.0.to_usize()
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        self.0.to_u8()
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        self.0.to_u16()
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        self.0.to_u32()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

#[allow(clippy::derived_hash_with_manual_eq)]
impl<T: hash::Hash> hash::Hash for Partial<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Defines an interface for types that have an identity and can be commuted.
///
/// The value returned by `Default::default` must be its identity with respect
/// to the `merge` operation.
pub trait Commute: Sized {
    /// Merges the value `other` into `self`.
    fn merge(&mut self, other: Self);

    /// Merges the values in the iterator into `self`.
    #[inline]
    fn consume<I: Iterator<Item = Self>>(&mut self, other: I) {
        for v in other {
            self.merge(v);
        }
    }
}

/// Merges all items in the stream.
///
/// If the stream is empty, `None` is returned.
#[inline]
pub fn merge_all<T: Commute, I: Iterator<Item = T>>(mut it: I) -> Option<T> {
    it.next().map_or_else(
        || None,
        |mut init| {
            init.consume(it);
            Some(init)
        },
    )
}

impl<T: Commute> Commute for Option<T> {
    #[inline]
    fn merge(&mut self, other: Option<T>) {
        match *self {
            None => {
                *self = other;
            }
            Some(ref mut v1) => {
                if let Some(v2) = other {
                    v1.merge(v2);
                }
            }
        }
    }
}

impl<T: Commute, E> Commute for Result<T, E> {
    #[inline]
    fn merge(&mut self, other: Result<T, E>) {
        if !self.is_err() && other.is_err() {
            *self = other;
            return;
        }
        #[allow(clippy::let_unit_value)]
        #[allow(clippy::ignored_unit_patterns)]
        let _ = self.as_mut().map_or((), |v1| {
            other.map_or_else(
                |_| {
                    unreachable!();
                },
                |v2| {
                    v1.merge(v2);
                },
            );
        });
    }
}

impl<T: Commute> Commute for Vec<T> {
    #[inline]
    fn merge(&mut self, other: Vec<T>) {
        assert_eq!(self.len(), other.len());
        for (v1, v2) in self.iter_mut().zip(other) {
            v1.merge(v2);
        }
    }
}

mod frequency;
mod minmax;
mod online;
mod unsorted;

#[cfg(test)]
mod test {
    use crate::Commute;
    use crate::unsorted::Unsorted;

    #[test]
    fn options() {
        let v1: Unsorted<usize> = vec![2, 1, 3, 2].into_iter().collect();
        let v2: Unsorted<usize> = vec![5, 6, 5, 5].into_iter().collect();
        let mut merged = Some(v1);
        merged.merge(Some(v2));
        assert_eq!(merged.unwrap().mode(), Some(5));
    }
}
