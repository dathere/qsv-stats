//! `ByteKey`: a byte-string key that stores short values inline.
//!
//! `Frequencies<Vec<u8>>` heap-allocates (and later frees) one
//! buffer per unique key, and every lookup hit must follow that pointer to
//! compare bytes. `ByteKey` keeps keys of up to [`BYTEKEY_INLINE_CAP`] bytes inside
//! the 24-byte key itself (the same size as `Vec<u8>`), so short and
//! medium-length values (IDs, codes, coordinates, `MM/DD/YYYY HH:MM:SS AM`
//! timestamps) skip the allocation, the free and the pointer chase. Longer
//! keys spill to the heap like `Vec<u8>`.
//!
//! Storage is `compact_bytes::CompactBytes` (23 bytes inline in 24, using the
//! last byte as the inline/heap tag). It is wrapped, not re-exported, so the
//! backing type can change without an API break.
//!
//! Hashing, equality, ordering and serde all behave as the equivalent
//! `[u8]` / `Vec<u8>` would, so `Frequencies<ByteKey>` produces the same
//! counts, modes and antimodes (same order) as `Frequencies<Vec<u8>>`.

use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use compact_bytes::CompactBytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Keys up to this many bytes are stored inline (no heap allocation).
pub const BYTEKEY_INLINE_CAP: usize = CompactBytes::MAX_INLINE;

/// An owned byte-string key with small-buffer optimization. See the module docs.
#[derive(Clone, PartialEq, Eq)]
pub struct ByteKey(CompactBytes);

/// The borrowed form of [`ByteKey`] (a `[u8]` newtype), used so hashbrown's
/// `entry_ref` can probe with a borrowed slice and only build an owned
/// `ByteKey` on the vacant branch (hashbrown requires `Q: ToOwned<Owned = K>`).
#[derive(PartialEq, Eq)]
#[repr(transparent)]
pub struct ByteKeyRef([u8]);

impl ByteKeyRef {
    #[inline(always)]
    #[must_use]
    pub fn new(bytes: &[u8]) -> &Self {
        // SAFETY: `ByteKeyRef` is `#[repr(transparent)]` over `[u8]`, so the
        // two have identical layout and pointer metadata (the slice length).
        // `ByteKeyRef` adds no validity invariant (every `[u8]` is a valid
        // `ByteKeyRef`), and lifetime elision ties the result to `bytes`.
        unsafe { &*(std::ptr::from_ref::<[u8]>(bytes) as *const Self) }
    }
}

impl ByteKey {
    #[inline]
    #[must_use]
    pub fn from_slice(bytes: &[u8]) -> Self {
        Self(CompactBytes::new(bytes))
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    /// True when the key is stored inline (no heap buffer).
    #[inline]
    #[must_use]
    pub fn is_inline(&self) -> bool {
        !self.0.spilled()
    }
}

impl Default for ByteKey {
    #[inline]
    fn default() -> Self {
        Self(CompactBytes::empty())
    }
}

// Order exactly as `[u8]` / `Vec<u8>` (CompactBytes has no `Ord`), so modes
// and antimodes come out in the same order as with `Vec<u8>` keys.
impl PartialOrd for ByteKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ByteKey {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Deref for ByteKey {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl AsRef<[u8]> for ByteKey {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

// Hash exactly as `[u8]` on both sides so the owned and borrowed forms agree
// (required for `entry_ref`), and so hashing matches `Vec<u8>` keys.
impl Hash for ByteKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl Hash for ByteKeyRef {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Borrow<ByteKeyRef> for ByteKey {
    #[inline]
    fn borrow(&self) -> &ByteKeyRef {
        ByteKeyRef::new(self.0.as_slice())
    }
}

impl ToOwned for ByteKeyRef {
    type Owned = ByteKey;
    #[inline]
    fn to_owned(&self) -> ByteKey {
        ByteKey::from_slice(&self.0)
    }
}

impl From<&[u8]> for ByteKey {
    #[inline]
    fn from(bytes: &[u8]) -> Self {
        Self::from_slice(bytes)
    }
}

impl From<Vec<u8>> for ByteKey {
    #[inline]
    fn from(v: Vec<u8>) -> Self {
        Self(CompactBytes::from(v))
    }
}

impl From<ByteKey> for Vec<u8> {
    #[inline]
    fn from(k: ByteKey) -> Self {
        k.0.into_vec()
    }
}

impl std::fmt::Debug for ByteKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

// Serialize/deserialize with the same wire shape as `Vec<u8>` (a sequence of
// bytes), so serialized `Frequencies` stay format-compatible.
impl Serialize for ByteKey {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(s)
    }
}

impl<'de> Deserialize<'de> for ByteKey {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Vec::<u8>::deserialize(d).map(Self::from)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn size_matches_vec() {
        assert_eq!(size_of::<ByteKey>(), size_of::<Vec<u8>>());
    }

    #[test]
    fn inline_and_spill() {
        assert!(ByteKey::from_slice(&[7u8; BYTEKEY_INLINE_CAP]).is_inline());
        assert!(!ByteKey::from_slice(&[7u8; BYTEKEY_INLINE_CAP + 1]).is_inline());
        assert!(ByteKey::from_slice(b"").is_inline());
    }

    #[test]
    fn hash_matches_slice_and_vec() {
        use std::hash::BuildHasher;
        let bh = hashbrown::DefaultHashBuilder::default();
        for s in [&b""[..], b"abc", &[1u8; 40]] {
            let k = ByteKey::from_slice(s);
            assert_eq!(bh.hash_one(&k), bh.hash_one(ByteKeyRef::new(s)));
            assert_eq!(bh.hash_one(&k), bh.hash_one(s.to_vec()));
        }
    }

    #[test]
    fn ord_matches_vec() {
        let mut a: Vec<Vec<u8>> = vec![b"b".to_vec(), b"".to_vec(), b"ab".to_vec(), vec![b'a'; 30]];
        let mut b: Vec<ByteKey> = a.iter().map(|v| ByteKey::from_slice(v)).collect();
        a.sort();
        b.sort();
        assert!(a.iter().zip(&b).all(|(x, y)| x.as_slice() == y.as_slice()));
    }
}
