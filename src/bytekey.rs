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
//! On big-endian targets (e.g. s390x) storage falls back to a plain `Vec<u8>`
//! (no inline keys, [`BYTEKEY_INLINE_CAP`] is 0); see the `storage` module.
//!
//! Hashing, equality, ordering and serde all behave as the equivalent
//! `[u8]` / `Vec<u8>` would, so `Frequencies<ByteKey>` produces the same
//! counts, modes and antimodes (same order) as `Frequencies<Vec<u8>>`.

use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use storage::Storage;

/// Keys up to this many bytes are stored inline (no heap allocation).
/// 0 on big-endian targets, where storage falls back to `Vec<u8>`.
pub const BYTEKEY_INLINE_CAP: usize = storage::INLINE_CAP;

/// Backing storage for [`ByteKey`].
///
/// TEMPORARY big-endian fallback: compact_bytes 0.2.1 overlaps its inline tag
/// byte with the last byte of the heap `cap` field. On big-endian that is the
/// LEAST significant byte, so any heap capacity whose low byte is >= 0x80
/// (e.g. 128) is misread as inline: wrong contents, leaked buffers, and
/// out-of-bounds reads (reproduced under miri on s390x). Fixed upstream in
/// <https://github.com/ParkMyCar/compact_bytes/pull/12>. Remove this fallback
/// (and the target-specific dependency in Cargo.toml) once a compact_bytes
/// release with that fix is on crates.io.
#[cfg(target_endian = "little")]
mod storage {
    pub(super) type Storage = compact_bytes::CompactBytes;
    pub(super) const INLINE_CAP: usize = Storage::MAX_INLINE;

    #[inline]
    pub(super) fn new(bytes: &[u8]) -> Storage {
        Storage::new(bytes)
    }
    #[inline]
    pub(super) fn empty() -> Storage {
        Storage::empty()
    }
    #[inline]
    pub(super) fn from_vec(v: Vec<u8>) -> Storage {
        Storage::from(v)
    }
    #[inline]
    pub(super) fn into_vec(s: Storage) -> Vec<u8> {
        s.into_vec()
    }
    #[inline]
    pub(super) fn is_inline(s: &Storage) -> bool {
        !s.spilled()
    }
}

#[cfg(target_endian = "big")]
mod storage {
    pub(super) type Storage = Vec<u8>;
    pub(super) const INLINE_CAP: usize = 0;

    #[inline]
    pub(super) fn new(bytes: &[u8]) -> Storage {
        bytes.to_vec()
    }
    #[inline]
    pub(super) fn empty() -> Storage {
        Vec::new()
    }
    #[inline]
    pub(super) fn from_vec(v: Vec<u8>) -> Storage {
        v
    }
    #[inline]
    pub(super) fn into_vec(s: Storage) -> Vec<u8> {
        s
    }
    #[inline]
    pub(super) fn is_inline(s: &Storage) -> bool {
        // Only an empty, never-allocated key has no heap buffer.
        s.capacity() == 0
    }
}

/// An owned byte-string key with small-buffer optimization. See the module docs.
#[derive(Clone, PartialEq, Eq)]
pub struct ByteKey(Storage);

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
        Self(storage::new(bytes))
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
        storage::is_inline(&self.0)
    }
}

impl Default for ByteKey {
    #[inline]
    fn default() -> Self {
        Self(storage::empty())
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
        Self(storage::from_vec(v))
    }
}

impl From<ByteKey> for Vec<u8> {
    #[inline]
    fn from(k: ByteKey) -> Self {
        storage::into_vec(k.0)
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

    /// Endian-agnostic round trip over every length through the inline/heap
    /// boundary and past 255, so heap capacities with every low-byte value are
    /// covered (the case compact_bytes 0.2.1 mis-tags on big-endian).
    #[test]
    fn round_trip_all_lengths() {
        use std::hash::BuildHasher;
        let bh = hashbrown::DefaultHashBuilder::default();
        let mut prev: Option<Vec<u8>> = None;
        for len in 0..=600usize {
            let v: Vec<u8> = (0..len).map(|i| (i * 31 + len) as u8).collect();
            let k = ByteKey::from_slice(&v);
            assert_eq!(k.as_slice(), v.as_slice(), "from_slice len {len}");
            assert_eq!(&*k, v.as_slice());
            assert_eq!(bh.hash_one(&k), bh.hash_one(v.as_slice()));
            assert_eq!(bh.hash_one(&k), bh.hash_one(ByteKeyRef::new(&v)));
            let kc = k.clone();
            assert_eq!(kc, k);
            let kv = ByteKey::from(v.clone());
            assert_eq!(kv.as_slice(), v.as_slice(), "From<Vec> len {len}");
            assert_eq!(kv, k);
            assert_eq!(Vec::<u8>::from(kv), v, "into Vec len {len}");
            if let Some(p) = &prev {
                let pk = ByteKey::from_slice(p);
                assert_eq!(pk.cmp(&k), p.cmp(&v), "Ord len {len}");
            }
            prev = Some(v);
        }
    }
}
