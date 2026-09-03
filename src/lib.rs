//! Fano: a per-chunk compression scheme for tensor data in a content-addressed store.
//!
//! Split each fixed-width element into its byte planes, Huffman-code every plane with its own table,
//! and pass incompressible planes through raw. On floating-point model weights almost all of the
//! redundancy is the order-0 entropy of the exponent byte, and this is the cheapest coder that
//! captures it. See `SPEC.md` for the byte-exact stream format and `BENCH.md` for measurements
//! against what HuggingFace Xet stores today.
//!
//! # Framing
//!
//! The C++ stream carries no length, because the container that stores it already knows one. This
//! crate prepends the uncompressed length as an LEB128 varint so that [`compress`] and
//! [`decompress`] are self-contained, and a caller with an existing
//! `fn(&[u8]) -> Result<Vec<u8>>` shaped API needs no signature change to adopt it. The cost is
//! three bytes on a 64 KiB chunk, 0.005%.
//!
//! # Example
//!
//! ```
//! let weights: Vec<u8> = (0..65536u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
//! match fano::compress(&weights) {
//!     Some(packed) => {
//!         let back = fano::decompress(&packed).unwrap();
//!         assert_eq!(back, weights);
//!     }
//!     None => { /* incompressible or shorter than 64 bytes: store it raw */ }
//! }
//! ```

use std::os::raw::c_void;

unsafe extern "C" {
    fn pe_compress_bound(n: usize) -> usize;
    fn pe_compress(dst: *mut c_void, dst_cap: usize, src: *const c_void, n: usize) -> usize;
    fn pe_decompress(dst: *mut c_void, n: usize, src: *const c_void, sz: usize) -> usize;
}

/// The shortest input the scheme accepts. Anything smaller is for the caller's raw path.
pub const MIN_INPUT: usize = 64;

/// Why a stream could not be decoded. Every variant means "store or fetch the chunk another way";
/// none of them can be produced by a stream this crate wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The length prefix is missing, truncated, or names a length this build cannot allocate.
    BadLength,
    /// The payload failed one of the format's validation rules (SPEC.md section 5).
    Malformed,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BadLength => write!(f, "fano: missing or invalid length prefix"),
            Error::Malformed => write!(f, "fano: malformed stream"),
        }
    }
}
impl std::error::Error for Error {}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(src: &[u8]) -> Option<(u64, usize)> {
    let mut v: u64 = 0;
    let mut shift = 0u32;
    for (i, &b) in src.iter().enumerate() {
        if shift == 63 && (b & 0x7e) != 0 {
            return None; // no payload bit above 63: two encodings must not name one value
        }
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((v, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

/// An upper bound on the compressed size of `n` bytes, including the length prefix.
pub fn compress_bound(n: usize) -> usize {
    let b = unsafe { pe_compress_bound(n) };
    if b == usize::MAX { usize::MAX } else { b.saturating_add(10) }
}

/// Compress `data`. Returns `None` when the scheme does not apply — the input is shorter than
/// [`MIN_INPUT`], or the stream would not be smaller than the input — in which case the caller
/// should store the chunk with its raw scheme, exactly as Xet already does when a scheme does not
/// pay. Never panics and never allocates more than [`compress_bound`].
pub fn compress(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < MIN_INPUT {
        return None;
    }
    let bound = compress_bound(data.len());
    if bound == usize::MAX {
        return None;
    }
    let mut out = Vec::with_capacity(bound);
    put_varint(&mut out, data.len() as u64);
    let head = out.len();
    out.resize(bound, 0);
    let written = unsafe {
        pe_compress(
            out.as_mut_ptr().add(head) as *mut c_void,
            bound - head,
            data.as_ptr() as *const c_void,
            data.len(),
        )
    };
    if written == 0 || head + written >= data.len() {
        return None; // failed, or no smaller than storing the chunk as it is
    }
    out.truncate(head + written);
    Some(out)
}

/// Decompress a stream produced by [`compress`]. The uncompressed length is read from the stream's
/// own prefix, so no side channel is needed. Returns [`Error`] on any malformed input; never reads
/// outside `data` and never writes outside the returned buffer.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, Error> {
    let (n, used) = get_varint(data).ok_or(Error::BadLength)?;
    let n = usize::try_from(n).map_err(|_| Error::BadLength)?;
    if n < MIN_INPUT || n > (isize::MAX as usize) {
        return Err(Error::BadLength);
    }
    let mut out = vec![0u8; n];
    let body = &data[used..];
    let got = unsafe {
        pe_decompress(
            out.as_mut_ptr() as *mut c_void,
            n,
            body.as_ptr() as *const c_void,
            body.len(),
        )
    };
    if got != n {
        return Err(Error::Malformed);
    }
    Ok(out)
}

/// Decompress into a caller-provided buffer whose length is the expected uncompressed size.
/// For a store that already records the chunk length and does not want the prefix, pass the body
/// after stripping it with [`split_prefix`].
pub fn decompress_into(dst: &mut [u8], body: &[u8]) -> Result<(), Error> {
    let got = unsafe {
        pe_decompress(
            dst.as_mut_ptr() as *mut c_void,
            dst.len(),
            body.as_ptr() as *const c_void,
            body.len(),
        )
    };
    if got == dst.len() { Ok(()) } else { Err(Error::Malformed) }
}

/// Split a stream into its declared length and the payload the C format defines.
pub fn split_prefix(data: &[u8]) -> Result<(usize, &[u8]), Error> {
    let (n, used) = get_varint(data).ok_or(Error::BadLength)?;
    let n = usize::try_from(n).map_err(|_| Error::BadLength)?;
    Ok((n, &data[used..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_weights(n: usize, seed: u64) -> Vec<u8> {
        // little-endian BF16 with a realistic exponent spread: the shape the coder is built for
        let mut s = seed | 1;
        let mut v = Vec::with_capacity(n);
        while v.len() + 2 <= n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let exp = 0x3c00u16 + ((s >> 5) % 0x300) as u16;
            let bits = exp | if s & 1 == 0 { 0 } else { 0x8000 };
            v.push((bits & 0xff) as u8);
            v.push((bits >> 8) as u8);
        }
        while v.len() < n {
            v.push(0);
        }
        v
    }

    #[test]
    fn round_trips_and_shrinks_weights() {
        // Below a few hundred bytes a Huffman table costs more than it saves, so the scheme
        // declines and the caller stores the chunk raw. That is the designed behaviour, not a
        // failure, so only the sizes a chunk store actually uses are required to shrink.
        for n in [64usize, 65, 100, 4096, 65536, 262_154] {
            let w = bf16_weights(n, 0x9E37_79B9_7F4A_7C15);
            match compress(&w) {
                Some(packed) => {
                    assert!(packed.len() < w.len(), "n={n}: {} not < {}", packed.len(), w.len());
                    assert!(packed.len() <= compress_bound(n));
                    assert_eq!(decompress(&packed).unwrap(), w, "n={n}");
                },
                None => assert!(n < 1024, "n={n} should have compressed"),
            }
        }
    }

    #[test]
    fn declines_where_it_does_not_pay() {
        assert!(compress(&[0u8; 63]).is_none(), "below the minimum");
        let mut s: u64 = 12345;
        let noise: Vec<u8> = (0..65536)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect();
        assert!(compress(&noise).is_none(), "incompressible input must fall back to raw");
    }

    #[test]
    fn refuses_malformed_streams() {
        let w = bf16_weights(65536, 7);
        let packed = compress(&w).unwrap();
        for cut in [0usize, 1, 2, 5, packed.len() / 2, packed.len() - 1] {
            assert!(decompress(&packed[..cut]).is_err(), "truncation at {cut} must be refused");
        }
        let mut bad = packed.clone();
        let (_, used) = get_varint(&bad).unwrap();
        bad[used] ^= 0x01; // the k byte of the first segment
        assert!(decompress(&bad).is_err(), "a corrupt plane count must be refused");
        // a varint that would wrap a 64-bit length
        let overlong = [0x81u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02];
        assert_eq!(decompress(&overlong), Err(Error::BadLength));
    }

    #[test]
    fn prefix_is_self_describing() {
        let w = bf16_weights(4096, 3);
        let packed = compress(&w).unwrap();
        let (n, body) = split_prefix(&packed).unwrap();
        assert_eq!(n, w.len());
        let mut out = vec![0u8; n];
        decompress_into(&mut out, body).unwrap();
        assert_eq!(out, w);
    }
}
