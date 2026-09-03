# Adding Fano to a chunk store: the xet-core touch points

This is what the change looks like inside xet-core (`xet_core_structures/src/xorb_object/`,
read at upstream main 7af65ba2, 2026-09-01). Nothing here is merged anywhere; it is the shortest
path from the package to a running per-chunk scheme, with every touch point named.

## 1. The chunk record already has everything the coder needs

```rust
// xorb_chunk_format.rs
#[repr(C, packed)]
pub struct XorbChunkHeader {
    pub version: u8,              // CURRENT_VERSION = 0
    compressed_length: [u8; 3],   // 24-bit
    compression_scheme: u8,       // CompressionScheme as u8
    uncompressed_length: [u8; 3], // 24-bit
}
```

* `uncompressed_length` is the `n` that `pe_decompress` requires (the stream carries no length by
  design, SPEC.md section 1). 24 bits caps a chunk at 16 MiB; the coder's segments are 256 KiB.
* The chunk hash is computed over the uncompressed bytes, so dedup is untouched by the scheme.
* `serialize_chunk` already stores raw when `compressed.len() >= chunk.len()`. `pe_compress` may
  return a stream larger than its input on incompressible data; that path needs no new code.
* There is no checksum in the payload and none is needed: the xorb layer hashes the chunk.

## 2. The enum, four arms

```rust
// compression_scheme.rs
pub enum CompressionScheme {
    None = 0,
    LZ4 = 1,
    ByteGrouping4LZ4 = 2,
    Fano = 3,          // new
    Auto = 99,
}
// TryFrom<u8>: 3 => Ok(CompressionScheme::Fano)
// &str mapping: "Fano"
// compress_from_slice:   Fano => fano_compress(data)
// decompress_from_slice: Fano => fano_decompress(data, uncompressed_len)
// decompress_from_reader: read compressed_length bytes, then the slice path
```

`decompress_from_slice` today takes only the compressed bytes; the Fano arm needs the
uncompressed length as well. It is in the header the caller just parsed
(`deserialize_chunk_to_writer` reads `get_uncompressed_length()` before dispatching), so the
signature change is one extra `usize` on the decompress arms, or a scheme-specific call at the
dispatch site.

## 3. The bridge: C ABI through `cc`, no Rust port needed

The package is three C functions (`plane_entropy.h`) plus vendored huff0 (BSD-2, seven `.c`
files). A `build.rs` with the `cc` crate compiles `plane_entropy.cpp` and `third_party/fse/*.c`;
the Rust side is:

```rust
extern "C" {
    fn pe_compress_bound(n: usize) -> usize;
    fn pe_compress(dst: *mut u8, dst_cap: usize, src: *const u8, n: usize) -> usize;
    fn pe_decompress(dst: *mut u8, n: usize, src: *const u8, sz: usize) -> usize;
}
fn fano_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = vec![0u8; unsafe { pe_compress_bound(data.len()) }];
    let sz = unsafe { pe_compress(out.as_mut_ptr(), out.len(), data.as_ptr(), data.len()) };
    if sz == 0 { return Err(CoreError::MalformedData("pe_compress failed".into())); }
    out.truncate(sz); Ok(out)
}
fn fano_decompress(data: &[u8], n: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; n];
    let got = unsafe { pe_decompress(out.as_mut_ptr(), n, data.as_ptr(), data.len()) };
    if got != n { return Err(CoreError::MalformedData("Fano stream rejected".into())); }
    Ok(out)
}
```

`pe_decompress` returns 0 on any malformed input and never touches memory outside the two
buffers (SPEC.md section 5; `test_pe` and `fuzz_pe` under ASAN/UBSAN). A pure-Rust port is
possible later; `vectors.txt` carries 144 frozen streams to pin it, and the only real work is
huff0, for which no maintained Rust crate exists.

## 4. Selection: when to pick scheme 3

`CompressionScheme::choose_from_data` today runs `BG4Predictor` over the chunk and picks
`ByteGrouping4LZ4` when the max KL divergence of the four byte-position histograms exceeds 0.02,
else `LZ4`. Two policies, both measured in this programme (FINAL-TABLE.md):

* **Cheapest change**: keep the predictor, and where it says bg4 (F32/BF16 tensors), call
  `pe_compress` instead. That is the set where the coder wins all three axes (11-36% smaller,
  1.4-2.4x faster to compress, 1.4-2.1x faster to decompress than bg4-lz4).
* **Full coverage**: also try `pe_compress` where the predictor says LZ4 and the chunk is not
  text-like (F16 tensors fail the KL test, which is why Xet stores them raw today; the coder takes
  12-33% off them). The trial costs one encode, 50-75 us per 64 KiB chunk on one core. Keep the
  result only if it beats the alternative by a margin the store chooses; on GGUF quantised
  blocks the gain is 6% and the decode is 20x slower than the memcpy it replaces, so a threshold
  of about 8% keeps those raw.

A predictor that avoids the trial encode is a natural next step: the encoder already computes the
Miller-Madow order-0 estimate of every plane to choose `k` (SPEC.md section 4); exposing it as
`pe_estimate_bits(src, n)` would give the selector the expected size for the cost of one
histogram pass.

## 5. Rollout: readers first

`XorbChunkHeader::validate` rejects an unknown scheme byte (`TryFrom<u8>` returns
`MalformedData`), so a client without the decoder cannot read a scheme-3 chunk. The order is:

1. ship the decoder in the client library (`hf-xet`) and in every server-side path that
   reconstructs chunks;
2. only then enable scheme 3 in the writer, or enable it server-side at rest (re-encode xorbs
   in place and serve decoded bytes to clients that predate the decoder; the reconstruction path
   already reads every chunk it serves).

The header `version` byte stays 0: the record layout does not change, only the scheme table.

## 6. What it costs at run time

Per 64 KiB chunk on one core (FINAL-TABLE.md, interleaved arms, three reps):

| class | today (bg4-lz4 / raw) | Fano |
|---|---|---|
| F32 compress / decompress | 72-124 us / 23-30 us | 52-57 us / 13-22 us |
| BF16 compress / decompress | 126-137 us / 27-31 us | 56-58 us / 18 us |
| F16 (stored raw today) | 42-55 us / 3-9 us | 63-73 us / 21-29 us |

Encoder plane and body scratch is thread-local and grows to one segment once per thread; the
per-call allocations are the output vector and a 1 KiB histogram. The decoder writes only into
the caller's `dst`.
