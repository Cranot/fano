# plane-entropy stream format, version 1

Status: byte-exact specification of the payload produced by `pe::encode` / `pe_compress` and
accepted by `pe::decode` / `pe_decompress`. A decoder that follows this document interoperates with
the reference implementation; `test_pe` carries stream vectors (`vectors.txt`) that pin the bytes.

## 1. Scope

The payload compresses a byte string of length `n >= 64` whose elements are fixed-width binary
values (16-bit or 32-bit floats or integers). It carries no magic number, no version field and
no uncompressed length: the container that stores it (for example a chunk record in a
content-addressed store, which already records a scheme id and the chunk length) supplies both.
The decoder MUST be given the exact original length `n`.

There is no checksum. Integrity is the container's job (content-addressed stores hash the
uncompressed chunk). The decoder rejects every malformed structure it can detect (section 5),
but a corrupted byte inside a raw plane is by design undetectable at this layer.

## 2. Notation

* `varint(v)` — unsigned LEB128: 7 bits per byte, little-endian groups, high bit set on all but
  the last byte. Values are 64-bit; a decoder MUST reject a varint longer than 10 bytes, one that
  runs past the end of input, and one whose tenth byte has any of bits 1-6 set (those bits would
  be shifted out of a 64-bit value, so two encodings would name the same number).
* `huff0(P)` — the Huffman block produced by `HUF_compress` of FiniteStateEntropy's huff0
  (`third_party/fse/huf_compress.c`) for plane `P`, decoded by `HUF_decompress(dst, L, src, len)`.
  `len == 1` denotes a single-symbol plane: the one byte is the symbol, repeated `L` times (huff0's
  RLE convention). Otherwise the block is a huff0 4-stream Huffman block: a Huffman tree
  description followed by a 6-byte jump table and four bit streams, as produced by huff0 for all
  input sizes through this API. A plane is at most 131072 bytes (`HUF_BLOCKSIZE_MAX`).

## 3. Layout

```
stream   := varint(nseg) segment[nseg]
segment  := varint(seg_n) varint(seg_len) body            ; body is exactly seg_len bytes
body     := k:u8  flags:u8  hflags:u8  varint(clen[j]) for j in 0..k-1
            payload[j] for j in 0..k-1                     ; payload[j] is clen[j] bytes
            tail                                           ; seg_n - L*k bytes, verbatim
```

with `L = seg_n / k` (integer division) and the tail the last `seg_n mod k` input bytes.

* `k` is 2 or 4. Plane `j` is the byte sequence `d[j], d[j+k], d[j+2k], ...` of the segment's
  first `L*k` bytes (byte `j` of every element).
* `flags` bit `j` (0 = least significant) set means plane `j` is huff0-coded: `payload[j] =
  huff0(plane_j)` and `1 <= clen[j] < L`. Clear means the plane is stored verbatim and
  `clen[j] == L`. Bits `>= k` MUST be 0.
* `hflags` MUST equal `flags` in version 1 (reserved to name a different plane codec later).
* Segments cover the input in order without gaps: the sum of `seg_n` equals `n`. Segment sizes
  are chosen by the encoder (section 4); a decoder MUST accept any partition whose segments are
  each `>= 64` bytes and whose planes are each `<= 131072` bytes.

## 4. Encoder (reference behaviour; other encoders MAY differ as long as section 3 holds)

1. Partition the input into segments of `SEGMENT = 262080` bytes. If the remainder after the last
   full segment is between 1 and 63 bytes, fold it into the last segment (so a segment is at most
   262143 bytes and a k = 2 plane at most 131071 bytes). `nseg = ceil(n / 262080)` or one less
   when a fold happened.
2. For each segment choose `k`: compute the order-0 entropy estimate of the byte planes for k = 2
   and k = 4 as `sum over planes of min(8, H(plane) + 0.02) * L` bits, where `H(plane)` is the
   empirical entropy plus the Miller-Madow bias correction `(K - 1) / (2 L ln 2)` (K = number of
   distinct byte values in the plane); choose k = 4 only if its estimate is below 0.99 times the
   k = 2 estimate, otherwise k = 2.
3. For each plane call huff0. If the result is an error, or `>= L` (not compressible), or 0,
   store the plane raw. If the result is 1 (single symbol) or in `[2, L-1]`, store it coded.
4. Emit the body, then the segment header, then the stream header.

The k rule and the segment size are encoder policy. A decoder never depends on them.

The entropy estimate in step 2 is computed in **integer arithmetic** (bits scaled by 2^16, with a
fixed-point base-2 logarithm), so two conforming builds of the same encoder emit the same bytes
regardless of compiler, optimisation level, floating-point contraction or libm. The constants are
`round(2^16 / (2 ln 2)) = 47274` for the bias correction and `round(0.02 * 2^16) = 1311` for the
table allowance, and `k = 4` is chosen iff `100 * est4 < 99 * est2`.

## 5. Decoder validation (all MUST be enforced)

* every varint is well formed (section 2), including the tenth-byte rule.
* `n >= 64`; `nseg >= 1` and `nseg <= n / 64`.
* For each segment: `seg_n >= 64`; the segments' `seg_n` sum to exactly `n`; `seg_len` does not
  exceed the remaining input; after the last segment no bytes remain.
* In the body: `seg_len >= 3`; `k in {2, 4}`; `hflags == flags`; `flags >> k == 0`;
  `L <= 131072`; each `clen[j]` fits in the remaining body; coded planes have `1 <= clen[j] < L`
  and huff0 must return exactly `L`; raw planes have `clen[j] == L`; after the planes exactly
  `seg_n - L*k` tail bytes remain.

The reference decoder never reads outside the input buffer or writes outside `dst[0..n)`; this is
exercised by `test_pe` (every truncation, header bit flips, random mutations, garbage) and
`fuzz_pe` under AddressSanitizer and UndefinedBehaviorSanitizer.

## 6. Limits and non-goals

* Minimum input 64 bytes. Inputs of a few hundred bytes gain little (each coded plane pays a
  Huffman table); stores typically apply the scheme to chunks of tens of KiB.
* Not for text, quantised integer formats with sub-byte fields, or already-compressed data: the
  planes of such data are near 8 bits/byte and pass through raw (the stream then exceeds the
  input by the headers; a store should fall back to its raw scheme, as `bench_pe` does).
* `k` is limited to 2 and 4; 8-byte elements are coded as two 4-byte planes groups implicitly.

## 7. Compatibility with mzip

The body of a segment is byte-identical to mzip's `PLANE_ENTROPY` block payload (mzip >= the
2026-09 scratch line, `mzpe` namespace); `pe::encode` reproduces `mzpe::encode` bit for bit on
every tested input (see README, cross-check).
