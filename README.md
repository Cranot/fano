# Fano

A per-chunk compression scheme for tensor data. It splits each fixed-width element into its byte
planes and entropy-codes each plane separately, because in a floating-point weight almost all of the
redundancy sits in one byte — the sign and exponent — while the mantissa is close to noise. A coder
that treats the element as an opaque run of bytes sees neither.

Measured against what HuggingFace's Xet storage does today, on real files from the Hub, it is
**11–36% smaller on F32 and BF16 weights while being faster to both write and read**, and it takes
**4–42% off eight classes of file that Xet stores uncompressed**.

Apache-2.0. One vendored dependency: huff0 from
[FiniteStateEntropy](https://github.com/Cyan4973/FiniteStateEntropy) (BSD-2-Clause, in
`third_party/`). Nothing else.

---

## What it does, in one paragraph

A BF16 weight is two bytes: one carries a sign and an 8-bit exponent, the other carries mantissa
bits. Across a tensor the exponent byte takes maybe 30 distinct values with a sharply peaked
distribution — 2.7 bits of entropy in an 8-bit byte — while the mantissa byte is nearly uniform at
7.96 bits. Interleaved, they average out and a general-purpose compressor finds little. Separated,
the first is worth coding and the second is worth passing through untouched. That is the whole idea;
everything else is making it fast and making the decoder safe.

## Measured

Both schemes run over identical bytes of real Hub files, alternating per repetition on one pinned
core, medians of at least three runs, every output decompressed and compared against the original.
The comparison is xet-core's own `compression_scheme.rs`, `bg4_prediction.rs` and byte grouping,
compiled unmodified against `lz4_flex 0.13.0` and verified byte-identical to upstream `7af65ba2`.

| class | Xet today | Fano | size | compress | read back |
|---|---:|---:|---|---|---|
| BF16 · LLM weights | 1.1447 | **1.4885** | −23.1% | 2.18× faster | 1.51× faster |
| F32 · embeddings | 1.2522 | **1.5872** | −21.1% | 1.91× faster | 1.56× faster |
| F32 · audio | 1.4785 | **2.3209** | −36.3% | 2.12× faster | 1.43× faster |
| F16 · LLM weights | 1.0000 | **1.3230** | −24.4% | — | — |
| FP8 E4M3 | 1.0000 | **1.2133** | −17.6% | — | — |
| INT8 | 1.0000 | **1.7272** | −42.1% | — | — |
| INT32 | 1.0000 | **1.1031** | −9.3% | — | — |
| GGUF · F16 | 1.0006 | **1.5012** | −33.3% | — | — |

The rows at ratio 1.0000 are the interesting ones: Xet inspects those files, finds no repeated byte
patterns, and stores them exactly as uploaded. They are not incompressible — they are compressible a
different way, and every byte taken there is free. Where the last two columns are blank, today's
scheme is doing a `memcpy`, so there is no decode to be faster than; the honest comparison is that
Fano spends 20–60 µs per 64 KiB chunk to remove bytes that are currently not removed at all.

**On the nine F32 and BF16 classes there is no trade at all**: smaller *and* quicker in both
directions. Elsewhere it does more work to get the file smaller, which still arrives sooner over any
link below several gigabits per second, because a smaller file is a shorter download.

Full numbers, including the classes where it loses, are in [BENCH.md](BENCH.md).

## Using it

### Rust

```toml
[dependencies]
fano = "0.1"
```

```rust
match fano::compress(&chunk) {
    Some(packed) => store(packed),           // smaller; the stream carries its own length
    None         => store_raw(&chunk),       // too small or incompressible: use your raw path
}

let chunk = fano::decompress(&packed)?;      // exact bytes back, or an error
```

`compress` returns `None` rather than a larger buffer when the scheme does not pay, so it drops
into a store that already has a raw fallback. The uncompressed length travels inside the payload as
a varint, which costs three bytes on a 64 KiB chunk and means a caller with a
`fn(&[u8]) -> Result<Vec<u8>>` shaped interface needs no signature change.

### C or C++

```c
#include "plane_entropy.h"

size_t cap = pe_compress_bound(n);
size_t sz  = pe_compress(dst, cap, src, n);          /* 0 = does not apply, store raw */
size_t got = pe_decompress(out, n, dst, sz);         /* returns n, or 0 if malformed */
```

Header-only C++ is available too (`plane_entropy.hpp`, namespace `pe`). `make check` builds and runs
the suite; `make` alone produces `libplane_entropy.a`.

### Adopting it in a content-addressed store

[`xet-core.patch`](xet-core.patch) is the complete change against xet-core main `7af65ba2`: one
variant on `CompressionScheme`, the four trait arms, one dependency line. It also states what does
*not* move — the chunk record layout, the chunk hash taken over uncompressed bytes so deduplication
is untouched, and the existing fallback to `None` when a scheme fails to shrink a chunk. It gives
two selection policies with the measured numbers behind each, and a rollout order that respects the
fact that an unknown scheme byte is rejected by the header validator, so readers must ship before
writers.

## The format

[SPEC.md](SPEC.md) is a byte-exact specification: the stream layout, the encoder's policy, and every
rule a decoder must enforce. A decoder written from it interoperates with this implementation, and
`vectors.txt` pins 144 streams so a second implementation can prove it.

The stream carries no magic number, no version field and no checksum. That is deliberate: the
container already records a scheme id and a length, and a content-addressed store already hashes the
chunk. Paying for those again per chunk would be waste.

## Testing

`make check && make check-asan && make check-nosimd && make fuzz` — 162,799 assertions in each of
three build configurations, plus a differential fuzzer.

What the tests establish, and what they rest on, is written out in [TESTING.md](TESTING.md),
including the two things they do *not* independently verify. Briefly:

- **An independent decoder.** `test_stream.hpp` contains a parser and decoder written from the spec
  alone, sharing no code with the implementation and using no Huffman at all. It handles 78 of the
  144 round-trip streams and checks them without executing a line of the coder.
- **Constructed malformed streams.** Every validation rule has at least one stream that breaks it,
  and each must be refused by the optimised decoder, the reference decoder and the C API alike.
- **Frozen streams.** 144 entries pinned by SHA-256, 36 of them by full bytes. The digest is
  implemented in the test and self-checked against the published FIPS vectors, so a broken digest
  cannot quietly make the check vacuous.
- **Differential fuzzing.** Two implementations must agree on accept or reject, produce identical
  bytes when both accept, and never touch the guard bands around the destination.

The suite exists because three rounds of adversarial review found real defects in it — a fuzzer that
counted outcomes instead of asserting them, frozen vectors that were skipped when absent, a
portability build that left half the vector paths compiled in. Those are fixed; two findings remain
open by choice and are named in TESTING.md.

## Reproducibility

The encoder's choice of plane count is computed in integer arithmetic — a fixed-point logarithm,
with the Miller-Madow correction as an exact constant — so every conforming build makes the same
choice and emits the same bytes. Floating point there would have made the output depend on
compiler flags, on x87 excess precision and on the platform's `log2`. The frozen vectors hold across
compilers and architectures because of it.

## Limits

- Not for text, for already-compressed data, or for quantised formats with sub-byte fields packed
  across byte boundaries. The planes of such data sit near 8 bits per byte and pass through raw,
  and the stream then exceeds the input by its headers — the caller should fall back, as `compress`
  returning `None` tells it to.
- Element widths of 2 and 4 bytes. Single-byte formats are coded as one stream; 8-byte elements are
  handled as two 4-byte groups.
- Minimum input 64 bytes. Below a few hundred bytes a Huffman table costs more than it saves.
- It loses on one measured class: llama.cpp importance-matrix files, by 1.2%, because their content
  repeats in a way that suits an LZ scheme and not an order-0 one.

## Provenance

The scheme came out of measuring [mzip](https://github.com/Cranot/mzip) against what Xet actually
does per 64 KiB chunk. mzip keeps its dual licence; this coder is Apache-2.0 so that a storage
system can adopt it without a licensing conversation.

Named for Robert Fano, who in 1951 offered his MIT information theory class a term paper in place of
a final exam: find the most efficient binary code. David Huffman took the option. His algorithm is
the engine inside this package.
