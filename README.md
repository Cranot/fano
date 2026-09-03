# Fano

**Tensor files stored 11–36% smaller than Hugging Face's Xet stores them today, and read back
faster.** Drop-in for a content-addressed store: one new scheme byte, dedup untouched, exact bytes
back or an error. Apache-2.0, one vendored dependency, decoder written twice so the two can be
checked against each other.

The idea fits in a sentence. A floating-point weight is two or four bytes, and nearly all of the
redundancy sits in one of them — the sign and exponent — while the mantissa is close to noise. Code
that byte on its own and pass the rest through untouched. A compressor that sees the element as an
opaque run of bytes averages the two together and finds little in either.

---

## What you get

Measured on real files from the Hub, 64 KiB chunks, one pinned core, medians of five runs, every
output decompressed and compared byte-for-byte. The comparison is xet-core's own `compression_scheme.rs`,
`bg4_prediction.rs` and byte grouping, compiled unmodified against `lz4_flex 0.13.0` and verified
byte-identical to upstream `7af65ba2`.

| file class | Xet today | Fano | bytes | write MB/s | read MB/s | read µs / chunk |
|---|---:|---:|---:|---|---|---|
| BF16 · LLM weights | 1.14× | **1.49×** | **−23%** | 495 → **1,078** | 2,275 → **3,446** | 27 → **18** |
| F32 · embeddings | 1.25× | **1.59×** | **−21%** | 591 → **1,131** | 2,273 → **3,539** | 27 → **18** |
| F32 · audio model | 1.48× | **2.32×** | **−36%** | 571 → **1,210** | 2,285 → **3,277** | 27 → **19** |
| F16 · GGUF weights | stored raw | **1.50×** | **−33%** | memcpy → 375 | memcpy → 920 | ~0 → 71 |
| INT8 weights | stored raw | **1.73×** | **−42%** | memcpy → coded | memcpy → 1,913 | ~0 → 31 |
| FP8 E4M3 weights | stored raw | **1.21×** | **−18%** | memcpy → coded | memcpy → 1,169 | ~0 → 53 |
| GGUF Q8_0 quantised | stored raw | 1.04× | −4% | memcpy → coded | memcpy → coded | — |
| llama.cpp imatrix | 1.45× | 1.26× | **+13%** | — | — | — |

Three things to read off this table.

**On F32 and BF16 there is no trade.** The file is smaller *and* both directions are faster, because
coding one byte plane with a Huffman table is less work than LZ4 searching four for matches it will
not find. A 64 KiB chunk that took Xet 126 µs to write takes Fano 58.

**The "stored raw" rows are free ground.** Xet inspects those files, finds no repeated byte patterns,
and stores them exactly as uploaded. They are not incompressible — they are compressible a different
way. Every byte taken there is a byte nobody was taking, for 30–70 µs per chunk that was previously a
`memcpy`. Whether that is worth it depends on your link: the INT8 row pays for itself below 7 Gb/s,
the FP8 row below 1.7 Gb/s. Above that, store raw.

**Fano loses on importance-matrix files, by 13%.** Their byte planes are near-random but 18% of
4-byte-aligned positions repeat, which is exactly what LZ4 finds and an order-0 coder cannot. The row
is here because a table with only wins is marketing, and because it points at the real answer for that
class: an imatrix is regenerable from the model, the calibration corpus and the llama.cpp version —
the same recipe shape as the section below.

Full numbers, including what was tried and did not work, are in [BENCH.md](BENCH.md).

## Where the bigger saving is, and why it is not in this package yet

Measuring Fano against a week of Hub uploads turned up something larger than any codec: **most
quantised models on the Hub are the deterministic output of a program whose input the Hub already
holds.** A `Q8_0` GGUF is `llama.cpp` run over weights that sit in another repository, and the Hub
records that relationship in its own `base_model:quantized` tags — on 80% of GGUF bytes.

Given the parent, the quantised file does not need storing at all. Running the publisher's own
toolchain on the parent — `convert_hf_to_gguf`, then `llama-quantize` with the publisher's imatrix
and the per-tensor types the published header already lists — **reproduces the published file byte
for byte.** Measured on six files from five publishers and five architectures. Three are exact: a
mradermacher static Q4_K_M (all 4,596,736 superblocks); a bartowski imatrix Q4_K_M on a hybrid-SSM
model (all 19,850,240 superblocks, 152 bytes differing — thirty-eight float32 values one ulp apart
where the converter calls `exp()`); and a Q8_0 written by a *different exporter entirely*, which
llama.cpp's toolchain reproduced across all 113 tensors — the format is a closed form and two
implementations agree on every byte. Two mixture-of-experts models reproduce every fused expert tensor
exactly and land at 99.96% and 96.5% overall; both residuals trace to the quantiser *build* — a type
rule that ignores overrides, and imatrix weighting that moved between versions — so a recipe names the
llama.cpp commit, not just its version. The sixth file did not reproduce because its declared parent
was wrong: a base-plus-instruct merge tagged as a quantisation of the base, caught in one step because
its F32 norms did not match. The toolchain itself is deterministic and hardware-independent: two
builds, one with every SIMD extension off, produced identical bytes.

So the file is a *recipe*: parent, tool versions, output precision, quant type, imatrix, and a type
map that costs nothing because the header carries it. That belongs inside a content-addressed store
as a derived-chunk type, not in a codec — the saving holds only while the parent is held, and it is
bought with ingest compute rather than bytes. Fano is the codec. The recipe layer is measured and
documented but not shipped; the record, every script, and the honest accounting of what resolves
(41–66% of declared parents today) and what does not are in
[`mzip/hfbench`](https://github.com/Cranot/mzip/tree/master/hfbench). If you run a store and want
to talk about that layer, the numbers are there to argue with.

## Using it

**Rust**

```toml
[dependencies]
fano = "0.1"
```

```rust
match fano::compress(&chunk) {
    Some(packed) => store(packed),        // smaller; the stream carries its own length
    None         => store_raw(&chunk),    // too small or incompressible: use your raw path
}
let chunk = fano::decompress(&packed)?;   // exact bytes back, or an error
```

`compress` returns `None` instead of a larger buffer when the scheme does not pay, so it slots into
a store that already has a raw fallback. The length travels inside the payload as a varint — three
bytes on a 64 KiB chunk — so a `fn(&[u8]) -> Result<Vec<u8>>` interface needs no signature change.

**C or C++**

```c
#include "plane_entropy.h"
size_t cap = pe_compress_bound(n);
size_t sz  = pe_compress(dst, cap, src, n);     /* 0 = does not apply, store raw */
size_t got = pe_decompress(out, n, dst, sz);    /* n on success, 0 if malformed  */
```

Header-only C++ is in `plane_entropy.hpp` (namespace `pe`). `make` builds `libplane_entropy.a`;
`make check` runs the suite.

**If you run a content-addressed store**

[`xet-core.patch`](xet-core.patch) is the complete change against xet-core `7af65ba2`: one variant on
`CompressionScheme`, four trait arms, one dependency line. It states what does *not* move — the
chunk record, the hash over uncompressed bytes so dedup is untouched, the existing fallback to
`None` when a scheme fails to shrink. One operational fact matters more than the rest: **readers
must ship before writers**, because the header validator rejects an unknown scheme byte.
[INTEGRATION.md](INTEGRATION.md) gives two selection policies with the measured numbers behind each
and the rollout order.

## Why you can trust the decoder

[SPEC.md](SPEC.md) is byte-exact: stream layout, encoder policy, every rule a decoder must enforce.
[`vectors.txt`](vectors.txt) pins 144 streams by SHA-256, 36 by full bytes, so a second
implementation can prove interoperability rather than assert it.

The suite is 162,799 assertions in three build configurations plus a differential fuzzer, and the
parts that matter most were built to be independent of the code they check:

- **A second decoder written from the spec alone**, sharing no code with the implementation and
  using no Huffman at all, handles 78 of the 144 streams without executing a line of the coder.
- **Every validation rule has a stream constructed to break it**, and the optimised decoder, the
  reference decoder and the C API must all refuse it.
- **The fuzzer requires two implementations to agree** on accept or reject, produce identical bytes
  when both accept, and never touch the guard bands around the destination.
- **The encoder's one decision — how many planes — is integer arithmetic.** A fixed-point log with
  the Miller-Madow correction as an exact constant, so every conforming build emits the same bytes
  regardless of compiler, flags or platform `log2`. The frozen vectors hold across gcc, clang and
  aarch64 because of it.

This suite exists in its current form because three rounds of adversarial review found defects in
the previous one — a fuzzer that counted outcomes instead of asserting them, frozen vectors silently
skipped when absent. Those are fixed. Two findings stay open by choice and are named in
[TESTING.md](TESTING.md), along with the two things the tests do *not* independently establish.

## What it is not for

Text, already-compressed data, or quantised formats with sub-byte fields packed across byte
boundaries. Those planes sit near 8 bits per byte and pass through raw, at which point the stream
costs more than the input by its headers — which is exactly when `compress` returns `None` and tells
you to store raw. Element widths are 2 and 4 bytes; 8-byte elements are handled as two groups.
Minimum input is 64 bytes, and below a few hundred a Huffman table costs more than it saves.

The stream carries no magic, no version, no checksum. Deliberately: the container already records a
scheme id and a length, and a content-addressed store already hashes the chunk. Paying for those
again per chunk is waste.

## Provenance

This came out of measuring [mzip](https://github.com/Cranot/mzip) against what Xet actually does per
64 KiB chunk. mzip keeps its dual licence; this coder is Apache-2.0 so that a storage system can
adopt it without a licensing conversation.

Named for Robert Fano, who in 1951 offered his MIT information theory class a term paper in place of
a final exam: find the most efficient binary code. David Huffman took the option. His algorithm is
the engine inside this package.
