# What the tests establish, and what they rest on

Run everything with `make check && make check-asan && make check-nosimd && make fuzz`. This file
says what each part proves, and — the part that usually goes unwritten — what it does **not**.

## The three build configurations

| target | build | what it adds |
|---|---|---|
| `make check` | `-O3`, SIMD as the host allows | the suite |
| `make check-asan` | `-O1 -g -fsanitize=address,undefined`, C API compiled in | out-of-bounds and undefined behaviour |
| `make check-nosimd` | `-O2 -DPE_NO_SIMD -mno-avx2 -mno-avx` | every AVX2 **and SSE2** branch compiled out; the scalar path must produce the same bytes, including the frozen vectors |

All three run the same `test_pe`, which exits non-zero on any failure.

## The four oracles, from strongest to weakest

**1. An independent decoder (no code from the coder, no huff0).** `test_stream.hpp` contains a
stream parser and a decoder written from `SPEC.md` alone. It handles every stream whose planes are
all either raw or single-symbol (huff0's RLE form, one byte repeated), which is 78 of the 144
round-trip streams. For those, the decoded bytes are checked without executing one line of
`plane_entropy.hpp`, its reference, or huff0. This covers varint parsing, segment framing, the
plane interleave, the tail, and every length rule.

**2. Constructed malformed streams.** Every rule in `SPEC.md` section 5 has at least one stream that
breaks it, built by the same independent writer, and each must be refused by the optimised decoder,
the reference decoder **and** the C API. Where a stream can only break one rule together with
another (an `nseg` of 0 also makes the segment sum impossible), the case is labelled for what it is.
Seven of the cases carry a complete, otherwise-valid body, so removing the guard under test would
let the stream through rather than failing later for an unrelated reason.

**3. Frozen streams.** `vectors.txt` holds, for each of the 144 generator/size combinations, the
stream length and its SHA-256; for the 36 streams from inputs of 1024 bytes or less it also holds
the full stream in hex. The SHA-256 is implemented in the test file and self-tested against the
three published FIPS 180-4 vectors, so a broken digest cannot make the check vacuous. The file is
**required**: a missing `vectors.txt` is a failure, not a skip. The entry count and the key set are
asserted in both directions. Regeneration happens only under `--write-vectors`.

**4. Differential fuzzing.** `fuzz_pe` mutates encoded streams two ways — unstructured (flips,
truncation, insertion, deletion, swaps) and field-aware (the stream is parsed and one named field is
set to a boundary value) — and requires on every iteration that the optimised decoder and the
reference agree on accept/reject, that they produce identical bytes when both accept, that the C API
returns `n` exactly when the C++ API accepts, and that 64-byte guard bands either side of the
destination are untouched. It exits non-zero on any violation.

## What is NOT independently verified, stated plainly

* **huff0's own behaviour.** The optimised encoder and the reference both call the vendored huff0
  (BSD-2, `third_party/fse`, unmodified from the commit named in `UPSTREAM.txt`). A stream with a
  real Huffman block is therefore checked against frozen bytes and digests, not against a second
  Huffman implementation. Writing an independent huff0 decoder for the test would close this; the
  36 byte-frozen vectors and the 144 digests are what pins it today.
* **The reference is a second implementation, not a second specification.** It was written from the
  same document by the same author. It catches optimisation defects — a wrong SIMD lane, a fused
  histogram that disagrees with a plain one, a CTable path that diverges from `HUF_compress` — which
  is what it exists for. It is not evidence that the format itself is well designed.
* **The encoder's choice of `k`** is heuristic and is not asserted to be optimal, only to be
  reproducible: it is computed in integer arithmetic (`SPEC.md` section 4), so every conforming
  build makes the same choice and the frozen vectors hold across compilers and architectures.
* **Timing and ratio** are not tested here; `bench_pe` measures them and `BENCH.md` records them.

## Counts, as of the last run on the development machine

162,799 assertions pass in each of the three configurations. 144 round-trip streams (78 of them
also decoded independently), 28 edge shapes, 160,181 truncations rejected, 368 structural bit flips
all rejected, 98,904 payload and tail flips (13,220 rejected, the rest undetectable by design), and
about 24,000 fuzz iterations per 90 seconds with roughly a quarter of them field-aware.
