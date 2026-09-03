# Measurements

Machine: 12-core Intel i5-12500, Linux, gcc `-O3 -march=native`, single thread pinned with
`taskset`, medians of 5 reps, 64 KiB chunks, every chunk round-trip verified by full byte compare
outside the timed region. The machine is shared; the load average is stated with every run.
Absolute throughput on it moves up to 40% with the core and its hyperthread neighbour, so the
optimised-vs-reference comparison below is **interleaved on one core** (both arms in the same loop,
alternating) and the ratio is the deliverable, not the absolute figure.

Data: 8 MiB slices of real tensors downloaded from the Hugging Face Hub (safetensors F32 and BF16,
GGUF F16).

## Optimised vs reference implementation

`bench_ref` runs `pe_ref::` (the plain scalar reference in `reference/`) and `pe::` (this package's
implementation) alternately on the same data and asserts the streams are identical. Load 4.4 → 9.6
across the three rounds; round-to-round spread is under 4%.

| slice | reference encode | encode | speedup | reference decode | decode | speedup | bytes |
|---|---:|---:|---:|---:|---:|---:|---|
| BF16 weights | 290–310 MB/s | 639–645 | **2.06–2.22×** | 1,013–1,037 MB/s | 1,784–1,821 | **1.74–1.80×** | identical |
| MiniLM F32 | 293–303 | 707–715 | **2.35–2.42×** | 1,224–1,243 | 2,625–2,672 | **2.14–2.15×** | identical |
| bge-small F32 | 283–285 | 672–680 | **2.37–2.40×** | 939–963 | 1,524–1,589 | **1.62–1.65×** | identical |
| GGUF F16 | 254–258 | 372–375 | **1.44–1.48×** | 644–668 | 914–926 | **1.39–1.42×** | identical |

GGUF F16 gains least because both of its planes compress, so nearly all of its time is inside
huff0 itself rather than in the framing the optimisation removes.

## Ratio

`bench_pe`, same conditions. "Xet Auto" is HuggingFace's own scheme selection compiled from
xet-core against `lz4_flex 0.13.0` and run on the same chunks (see the mzip measurement record
for that harness).

| slice | plane-entropy | Xet Auto | fewer bytes | planes coded |
|---|---:|---:|---:|---|
| BF16 weights | 1.482 | 1.135 | 23.0% | 1 of 2 |
| Qwen2.5 BF16 | 1.495 | 1.158 | 22.5% | 1 of 2 |
| MiniLM F32 | 1.198 | 1.053 | 12.1% | 1 of 4 |
| gpt2 F32 | 1.191 | 1.035 | 13.1% | 1 of 4 |
| bge-small F32 | 2.350 | 1.545 | 34.2% | 3 of 4 |
| GGUF F16 | 1.501 | 1.001 (stored raw) | 33.3% | 2 of 2 |
| GGUF Q8_0 | 1.042 | 1.000 (stored raw) | 4.1% | 2 of 2 |

Xet Auto on this machine compresses those weight chunks at 484–839 MB/s and decodes them at
2.07–2.64 GB/s (Rust, safe decode, per-chunk frame allocation — what the client actually runs).
The optimised coder above encodes at 372–715 MB/s and decodes at 0.9–2.7 GB/s, i.e. the same speed
class, while storing 12–34% fewer bytes.

## Correctness

`make check` (162,038 assertions): round-trips over 6 generators × 27 sizes, the C API, every
truncation of five streams rejected, header bit flips, 10,000 random mutations, 5,000 garbage
streams, segment arithmetic, 144 frozen stream vectors, and — on every input — the optimised coder
byte-identical to the reference plus each decoding the other's stream.

`make check-asan` the same under AddressSanitizer and UndefinedBehaviorSanitizer: 0 failures.
`make test_pe_noavx` (SIMD compiled out): identical bytes, 0 failures.
`make fuzz`: 880k mutated and garbage inputs under both sanitizers, no report.
