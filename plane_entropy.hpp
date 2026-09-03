// plane_entropy.hpp - byte-plane order-0 entropy coder for tensor data (header-only, C++17).
//
// Copyright 2025 Cranot
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions
// and limitations under the License.
//
// What it does: splits the input into k byte planes (k = 2 for 16-bit elements such as BF16/F16,
// k = 4 for 32-bit elements such as F32), codes each plane with its own order-0 Huffman table
// (huff0 from FiniteStateEntropy, BSD-2-Clause) and stores planes that do not compress as raw
// bytes. On floating-point weights the exponent plane carries almost all the redundancy; the
// mantissa planes are near-incompressible and pass through at memcpy speed. See SPEC.md for the
// byte-exact stream layout and README.md for measurements against HuggingFace Xet's schemes.
//
// The stream is a PAYLOAD: it carries neither a magic number nor the uncompressed length. The
// caller stores the uncompressed length (Xet does this per chunk) and passes it to decode().
//
// This file is the OPTIMISED implementation. `reference/plane_entropy_ref.hpp` is the plain
// scalar one, kept as the executable definition of the format; `test_pe` asserts that the two
// produce identical streams on every input it generates, so the optimisations below can never
// silently change the format. What they do (all measured, all byte-neutral):
//   * one 4-way histogram pass feeds BOTH the plane-count choice and huff0's Huffman table
//     (the plain version walks the data three times: twice for the estimate, once inside huff0);
//   * huff0 is driven through its public CTable API, replicating HUF_compress()'s decisions
//     exactly, so the table is built from the histogram already computed;
//   * planes are de-interleaved and re-interleaved with SSE2/AVX2 where available;
//   * scratch buffers are reused per thread, the output is assembled with one copy per segment,
//     and on decode a stored plane is read straight from the payload with no copy.
//
// Dependency: huf.h / huff0 (third_party/fse). No other library.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef HUF_STATIC_LINKING_ONLY
#define HUF_STATIC_LINKING_ONLY   // HUF_buildCTable / HUF_writeCTable / HUF_compress4X_usingCTable
#endif
#include "huf.h"
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace pe {

// ---- format constants (see SPEC.md) --------------------------------------------------------
static const size_t MIN_INPUT   = 64;       // encode() rejects smaller inputs; decode() rejects smaller segments
static const size_t SEGMENT     = 262080;   // 256 KiB - 64: every plane of a (possibly tail-folded) segment fits huff0's 128 KiB block limit
static const size_t TAIL_FOLD   = 64;       // a remainder shorter than this is folded into the last segment
static const unsigned MAX_K     = 4;

// ---- LEB128 varints ------------------------------------------------------------------------
inline size_t put_varint(uint8_t* o, uint64_t v) { size_t n = 0; while (v >= 0x80) { o[n++] = (uint8_t)(v | 0x80); v >>= 7; } o[n++] = (uint8_t)v; return n; }
inline bool get_varint(const uint8_t*& p, const uint8_t* e, uint64_t& v) {
    v = 0; int sh = 0;
    while (p < e && sh < 64) {
        uint8_t b = *p++;
        // the tenth byte carries only bit 63: any higher payload bit would be shifted out, so two
        // different encodings would name the same value. Reject instead of wrapping (SPEC.md 2).
        if (sh == 63 && (b & 0x7e) != 0) return false;
        v |= (uint64_t)(b & 0x7f) << sh;
        if (!(b & 0x80)) return true;
        sh += 7;
    }
    return false;
}

// ---- per-thread scratch --------------------------------------------------------------------
struct Scratch {
    uint32_t h4[4][256];
    uint32_t h2[2][256];
    std::vector<uint8_t> planes, body;
    uint8_t hdr[3 + 10 * MAX_K];
    size_t hdrlen = 0;
};
inline Scratch& scratch() { static thread_local Scratch s; return s; }

// ---- de-interleave (gather): plane_j[i] = d[i*k + j] ---------------------------------------
inline void gather2(const uint8_t* d, size_t L, uint8_t* p0, uint8_t* p1) {
    size_t i = 0;
#if defined(__AVX2__) && !defined(PE_NO_SIMD)
    const __m256i m = _mm256_set1_epi16(0x00FF);
    for (; i + 32 <= L; i += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(d + 2 * i));
        __m256i b = _mm256_loadu_si256((const __m256i*)(d + 2 * i + 32));
        __m256i e = _mm256_packus_epi16(_mm256_and_si256(a, m), _mm256_and_si256(b, m));
        __m256i o = _mm256_packus_epi16(_mm256_srli_epi16(a, 8), _mm256_srli_epi16(b, 8));
        _mm256_storeu_si256((__m256i*)(p0 + i), _mm256_permute4x64_epi64(e, 0xD8));
        _mm256_storeu_si256((__m256i*)(p1 + i), _mm256_permute4x64_epi64(o, 0xD8));
    }
#endif
    for (; i < L; i++) { p0[i] = d[2 * i]; p1[i] = d[2 * i + 1]; }
}
inline void gather4(const uint8_t* d, size_t L, uint8_t* p0, uint8_t* p1, uint8_t* p2, uint8_t* p3) {
    size_t i = 0;
#if defined(__AVX2__) && !defined(PE_NO_SIMD)
    const __m256i m16 = _mm256_set1_epi32(0x0000FFFF), m8 = _mm256_set1_epi16(0x00FF);
    for (; i + 32 <= L; i += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(d + 4 * i));
        __m256i b = _mm256_loadu_si256((const __m256i*)(d + 4 * i + 32));
        __m256i c = _mm256_loadu_si256((const __m256i*)(d + 4 * i + 64));
        __m256i e = _mm256_loadu_si256((const __m256i*)(d + 4 * i + 96));
        __m256i lo_ab = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_and_si256(a, m16), _mm256_and_si256(b, m16)), 0xD8);
        __m256i hi_ab = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_srli_epi32(a, 16), _mm256_srli_epi32(b, 16)), 0xD8);
        __m256i lo_ce = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_and_si256(c, m16), _mm256_and_si256(e, m16)), 0xD8);
        __m256i hi_ce = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_srli_epi32(c, 16), _mm256_srli_epi32(e, 16)), 0xD8);
        _mm256_storeu_si256((__m256i*)(p0 + i), _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_and_si256(lo_ab, m8), _mm256_and_si256(lo_ce, m8)), 0xD8));
        _mm256_storeu_si256((__m256i*)(p1 + i), _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_srli_epi16(lo_ab, 8), _mm256_srli_epi16(lo_ce, 8)), 0xD8));
        _mm256_storeu_si256((__m256i*)(p2 + i), _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_and_si256(hi_ab, m8), _mm256_and_si256(hi_ce, m8)), 0xD8));
        _mm256_storeu_si256((__m256i*)(p3 + i), _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_srli_epi16(hi_ab, 8), _mm256_srli_epi16(hi_ce, 8)), 0xD8));
    }
#endif
    for (; i < L; i++) { p0[i] = d[4 * i]; p1[i] = d[4 * i + 1]; p2[i] = d[4 * i + 2]; p3[i] = d[4 * i + 3]; }
}
// ---- re-interleave (scatter): dst[i*k + j] = plane_j[i] ------------------------------------
inline void scatter2(uint8_t* dst, const uint8_t* a, const uint8_t* b, size_t L) {
    size_t i = 0;
#if defined(__AVX2__) && !defined(PE_NO_SIMD)
    for (; i + 32 <= L; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i)), vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i lo = _mm256_unpacklo_epi8(va, vb), hi = _mm256_unpackhi_epi8(va, vb);
        _mm256_storeu_si256((__m256i*)(dst + 2 * i),      _mm256_permute2x128_si256(lo, hi, 0x20));
        _mm256_storeu_si256((__m256i*)(dst + 2 * i + 32), _mm256_permute2x128_si256(lo, hi, 0x31));
    }
#elif defined(__SSE2__) && !defined(PE_NO_SIMD)
    for (; i + 16 <= L; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + i)), vb = _mm_loadu_si128((const __m128i*)(b + i));
        _mm_storeu_si128((__m128i*)(dst + 2 * i),      _mm_unpacklo_epi8(va, vb));
        _mm_storeu_si128((__m128i*)(dst + 2 * i + 16), _mm_unpackhi_epi8(va, vb));
    }
#endif
    for (; i < L; i++) { dst[2 * i] = a[i]; dst[2 * i + 1] = b[i]; }
}
inline void scatter4(uint8_t* dst, const uint8_t* a, const uint8_t* b, const uint8_t* c, const uint8_t* d, size_t L) {
    size_t i = 0;
#if defined(__AVX2__) && !defined(PE_NO_SIMD)
    for (; i + 32 <= L; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i)), vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vc = _mm256_loadu_si256((const __m256i*)(c + i)), vd = _mm256_loadu_si256((const __m256i*)(d + i));
        __m256i ab_lo = _mm256_unpacklo_epi8(va, vb), ab_hi = _mm256_unpackhi_epi8(va, vb);
        __m256i cd_lo = _mm256_unpacklo_epi8(vc, vd), cd_hi = _mm256_unpackhi_epi8(vc, vd);
        __m256i q0 = _mm256_unpacklo_epi16(ab_lo, cd_lo), q1 = _mm256_unpackhi_epi16(ab_lo, cd_lo);
        __m256i q2 = _mm256_unpacklo_epi16(ab_hi, cd_hi), q3 = _mm256_unpackhi_epi16(ab_hi, cd_hi);
        _mm256_storeu_si256((__m256i*)(dst + 4 * i),       _mm256_permute2x128_si256(q0, q1, 0x20));
        _mm256_storeu_si256((__m256i*)(dst + 4 * i + 32),  _mm256_permute2x128_si256(q2, q3, 0x20));
        _mm256_storeu_si256((__m256i*)(dst + 4 * i + 64),  _mm256_permute2x128_si256(q0, q1, 0x31));
        _mm256_storeu_si256((__m256i*)(dst + 4 * i + 96),  _mm256_permute2x128_si256(q2, q3, 0x31));
    }
#elif defined(__SSE2__) && !defined(PE_NO_SIMD)
    for (; i + 16 <= L; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + i)), vb = _mm_loadu_si128((const __m128i*)(b + i));
        __m128i vc = _mm_loadu_si128((const __m128i*)(c + i)), vd = _mm_loadu_si128((const __m128i*)(d + i));
        __m128i ab_lo = _mm_unpacklo_epi8(va, vb), ab_hi = _mm_unpackhi_epi8(va, vb);
        __m128i cd_lo = _mm_unpacklo_epi8(vc, vd), cd_hi = _mm_unpackhi_epi8(vc, vd);
        _mm_storeu_si128((__m128i*)(dst + 4 * i),      _mm_unpacklo_epi16(ab_lo, cd_lo));
        _mm_storeu_si128((__m128i*)(dst + 4 * i + 16), _mm_unpackhi_epi16(ab_lo, cd_lo));
        _mm_storeu_si128((__m128i*)(dst + 4 * i + 32), _mm_unpacklo_epi16(ab_hi, cd_hi));
        _mm_storeu_si128((__m128i*)(dst + 4 * i + 48), _mm_unpackhi_epi16(ab_hi, cd_hi));
    }
#endif
    for (; i < L; i++) { dst[4 * i] = a[i]; dst[4 * i + 1] = b[i]; dst[4 * i + 2] = c[i]; dst[4 * i + 3] = d[i]; }
}

// ---- one pass over the data, four histograms by position mod 4 -----------------------------
inline void hist4(const uint8_t* d, size_t n4, uint32_t h[4][256]) {
    uint32_t hb[4][256];
    std::memset(h, 0, sizeof(uint32_t) * 4 * 256);
    std::memset(hb, 0, sizeof(hb));
    size_t q = 0;
    for (; q + 8 <= n4; q += 8) {   // two table sets: fewer store-to-load stalls on repeated symbols
        h[0][d[q]]++;   h[1][d[q+1]]++; h[2][d[q+2]]++; h[3][d[q+3]]++;
        hb[0][d[q+4]]++; hb[1][d[q+5]]++; hb[2][d[q+6]]++; hb[3][d[q+7]]++;
    }
    for (; q + 4 <= n4; q += 4) { h[0][d[q]]++; h[1][d[q+1]]++; h[2][d[q+2]]++; h[3][d[q+3]]++; }
    for (unsigned j = 0; j < 4; j++) for (unsigned c = 0; c < 256; c++) h[j][c] += hb[j][c];
}

// ---- k selection: order-0 estimate of the planes (encoder-side heuristic only; k is stored) ----
//
// All of it is integer arithmetic in Q16 (bits x 65536). Floating point here would make the chosen
// k, and therefore the stream bytes, depend on the compiler's evaluation model (x87 excess
// precision, FMA contraction, -ffast-math, the libm's log2), so two conforming builds could emit
// different streams for the same input. Decoders never depend on k, but frozen test vectors and
// reproducible encoders do. The integer form below is exact on every C++17 implementation.
static const int64_t PE_Q = 16;                       // fractional bits
static const int64_t PE_MM_Q16 = 47274;               // round(2^16 / (2 ln 2)): Miller-Madow, per extra symbol
static const int64_t PE_TABLE_Q16 = 1311;             // round(0.02 * 2^16): per-symbol table-cost allowance
// round-toward-zero log2(x) in Q16 for x >= 1, by repeated squaring of the mantissa. Deterministic.
inline uint32_t log2_q16(uint32_t x) {
    if (x <= 1) return 0;
    int e = 0; { uint32_t t = x; while (t >>= 1) e++; }        // floor(log2(x)), no builtins
    uint64_t y = (uint64_t)x << (31 - e);                      // mantissa in Q31, [2^31, 2^32)
    uint32_t r = (uint32_t)e << PE_Q;
    for (int i = 0; i < (int)PE_Q; i++) {
        y = (y * y) >> 31;                                     // Q31, now in [2^31, 2^33)
        if (y >= (1ull << 32)) { y >>= 1; r |= 1u << (PE_Q - 1 - i); }
    }
    return r;
}
// Estimated coded size of one plane, in bits x 2^16: L*log2(L) - sum c*log2(c), plus the
// Miller-Madow bias correction and the table allowance, capped at 8 bits per byte.
// Miller-Madow: the empirical entropy of a short plane underestimates the true entropy by about
// (K-1)/(2 L ln 2) bits/symbol, which favoured k = 4 (shorter planes) on small inputs. Measured on
// 8 real weight slices at 4 KiB chunks: 1,897/16,384 wrong picks (0.11% lost) without the
// correction, 3 with it; no difference at 16 KiB and above.
inline int64_t plane_bits_hist_q16(const uint32_t* h, size_t L) {
    if (!L) return 0;
    int64_t sum = 0; unsigned K = 0;
    for (unsigned c = 0; c < 256; c++) if (h[c]) { K++; sum += (int64_t)h[c] * (int64_t)log2_q16(h[c]); }
    int64_t bits = (int64_t)L * (int64_t)log2_q16((uint32_t)L) - sum;
    if (K > 1) bits += (int64_t)(K - 1) * PE_MM_Q16;
    bits += (int64_t)L * PE_TABLE_Q16;
    const int64_t cap = (int64_t)L * (8 << PE_Q);              // a raw plane costs 8 bits/byte
    return bits < cap ? bits : cap;
}
inline double plane_bits_hist(const uint32_t* h, size_t L) { return (double)plane_bits_hist_q16(h, L) / 65536.0; }
// Fills s.h4 / s.h2 for this segment and returns the chosen k.
inline unsigned choose_k_hist(const uint8_t* d, size_t n, Scratch& s) {
    const size_t L4 = n / 4, L2 = n / 2;
    hist4(d, 4 * L4, s.h4);
    for (unsigned c = 0; c < 256; c++) { s.h2[0][c] = s.h4[0][c] + s.h4[2][c]; s.h2[1][c] = s.h4[1][c] + s.h4[3][c]; }
    for (size_t q = 4 * L4; q < 2 * L2; q++) s.h2[q & 1][d[q]]++;   // the bytes k=2 covers but k=4 does not
    const int64_t b2 = plane_bits_hist_q16(s.h2[0], L2) + plane_bits_hist_q16(s.h2[1], L2);
    int64_t b4 = 0; for (unsigned j = 0; j < 4; j++) b4 += plane_bits_hist_q16(s.h4[j], L4);
    return (b4 * 100 < b2 * 99) ? 4u : 2u;   // k=4 iff it is at least 1% smaller; ties go to 2 planes
}
// Plain entry point (recomputes the histograms); kept for callers and for the reference tests.
inline int64_t plane_bits_q16(const uint8_t* d, size_t n, unsigned k) {
    size_t L = n / k; int64_t bits = 0;
    std::vector<uint32_t> h(256);
    for (unsigned j = 0; j < k; j++) {
        std::fill(h.begin(), h.end(), 0u);
        for (size_t i = 0; i < L; i++) h[d[i * k + j]]++;
        bits += plane_bits_hist_q16(h.data(), L);
    }
    return bits;
}
inline double plane_bits(const uint8_t* d, size_t n, unsigned k) { return (double)plane_bits_q16(d, n, k) / 65536.0; }
inline unsigned choose_k(const uint8_t* d, size_t n) {
    return (plane_bits_q16(d, n, 4) * 100 < plane_bits_q16(d, n, 2) * 99) ? 4u : 2u;
}

// ---- huff0 from a precomputed histogram -----------------------------------------------------
// Returns exactly what HUF_compress(dst, cap, src, L) would: 0 (not compressible), 1 (RLE, the
// symbol in dst[0]), an error code, or the compressed size. The decision sequence is copied from
// HUF_compress_internal: RLE when one symbol covers the plane; the largest<=(L>>7)+4 heuristic;
// optimalTableLog -> buildCTable -> writeCTable; hSize+12 >= L; and total >= L-1 after coding.
inline size_t huf_from_hist(uint8_t* dst, size_t cap, const uint8_t* src, size_t L, const uint32_t* cnt) {
    if (!L || !cap) return 0;
    uint32_t largest = 0; unsigned maxSym = 0;
    for (unsigned c = 0; c < 256; c++) { if (cnt[c] > largest) largest = cnt[c]; if (cnt[c]) maxSym = c; }
    if (largest == L) { dst[0] = src[0]; return 1; }          // single symbol: huff0's RLE result
    if (largest <= (L >> 7) + 4) return 0;                    // huff0's "probably not compressible" heuristic
    unsigned huffLog = HUF_optimalTableLog(HUF_TABLELOG_DEFAULT, L, maxSym);
    HUF_CREATE_STATIC_CTABLE(ct, 255);
    std::memset(cthb, 0, sizeof(cthb));                       // zero unused symbols, as huff0 does
    size_t maxBits = HUF_buildCTable(ct, cnt, maxSym, huffLog);
    if (HUF_isError(maxBits)) return maxBits;
    huffLog = (unsigned)maxBits;
    size_t hSize = HUF_writeCTable(dst, cap, ct, maxSym, huffLog);
    if (HUF_isError(hSize)) return hSize;
    if (hSize + 12ul >= L) return 0;
    // Exact early exit: the 4-stream payload is at least 6 (jump table) + ceil(bits/8), so when
    // hSize + 6 + estimate >= L-1 the reference call is bound to return 0. Saves coding the plane.
    size_t est = HUF_estimateCompressedSize(ct, cnt, maxSym);
    if (hSize + 6 + est >= L - 1) return 0;
    size_t cSize = HUF_compress4X_usingCTable(dst + hSize, cap - hSize, src, L, ct);
    if (HUF_isError(cSize)) return cSize;
    if (cSize == 0) return 0;
    if (hSize + cSize >= L - 1) return 0;
    return hSize + cSize;
}

// ---- one segment ---------------------------------------------------------------------------
// Layout: [k:1][flags:1][hflags:1][clen varint x k][plane payloads][tail bytes]
//   flags  bit j = plane j is huff0-coded (else raw, clen == L)
//   hflags must equal flags in this version (reserved for other plane codecs)
inline bool encode_segment_hist(std::vector<uint8_t>& out, const uint8_t* d, size_t n, unsigned k,
                                const uint32_t (*h)[256], Scratch& s) {
    if (n < MIN_INPUT || (k != 2 && k != 4)) return false;
    const size_t L = n / k;   // the n - L*k tail bytes are appended verbatim below
    if (L > HUF_BLOCKSIZE_MAX) return false;
    const size_t per = (size_t)HUF_compressBound(L) + 16;
    if (s.planes.size() < k * L) s.planes.resize(k * L);
    if (s.body.size() < k * per) s.body.resize(k * per);
    uint8_t* pl = s.planes.data();
    uint8_t* body = s.body.data();
    if (k == 2) gather2(d, L, pl, pl + L); else gather4(d, L, pl, pl + L, pl + 2 * L, pl + 3 * L);
    uint8_t flags = 0; size_t clen[MAX_K] = {0, 0, 0, 0}, off = 0;
    for (unsigned j = 0; j < k; j++) {
        const uint8_t* pj = pl + (size_t)j * L;
        size_t r = huf_from_hist(body + off, s.body.size() - off, pj, L, h[j]);
        // r == 0: not compressible -> raw. r == 1: single-symbol plane (RLE; the byte is in body[off]).
        if (!HUF_isError(r) && r >= 1 && r < L) { flags |= (uint8_t)(1u << j); clen[j] = r; }
        else { std::memcpy(body + off, pj, L); clen[j] = L; }
        off += clen[j];
    }
    size_t m = 0;
    s.hdr[m++] = (uint8_t)k; s.hdr[m++] = flags; s.hdr[m++] = flags;
    for (unsigned j = 0; j < k; j++) m += put_varint(s.hdr + m, clen[j]);
    s.hdrlen = m;
    out.insert(out.end(), s.hdr, s.hdr + m);
    out.insert(out.end(), body, body + off);
    out.insert(out.end(), d + L * k, d + n);
    return true;
}
inline bool encode_segment(std::vector<uint8_t>& out, const uint8_t* d, size_t n, unsigned k) {
    if (n < MIN_INPUT || (k != 2 && k != 4)) return false;
    Scratch& s = scratch();
    const size_t L4 = n / 4, L2 = n / 2;
    hist4(d, 4 * L4, s.h4);
    for (unsigned c = 0; c < 256; c++) { s.h2[0][c] = s.h4[0][c] + s.h4[2][c]; s.h2[1][c] = s.h4[1][c] + s.h4[3][c]; }
    for (size_t q = 4 * L4; q < 2 * L2; q++) s.h2[q & 1][d[q]]++;
    return encode_segment_hist(out, d, n, k, k == 4 ? s.h4 : s.h2, s);
}

// Decodes one segment into dst[0..n). Returns false on any malformed input; never reads past src+sz.
inline bool decode_segment(uint8_t* dst, size_t n, const uint8_t* src, size_t sz) {
    const uint8_t* p = src; const uint8_t* e = src + sz;
    if (sz < 3 || n < MIN_INPUT) return false;
    const unsigned k = *p++; const uint8_t flags = *p++; const uint8_t hflags = *p++;
    if (k != 2 && k != 4) return false;
    if (hflags != flags || (flags >> k) != 0) return false;
    const size_t L = n / k, tail = n - L * k;
    if (L > HUF_BLOCKSIZE_MAX) return false;
    uint64_t cl[MAX_K];
    for (unsigned j = 0; j < k; j++) if (!get_varint(p, e, cl[j])) return false;
    Scratch& s = scratch();
    if (s.planes.size() < k * L) s.planes.resize(k * L);
    uint8_t* sc = s.planes.data();
    const uint8_t* pl[MAX_K] = {nullptr, nullptr, nullptr, nullptr};
    for (unsigned j = 0; j < k; j++) {
        if (cl[j] > (uint64_t)(e - p)) return false;
        if (flags & (1u << j)) {
            if (cl[j] == 0 || cl[j] >= L) return false;       // a coded plane is strictly shorter than the plane
            uint8_t* o = sc + (size_t)j * L;
            size_t r = HUF_decompress(o, L, p, (size_t)cl[j]);
            if (HUF_isError(r) || r != L) return false;
            pl[j] = o;
        } else {
            if (cl[j] != L) return false;
            pl[j] = p;                                        // stored plane: read in place, no copy
        }
        p += cl[j];
    }
    if ((size_t)(e - p) != tail) return false;
    if (k == 2) scatter2(dst, pl[0], pl[1], L); else scatter4(dst, pl[0], pl[1], pl[2], pl[3], L);
    if (tail) std::memcpy(dst + L * k, p, tail);
    return true;
}

// ---- whole input: segments ------------------------------------------------------------------
// Layout: [nseg varint] { [seg_n varint][seg_len varint][segment] } x nseg
// Upper bound on the stream size. Returns SIZE_MAX when the bound itself would overflow, and
// encode() rejects such an n rather than emitting a stream the bound does not cover.
inline size_t compress_bound(size_t n) {
    const size_t segs = n / SEGMENT + 1;
    if (segs > (SIZE_MAX - 32) / 24) return SIZE_MAX;
    const size_t ovh = 32 + 24 * segs;
    if (n > SIZE_MAX - ovh) return SIZE_MAX;
    return n + ovh;
}

inline std::vector<uint8_t> encode(const uint8_t* d, size_t n) {
    std::vector<uint8_t> out;
    if (n < MIN_INPUT || compress_bound(n) == SIZE_MAX) return out;
    size_t nseg = 0;
    for (size_t off = 0; off < n; ) { size_t sn = std::min(SEGMENT, n - off); if (n - off - sn > 0 && n - off - sn < TAIL_FOLD) sn = n - off; off += sn; nseg++; }
    out.reserve(compress_bound(n));
    uint8_t vb[16]; size_t m = put_varint(vb, nseg); out.insert(out.end(), vb, vb + m);
    Scratch& s = scratch();
    std::vector<uint8_t> seg;
    for (size_t off = 0; off < n; ) {
        size_t sn = std::min(SEGMENT, n - off);
        if (n - off - sn > 0 && n - off - sn < TAIL_FOLD) sn = n - off;
        const unsigned k = choose_k_hist(d + off, sn, s);
        seg.clear();
        if (!encode_segment_hist(seg, d + off, sn, k, k == 4 ? s.h4 : s.h2, s)) { out.clear(); return out; }
        m = put_varint(vb, sn); out.insert(out.end(), vb, vb + m);
        m = put_varint(vb, seg.size()); out.insert(out.end(), vb, vb + m);
        out.insert(out.end(), seg.begin(), seg.end());
        off += sn;
    }
    return out;
}

// Decodes a stream produced by encode() into dst[0..n). n must be the original length.
inline bool decode(uint8_t* dst, size_t n, const uint8_t* src, size_t sz) {
    const uint8_t* p = src; const uint8_t* e = src + sz;
    uint64_t nseg = 0;
    if (n < MIN_INPUT || !get_varint(p, e, nseg) || nseg == 0 || nseg > (n / MIN_INPUT)) return false;
    size_t done = 0;
    for (uint64_t g = 0; g < nseg; g++) {
        uint64_t sn = 0, cl = 0;
        if (!get_varint(p, e, sn) || !get_varint(p, e, cl)) return false;
        if (sn < MIN_INPUT || sn > n - done || cl > (uint64_t)(e - p)) return false;
        if (!decode_segment(dst + done, (size_t)sn, p, (size_t)cl)) return false;
        done += (size_t)sn; p += cl;
    }
    return done == n && p == e;
}
inline std::vector<uint8_t> decode(const uint8_t* src, size_t sz, size_t n) {
    std::vector<uint8_t> out(n);
    if (!decode(out.data(), n, src, sz)) out.clear();
    return out;
}

} // namespace pe
