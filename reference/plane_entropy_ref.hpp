// plane_entropy_ref.hpp - REFERENCE implementation of the plane-entropy coder (v1: plain scalar code,
// one huff0 call per plane). Kept verbatim as the executable definition of the format: the optimised
// coder in ../plane_entropy.hpp must produce byte-identical streams (test_pe asserts this on every
// generator and size) and must decode everything this decoder decodes.
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
// Dependency: huf.h / huff0 (third_party/fse). No other library.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "huf.h"

namespace pe_ref {

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
        if (sh == 63 && (b & 0x7e) != 0) return false;   // no payload bit above 63 (SPEC.md 2)
        v |= (uint64_t)(b & 0x7f) << sh;
        if (!(b & 0x80)) return true;
        sh += 7;
    }
    return false;
}

// ---- k selection: order-0 estimate of the planes (encoder-side heuristic only; k is stored) ----
// Integer arithmetic in Q16 (bits x 65536): floating point would make the chosen k, and therefore
// the stream bytes, depend on the compiler's evaluation model. See SPEC.md section 4.
static const int64_t MM_Q16 = 47274;      // round(2^16 / (2 ln 2)): Miller-Madow, per extra symbol
static const int64_t TABLE_Q16 = 1311;    // round(0.02 * 2^16): per-symbol table-cost allowance
inline uint32_t log2_q16(uint32_t x) {    // round-toward-zero log2(x) in Q16, x >= 1
    if (x <= 1) return 0;
    int e = 0; { uint32_t t = x; while (t >>= 1) e++; }
    uint64_t y = (uint64_t)x << (31 - e);                 // mantissa in Q31, [2^31, 2^32)
    uint32_t r = (uint32_t)e << 16;
    for (int i = 0; i < 16; i++) {
        y = (y * y) >> 31;
        if (y >= (1ull << 32)) { y >>= 1; r |= 1u << (15 - i); }
    }
    return r;
}
inline int64_t plane_bits_q16(const uint8_t* d, size_t n, unsigned k) {
    size_t L = n / k; int64_t bits = 0;
    uint32_t h[256];
    for (unsigned j = 0; j < k; j++) {
        std::memset(h, 0, sizeof(h));
        for (size_t i = 0; i < L; i++) h[d[i * k + j]]++;
        if (!L) continue;
        int64_t sum = 0; unsigned K = 0;
        for (unsigned c = 0; c < 256; c++) if (h[c]) { K++; sum += (int64_t)h[c] * (int64_t)log2_q16(h[c]); }
        int64_t hb = (int64_t)L * (int64_t)log2_q16((uint32_t)L) - sum;
        // Miller-Madow bias correction: the empirical entropy of a short plane underestimates the true
        // entropy by about (K-1)/(2 L ln 2) bits/symbol, which favoured k = 4 (shorter planes) on small
        // inputs. Measured on 8 real weight slices at 4 KiB chunks: 1,897/16,384 wrong picks (0.11% lost)
        // without the correction, 3 with it; no difference at 16 KiB and above.
        if (K > 1) hb += (int64_t)(K - 1) * MM_Q16;
        hb += (int64_t)L * TABLE_Q16;                     // +0.02 bits/symbol approximates table cost
        const int64_t cap = (int64_t)L * (8 << 16);       // a raw plane costs 8 bits/byte
        bits += hb < cap ? hb : cap;
    }
    return bits;
}
inline double plane_bits(const uint8_t* d, size_t n, unsigned k) { return (double)plane_bits_q16(d, n, k) / 65536.0; }
inline unsigned choose_k(const uint8_t* d, size_t n) {
    const int64_t b2 = plane_bits_q16(d, n, 2), b4 = plane_bits_q16(d, n, 4);
    return (b4 * 100 < b2 * 99) ? 4u : 2u;   // k=4 iff at least 1% smaller; ties go to 2 planes
}

// ---- one segment ---------------------------------------------------------------------------
// Layout: [k:1][flags:1][hflags:1][clen varint x k][plane payloads][tail bytes]
//   flags  bit j = plane j is huff0-coded (else raw, clen == L)
//   hflags must equal flags in this version (reserved for other plane codecs)
inline bool encode_segment(std::vector<uint8_t>& out, const uint8_t* d, size_t n, unsigned k) {
    if (n < MIN_INPUT || (k != 2 && k != 4)) return false;
    const size_t L = n / k;   // the n - L*k tail bytes are appended verbatim below
    if (L > HUF_BLOCKSIZE_MAX) return false;
    std::vector<uint8_t> plane(L);
    std::vector<std::vector<uint8_t>> pay(k);
    std::vector<uint8_t> cb(HUF_compressBound(L) + 16);
    uint8_t flags = 0;
    for (unsigned j = 0; j < k; j++) {
        for (size_t i = 0; i < L; i++) plane[i] = d[i * k + j];
        size_t h = HUF_compress(cb.data(), cb.size(), plane.data(), L);
        // h == 0: not compressible -> raw. h == 1: single-symbol plane (RLE; the byte is in cb[0]).
        if (!HUF_isError(h) && h >= 1 && h < L) { flags |= (uint8_t)(1u << j); pay[j].assign(cb.begin(), cb.begin() + h); }
        else pay[j] = plane;
    }
    out.push_back((uint8_t)k); out.push_back(flags); out.push_back(flags);
    uint8_t vb[16];
    for (unsigned j = 0; j < k; j++) { size_t m = put_varint(vb, pay[j].size()); out.insert(out.end(), vb, vb + m); }
    for (unsigned j = 0; j < k; j++) out.insert(out.end(), pay[j].begin(), pay[j].end());
    out.insert(out.end(), d + L * k, d + n);
    return true;
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
    std::vector<uint8_t> plane(L);
    for (unsigned j = 0; j < k; j++) {
        if (cl[j] > (uint64_t)(e - p)) return false;
        if (flags & (1u << j)) {
            if (cl[j] == 0 || cl[j] >= L) return false;           // a coded plane is strictly shorter than the plane
            size_t r = HUF_decompress(plane.data(), L, p, (size_t)cl[j]);
            if (HUF_isError(r) || r != L) return false;
        } else {
            if (cl[j] != L) return false;
            std::memcpy(plane.data(), p, L);
        }
        p += cl[j];
        for (size_t i = 0; i < L; i++) dst[i * k + j] = plane[i];
    }
    if ((size_t)(e - p) != tail) return false;
    if (tail) std::memcpy(dst + L * k, p, tail);
    return true;
}

// ---- whole input: segments ------------------------------------------------------------------
// Layout: [nseg varint] { [seg_n varint][seg_len varint][segment] } x nseg
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
    std::vector<uint8_t> seg;
    for (size_t off = 0; off < n; ) {
        size_t sn = std::min(SEGMENT, n - off);
        if (n - off - sn > 0 && n - off - sn < TAIL_FOLD) sn = n - off;
        seg.clear();
        if (!encode_segment(seg, d + off, sn, choose_k(d + off, sn))) { out.clear(); return out; }
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

} // namespace pe_ref
