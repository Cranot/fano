// test_pe.cpp - unit tests for plane-entropy. Exit status 0 = all passed.
// Copyright 2025 Cranot. Licensed under the Apache License, Version 2.0. See LICENSE.
//
//   ./test_pe                 run all tests; vectors.txt is REQUIRED and every entry is compared
//   ./test_pe --write-vectors regenerate vectors.txt from this build (review the diff, then commit)
//
// The stream is parsed and rebuilt here by test-local code (namespace tv) that shares nothing with
// the implementation, so the structural assertions and the malformed-stream cases below are not
// written in terms of the thing they are testing.
#include "plane_entropy.h"
#include "plane_entropy.hpp"
#include "reference/plane_entropy_ref.hpp"   // pe_ref:: - the plain scalar definition of the format
#include "test_stream.hpp"                    // tv:: - test-only parser, independent decoder, SHA-256
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) g_pass++; else { g_fail++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct Rng { uint64_t s; explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double unit() { return (next() >> 11) * (1.0 / 9007199254740992.0); } };

static std::vector<uint8_t> gen_zeros(size_t n, uint64_t) { return std::vector<uint8_t>(n, 0); }
static std::vector<uint8_t> gen_const(size_t n, uint64_t) { return std::vector<uint8_t>(n, 0x5A); }
static std::vector<uint8_t> gen_random(size_t n, uint64_t seed) { Rng r(seed); std::vector<uint8_t> v(n); for (auto& b : v) b = (uint8_t)r.next(); return v; }
// Gaussian weights as little-endian BF16 (2 bytes) or F32 (4 bytes): the shape of real model tensors.
static std::vector<uint8_t> gen_floats(size_t n, uint64_t seed, int width) {
    Rng r(seed); std::vector<uint8_t> v(n); size_t i = 0;
    while (i + width <= n) {
        double u1 = r.unit() + 1e-12, u2 = r.unit();
        float x = (float)(0.02 * std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
        uint32_t bits; std::memcpy(&bits, &x, 4);
        if (width == 4) { v[i] = bits & 0xff; v[i+1] = (bits >> 8) & 0xff; v[i+2] = (bits >> 16) & 0xff; v[i+3] = bits >> 24; }
        else { uint16_t b = (uint16_t)(bits >> 16); v[i] = b & 0xff; v[i+1] = b >> 8; }
        i += width;
    }
    for (; i < n; i++) v[i] = (uint8_t)r.next();   // tail bytes
    return v;
}
static std::vector<uint8_t> gen_bf16(size_t n, uint64_t s) { return gen_floats(n, s, 2); }
static std::vector<uint8_t> gen_f32(size_t n, uint64_t s) { return gen_floats(n, s, 4); }
// 4-byte records: constant plane, 3-symbol plane, random plane, 16-symbol plane.
static std::vector<uint8_t> gen_mixed(size_t n, uint64_t seed) { Rng r(seed); std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) { switch (i & 3) { case 0: v[i] = 7; break; case 1: v[i] = (uint8_t)(r.next() % 3); break; case 2: v[i] = (uint8_t)r.next(); break; default: v[i] = (uint8_t)(0x80 + r.next() % 16); } }
    return v; }

typedef std::vector<uint8_t> (*Gen)(size_t, uint64_t);
struct NamedGen { const char* name; Gen g; };
static const NamedGen GENS[] = { {"zeros", gen_zeros}, {"const", gen_const}, {"random", gen_random}, {"bf16", gen_bf16}, {"f32", gen_f32}, {"mixed", gen_mixed} };
static const size_t SIZES[] = { 0, 1, 63, 64, 65, 100, 255, 256, 257, 4095, 4096, 4097, 65535, 65536, 65537, 131071, 131072, 131073,
                                262079, 262080, 262081, 262143, 262144, 262154, 524160, 524161, 1048576 + 13 };
// streams for inputs at or below this size are frozen byte for byte in vectors.txt; larger ones by hash
static const size_t VECTOR_BYTES_MAX = 1024;



// Every decoder must refuse this stream, and the C API must return 0.
static void must_reject(const char* rule, const std::vector<uint8_t>& s, size_t n) {
    std::vector<uint8_t> o(n ? n : 1, 0xCD);
    bool a = pe::decode(o.data(), n, s.data(), s.size());
    size_t c = pe_decompress(o.data(), n, s.data(), s.size());
    bool r = pe_ref::decode(o.data(), n, s.data(), s.size());
    CHECK(!a, "rule [%s]: optimised decoder ACCEPTED a stream that breaks it", rule);
    CHECK(c == 0, "rule [%s]: C API returned %zu, expected 0", rule, c);
    CHECK(!r, "rule [%s]: reference decoder ACCEPTED a stream that breaks it", rule);
}

int main(int argc, char** argv) {
    const bool write_vectors = (argc > 1 && std::strcmp(argv[1], "--write-vectors") == 0);
    std::map<std::string, std::string> vectors;   // name -> "size hash [hex]"
    // ---- 1. round-trips over generators x sizes (C++ API and C API) ----
    printf("[1] round-trips\n");
    size_t rt_cases = 0, indep_cases = 0;
    for (const auto& ng : GENS) for (size_t n : SIZES) {
        std::vector<uint8_t> in = ng.g(n, 1234567 + n);
        std::vector<uint8_t> enc = pe::encode(in.data(), n);
        if (n < pe::MIN_INPUT) { CHECK(enc.empty(), "%s n=%zu: encode must reject n < 64", ng.name, n); continue; }
        CHECK(!enc.empty(), "%s n=%zu: encode failed", ng.name, n);
        if (enc.empty()) continue;
        CHECK(enc.size() <= pe::compress_bound(n), "%s n=%zu: stream %zu exceeds bound %zu", ng.name, n, enc.size(), pe::compress_bound(n));
        std::vector<uint8_t> dec = pe::decode(enc.data(), enc.size(), n);
        CHECK(dec.size() == n && std::memcmp(dec.data(), in.data(), n) == 0, "%s n=%zu: round-trip mismatch", ng.name, n);
        // C API
        std::vector<uint8_t> cbuf(pe_compress_bound(n)), cout(n);
        size_t cs = pe_compress(cbuf.data(), cbuf.size(), in.data(), n);
        CHECK(cs == enc.size() && std::memcmp(cbuf.data(), enc.data(), cs) == 0, "%s n=%zu: C API stream differs", ng.name, n);
        CHECK(pe_decompress(cout.data(), n, cbuf.data(), cs) == n && std::memcmp(cout.data(), in.data(), n) == 0, "%s n=%zu: C API round-trip", ng.name, n);
        CHECK(pe_compress(cbuf.data(), cs - 1, in.data(), n) == 0, "%s n=%zu: cap too small must return 0", ng.name, n);
        CHECK(pe_decompress(cout.data(), n - 1, cbuf.data(), cs) == 0, "%s n=%zu: wrong n must fail", ng.name, n);
        // The optimised coder must be byte-identical to the reference, and each must decode the other's
        // stream: this is what keeps the fused histogram / CTable path / SIMD from changing the format.
        std::vector<uint8_t> ref = pe_ref::encode(in.data(), n);
        CHECK(ref == enc, "%s n=%zu: optimised stream differs from reference (%zu vs %zu B)", ng.name, n, enc.size(), ref.size());
        std::vector<uint8_t> x1 = pe_ref::decode(enc.data(), enc.size(), n), x2 = pe::decode(ref.data(), ref.size(), n);
        CHECK(x1.size() == n && std::memcmp(x1.data(), in.data(), n) == 0, "%s n=%zu: reference cannot decode the optimised stream", ng.name, n);
        CHECK(x2.size() == n && std::memcmp(x2.data(), in.data(), n) == 0, "%s n=%zu: optimised cannot decode the reference stream", ng.name, n);
        rt_cases++;
        char key[64]; std::snprintf(key, sizeof key, "%s:%zu", ng.name, n);
        char val[64]; std::snprintf(val, sizeof val, "%zu ", enc.size());
        std::string rec = std::string(val) + tv::sha256_hex(enc);
        if (n <= VECTOR_BYTES_MAX) rec += " " + tv::to_hex(enc);   // frozen bytes as well as the digest
        // An independent decoder handles every stream whose planes are raw or single-symbol: for those
        // the round trip above is confirmed without huff0 and without any code from the coder.
        tv::IndepResult ir = tv::decode_independent(enc, n);
        if (ir.handled) { CHECK(ir.ok && ir.out.size() == n && std::memcmp(ir.out.data(), in.data(), n) == 0,
                                "%s n=%zu: the independent decoder disagrees with the stream", ng.name, n); indep_cases++; }
        vectors[key] = rec;
    }
    printf("    %zu cases, %zu of them also verified by the independent decoder (no huff0, no coder code)\n", rt_cases, indep_cases);
    CHECK(indep_cases >= 40, "only %zu streams were verified without huff0 (expected the raw and single-symbol ones, >= 40)", indep_cases);
    // ---- 2. structure of the produced streams (parsed by test-local code, not by pe::) ----
    printf("[2] stream structure\n");
    {
        auto structure = [](const std::vector<uint8_t>& in, const char* what, unsigned want_k, int want_coded /*-1 = any*/) {
            std::vector<uint8_t> e = pe::encode(in.data(), in.size());
            tv::Stream st = tv::parse(e);
            CHECK(st.ok && st.segs.size() == 1, "%s: expected one parsable segment", what);
            if (!st.ok || st.segs.empty()) return;
            CHECK(st.segs[0].seg_n == in.size(), "%s: seg_n %llu != input %zu", what, (unsigned long long)st.segs[0].seg_n, in.size());
            tv::Body y = tv::parse_body(st.segs[0].body, st.segs[0].seg_n);
            CHECK(y.ok, "%s: body does not parse (k=%u)", what, y.k);
            if (!y.ok) return;
            CHECK(y.k == want_k, "%s: k=%u, expected %u", what, y.k, want_k);
            CHECK(y.hflags == y.flags, "%s: hflags %u != flags %u", what, y.hflags, y.flags);
            CHECK((y.flags >> y.k) == 0, "%s: flag bits set above k", what);
            const size_t L = in.size() / y.k;
            int coded = 0;
            for (unsigned j = 0; j < y.k; j++) {
                bool is_coded = (y.flags >> j) & 1;
                if (is_coded) { coded++; CHECK(y.clen[j] >= 1 && y.clen[j] < L, "%s: coded plane %u clen %llu not in [1,%zu)", what, j, (unsigned long long)y.clen[j], L); }
                else CHECK(y.clen[j] == L, "%s: raw plane %u clen %llu != L %zu", what, j, (unsigned long long)y.clen[j], L);
            }
            if (want_coded >= 0) CHECK(coded == want_coded, "%s: %d coded planes, expected %d", what, coded, want_coded);
        };
        // all-zero input: every plane is one symbol, so every plane is huff0's RLE case (clen == 1)
        auto z = gen_zeros(65536, 0);
        structure(z, "zeros-64K", 2, 2);
        { auto e = pe::encode(z.data(), z.size()); tv::Stream st = tv::parse(e); tv::Body y = tv::parse_body(st.segs[0].body, st.segs[0].seg_n);
          CHECK(y.ok && y.clen.size() == 2 && y.clen[0] == 1 && y.clen[1] == 1, "zeros-64K: expected single-symbol planes (clen 1,1)");
          CHECK(y.ok && y.pay.size() == 2 && y.pay[0].size() == 1 && y.pay[0][0] == 0, "zeros-64K: RLE payload must be the symbol itself"); }
        // uniform random: no plane compresses, so every plane is stored raw
        structure(gen_random(65536, 9), "random-64K", 2, 0);
        // BF16 weights: the high (sign+exponent) plane compresses, the mantissa plane does not
        structure(gen_bf16(65536, 3), "bf16-64K", 2, 1);
        // F32 weights: 4 planes, and at least the exponent-carrying one compresses
        structure(gen_f32(65536, 4), "f32-64K", 4, -1);
        { auto f = gen_f32(65536, 4); auto e = pe::encode(f.data(), f.size()); tv::Stream st = tv::parse(e); tv::Body y = tv::parse_body(st.segs[0].body, st.segs[0].seg_n);
          int coded = 0; for (unsigned j = 0; j < 4; j++) coded += (y.flags >> j) & 1;
          CHECK(coded >= 1, "f32-64K: expected at least one coded plane, got %d", coded); }
        // segmentation: 262154 = 262080 + 74 (74 >= 64: NOT folded -> 2 segments); 262090 = 262080 + 10 (folded -> 1);
        // 524161 = 2 x 262080 + 1 (folded into the 2nd -> 2); 524167 = 2 x 262080 + 7 (folded -> 2); 524224 = 2 x 262080 + 64 (not folded -> 3)
        // 262144 = 262080 + 64: a remainder of exactly 64 is NOT folded (rule: fold only < 64) -> 2 segments
        struct S { size_t n; unsigned nseg; } segs[] = { {262154, 2}, {262090, 1}, {524161, 2}, {524167, 2}, {524224, 3}, {262080, 1}, {262081, 1}, {262143, 1}, {262144, 2} };
        for (auto& s : segs) {
            auto m = gen_mixed(s.n, 5); auto em = pe::encode(m.data(), m.size());
            tv::Stream st = tv::parse(em);
            CHECK(st.ok && st.nseg == s.nseg, "n=%zu: expected nseg %u, stream says %llu", s.n, s.nseg, (unsigned long long)(st.ok ? st.nseg : 0));
            if (st.ok) { uint64_t sum = 0; for (auto& g : st.segs) sum += g.seg_n; CHECK(sum == s.n, "n=%zu: segment sizes sum to %llu", s.n, (unsigned long long)sum);
                         CHECK(st.trailing.empty(), "n=%zu: %zu bytes after the last segment", s.n, st.trailing.size()); }
            std::vector<uint8_t> o(s.n);
            CHECK(pe::decode(o.data(), s.n, em.data(), em.size()) && std::memcmp(o.data(), m.data(), s.n) == 0, "n=%zu: round-trip", s.n);
        }
    }
    // ---- 3. every validation rule in SPEC.md section 5, one constructed stream each ----
    printf("[3] malformed streams, one per validation rule; rules that are logically coupled are\n        labelled [combined] rather than claiming isolation\n");
    {
        const size_t N = 65536;
        std::vector<uint8_t> rnd = gen_random(N, 21);          // k=2, both planes RAW  -> clen == L
        std::vector<uint8_t> bf  = gen_bf16(N, 22);            // k=2, high plane CODED -> 1 <= clen < L
        std::vector<uint8_t> enc_rnd = pe::encode(rnd.data(), N), enc_bf = pe::encode(bf.data(), N);
        tv::Stream s_rnd = tv::parse(enc_rnd), s_bf = tv::parse(enc_bf);
        CHECK(s_rnd.ok && s_bf.ok && s_rnd.segs.size() == 1 && s_bf.segs.size() == 1, "setup: base streams must parse as one segment");
        if (s_rnd.ok && s_bf.ok && s_rnd.segs.size() == 1 && s_bf.segs.size() == 1) {
            tv::Body b_rnd = tv::parse_body(s_rnd.segs[0].body, N), b_bf = tv::parse_body(s_bf.segs[0].body, N);
            CHECK(b_rnd.ok && b_rnd.flags == 0, "setup: random input should give raw planes (flags=%u)", b_rnd.flags);
            CHECK(b_bf.ok && b_bf.flags != 0, "setup: bf16 input should give at least one coded plane (flags=%u)", b_bf.flags);
            const size_t L = N / 2;

            // n >= 64
            must_reject("n >= 64 [combined: no set of >=64-byte segments can sum below 64]", enc_rnd, 63);
            must_reject("n >= 64, n = 0 [combined invariant]", enc_rnd, 0);
            // nseg >= 1
            { tv::Stream m = s_rnd; m.nseg = 0; m.segs.clear(); must_reject("nseg >= 1 [combined: zero segments cannot sum to n]", tv::build(m), N); }
            // nseg <= n / 64
            { tv::Stream m = s_rnd; m.nseg = N / 64 + 1; must_reject("nseg <= n/64 [combined with seg_n >= 64 and the exact sum]", tv::build(m), N); }
            // a varint must not carry payload bits above bit 63 (SPEC.md 2)
            { std::vector<uint8_t> m = { 0x81,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x02 };
              m.insert(m.end(), enc_rnd.begin() + 1, enc_rnd.end());
              must_reject("varint: no payload bit above 63", m, N); }
            // seg_n >= 64  (n = 128 split 63 + 65, so nseg <= n/64 and the sum are both satisfied)
            { tv::Stream m; m.nseg = 2; m.segs.resize(2);
              m.segs[0].seg_n = 63; m.segs[0].body = { 2, 0, 0 };
              m.segs[1].seg_n = 65; m.segs[1].body = { 2, 0, 0 };
              must_reject("seg_n >= 64", tv::build(m), 128); }
            // the segments' seg_n sum to exactly n
            must_reject("sum(seg_n) == n (too small)", enc_rnd, N - 1);
            must_reject("sum(seg_n) == n (too large)", enc_rnd, N + 1);
            // no bytes after the last segment
            { tv::Stream m = s_rnd; m.trailing.push_back(0x00); must_reject("no trailing bytes", tv::build(m), N); }
            // seg_len does not exceed the remaining input
            { std::vector<uint8_t> m; tv::wr_varint(m, 1); tv::wr_varint(m, N); tv::wr_varint(m, 1u << 20);
              m.insert(m.end(), 8, 0x00); must_reject("seg_len fits the input", m, N); }
            // seg_len >= 3
            { tv::Stream m = s_rnd; m.segs[0].body = { 2, 0 }; must_reject("seg_len >= 3", tv::build(m), N); }
            // k in {2,4}
            for (uint8_t bad : { 0, 1, 3, 5, 8, 255 }) { tv::Stream m = s_rnd; m.segs[0].body[0] = bad;
              char nm[32]; std::snprintf(nm, sizeof nm, "k in {2,4} (k=%u)", bad); must_reject(nm, tv::build(m), N); }
            // hflags == flags
            { tv::Stream m = s_rnd; m.segs[0].body[2] ^= 1; must_reject("hflags == flags", tv::build(m), N); }
            // flags >> k == 0
            { tv::Stream m = s_rnd; m.segs[0].body[1] = 4; m.segs[0].body[2] = 4; must_reject("flags >> k == 0", tv::build(m), N); }
            // L <= 131072  (seg_n = 262146 with k = 2 gives L = 131073; the check precedes the plane fields)
            { tv::Stream m; m.nseg = 1; m.segs.resize(1); m.segs[0].seg_n = 262146; m.segs[0].body = { 2, 0, 0 };
              must_reject("L <= 131072", tv::build(m), 262146); }
            // each clen fits in the remaining body
            { tv::Body y = b_bf; y.clen[0] = L - 1; y.pay[0].assign(10, 0x00);   // declared length not present
              tv::Stream m = s_bf; m.segs[0].body = tv::build_body(y); must_reject("clen fits the body", tv::build(m), N); }
            // a coded plane has 1 <= clen < L
            { tv::Body y = b_rnd; y.flags = y.hflags = 1; y.clen[0] = L; /* pay[0] already L bytes */
              tv::Stream m = s_rnd; m.segs[0].body = tv::build_body(y); must_reject("coded clen < L", tv::build(m), N); }
            { tv::Body y = b_rnd; y.flags = y.hflags = 1; y.clen[0] = 0; y.pay[0].clear();
              tv::Stream m = s_rnd; m.segs[0].body = tv::build_body(y); must_reject("coded clen >= 1", tv::build(m), N); }
            // a raw plane has clen == L
            { tv::Body y = b_rnd; y.clen[0] = L - 1; y.pay[0].resize(L - 1);
              tv::Stream m = s_rnd; m.segs[0].body = tv::build_body(y); must_reject("raw clen == L", tv::build(m), N); }
            // huff0 must return exactly L: truncate a coded plane by one byte and declare the shorter length
            { tv::Body y = b_bf; unsigned j = 0; while (j < y.k && !((y.flags >> j) & 1)) j++;
              if (j < y.k && y.clen[j] > 2) { y.pay[j].pop_back(); y.clen[j]--; tv::Stream m = s_bf; m.segs[0].body = tv::build_body(y);
                  must_reject("huff0 returns exactly L", tv::build(m), N); } }
            // the tail is exactly seg_n - L*k bytes
            { tv::Body y = b_rnd; y.tail.push_back(0x00); tv::Stream m = s_rnd; m.segs[0].body = tv::build_body(y);
              must_reject("tail length exact", tv::build(m), N); }
        }
        // Cases built from nothing but the spec, each with a COMPLETE otherwise-valid body, so that
        // removing the guard under test would let the stream through rather than failing later.
        // (Contributed by the round-4b review, which verified each one against the decoder.)
        {
            auto raw_body = [](size_t sn, unsigned k) {
                const size_t L = sn / k; std::vector<uint8_t> b = { (uint8_t)k, 0, 0 };
                for (unsigned j = 0; j < k; j++) tv::wr_varint(b, L);
                b.insert(b.end(), L * k, 0x5a); b.insert(b.end(), sn - L * k, 0xa5); return b;
            };
            auto seg = [](std::vector<uint8_t>& st, size_t sn, const std::vector<uint8_t>& body) {
                tv::wr_varint(st, sn); tv::wr_varint(st, body.size()); st.insert(st.end(), body.begin(), body.end());
            };
            { std::vector<uint8_t> m(10, 0x80); m.push_back(0x00); must_reject("varint at most 10 bytes", m, 64); }
            { std::vector<uint8_t> m = { 0x80 }; must_reject("varint must terminate inside the input", m, 64); }
            { std::vector<uint8_t> m; tv::wr_varint(m, 2); seg(m, 63, raw_body(63, 2)); seg(m, 65, raw_body(65, 2));
              must_reject("seg_n >= 64 (complete bodies)", m, 128); }
            { const size_t big = 262146; std::vector<uint8_t> m; tv::wr_varint(m, 1); seg(m, big, raw_body(big, 2));
              must_reject("L <= 131072 (complete body)", m, big); }
            { std::vector<uint8_t> body = raw_body(65, 2), m; tv::wr_varint(m, 1); tv::wr_varint(m, 65);
              tv::wr_varint(m, body.size() + 1); m.insert(m.end(), body.begin(), body.end());
              must_reject("seg_len exceeds the input by one", m, 65); }
            { std::vector<uint8_t> body = { 2, 1, 1 }; tv::wr_varint(body, 31); tv::wr_varint(body, 32);
              body.insert(body.end(), 30, 0x00); std::vector<uint8_t> m; tv::wr_varint(m, 1); seg(m, 64, body);
              must_reject("clen exceeds the remaining body", m, 64); }
            { std::vector<uint8_t> body = { 2, 1, 1 }; tv::wr_varint(body, 2); tv::wr_varint(body, 32);
              body.push_back(0x00); body.push_back(0x00); body.insert(body.end(), 32, 0x5a);
              std::vector<uint8_t> m; tv::wr_varint(m, 1); seg(m, 64, body);
              must_reject("huff0 must return exactly L (malformed block)", m, 64); }
        }
    }
    // ---- 4. corruption: truncation always fails; a flip in the structure always fails ----
    printf("[4] corruption\n");
    {
        struct C { const char* name; std::vector<uint8_t> in; } cases[] = { {"bf16-64K", gen_bf16(65536, 11)}, {"random-4K", gen_random(4096, 12)}, {"zeros-4K", gen_zeros(4096, 0)}, {"mixed-262154", gen_mixed(262154, 13)}, {"f32-100", gen_f32(100, 14)} };
        size_t trunc = 0, struct_flips = 0, payload_flips = 0, payload_rejected = 0;
        for (auto& c : cases) {
            auto enc = pe::encode(c.in.data(), c.in.size()); const size_t n = c.in.size(); std::vector<uint8_t> out(n);
            for (size_t len = 0; len < enc.size(); len++) { bool ok = pe::decode(out.data(), n, enc.data(), len); CHECK(!ok, "%s: truncated to %zu must fail", c.name, len); trunc++; }
            // Bit flips, classified by where they land. Everything before the first payload byte is
            // structure (lengths, k, flags, clen); a flip there changes a declared length or a
            // reserved field and MUST be rejected. A flip inside a raw plane or the tail is
            // undetectable by design (no checksum), so those are only required not to crash.
            const size_t send = tv::structure_end(enc);
            for (size_t pos = 0; pos < enc.size() && pos < 4096; pos++) for (int bit = 0; bit < 8; bit++) {
                auto m = enc; m[pos] ^= (uint8_t)(1u << bit);
                bool ok = pe::decode(out.data(), n, m.data(), m.size());
                if (send && pos < send) { CHECK(!ok, "%s: flip at structure byte %zu bit %d was accepted", c.name, pos, bit); struct_flips++; }
                else { payload_flips++; if (!ok) payload_rejected++; }
            }
            Rng r(77);
            for (int t = 0; t < 2000; t++) { auto m = enc; size_t pos = r.next() % m.size(); m[pos] = (uint8_t)r.next(); if (t & 1) m.resize(1 + r.next() % m.size());
                bool ok = pe::decode(out.data(), n, m.data(), m.size()); (void)ok; }
            for (size_t dn : { n - 1, n + 1, n / 2, 2 * n, (size_t)64, (size_t)1 << 20 }) { if (dn == n) continue; std::vector<uint8_t> o2(dn); bool ok = pe::decode(o2.data(), dn, enc.data(), enc.size()); CHECK(!ok, "%s: wrong n=%zu accepted", c.name, dn); }
        }
        printf("    %zu truncations rejected, %zu structure flips all rejected, %zu payload/tail flips (%zu rejected, the rest are undetectable by design), 10000 random mutations survived\n",
               trunc, struct_flips, payload_flips, payload_rejected);
        Rng r(99); std::vector<uint8_t> o(65536);
        for (int t = 0; t < 5000; t++) { std::vector<uint8_t> g(1 + r.next() % 300); for (auto& b : g) b = (uint8_t)r.next(); (void)pe::decode(o.data(), 64 + r.next() % 65472, g.data(), g.size()); }
    }
    // ---- 5. optimised == reference on edge shapes ----
    printf("[5] optimised vs reference, edge shapes\n");
    { Rng r(4242); size_t cases = 0;
      for (size_t n : { (size_t)64, (size_t)127, (size_t)4099, (size_t)65536, (size_t)262079, (size_t)262081, (size_t)524200 }) {
        std::vector<std::vector<uint8_t>> ins;
        { std::vector<uint8_t> v(n); for (size_t i = 0; i < n; i++) v[i] = (i & 3) == 0 ? 0 : (uint8_t)r.next(); ins.push_back(v); }          // one constant plane of 4
        { std::vector<uint8_t> v(n); for (size_t i = 0; i < n; i++) v[i] = (i & 1) ? 0xFF : (uint8_t)(r.next() & 0x0F); ins.push_back(v); }   // constant + low-entropy of 2
        { std::vector<uint8_t> v(n); for (size_t i = 0; i < n; i++) v[i] = (uint8_t)(r.next() % 250 == 0 ? r.next() : 42); ins.push_back(v); } // just past huff0's compressibility heuristic
        { std::vector<uint8_t> v(n); for (size_t i = 0; i < n; i++) v[i] = (uint8_t)(i % 251); ins.push_back(v); }                            // flat-ish histogram
        for (auto& in : ins) { auto a = pe::encode(in.data(), n), b = pe_ref::encode(in.data(), n);
            CHECK(a == b, "edge n=%zu: optimised %zu B vs reference %zu B", n, a.size(), b.size());
            std::vector<uint8_t> o(n); CHECK(!a.empty() && pe::decode(o.data(), n, a.data(), a.size()) && std::memcmp(o.data(), in.data(), n) == 0, "edge n=%zu: round-trip", n); cases++; }
      }
      printf("    %zu edge inputs\n", cases); }
    // ---- 6. spec constants, the k rule's determinism, and the C API's edges ----
    printf("[6] spec constants and API edges\n");
    CHECK(pe::SEGMENT == 262080 && pe::MIN_INPUT == 64 && pe::TAIL_FOLD == 64, "constants changed: update SPEC.md and vectors");
    CHECK(HUF_BLOCKSIZE_MAX == 131072 && (pe::SEGMENT + pe::TAIL_FOLD - 1) / 2 <= HUF_BLOCKSIZE_MAX, "a folded k=2 plane must fit huff0's block limit");
    // the fixed-point log used by the k rule must agree with the real logarithm to within one Q16 step
    { double worst = 0; for (uint32_t x : { 1u, 2u, 3u, 7u, 255u, 256u, 1000u, 65535u, 65536u, 131071u, 262143u, 1000000u }) {
          double got = pe::log2_q16(x) / 65536.0, want = std::log2((double)x); worst = std::max(worst, std::fabs(got - want)); }
      CHECK(worst < 1.0 / 4096.0, "log2_q16 is off by %g bits at worst", worst); }
    CHECK(pe::log2_q16(1) == 0 && pe::log2_q16(2) == (1u << 16) && pe::log2_q16(256) == (8u << 16), "log2_q16 must be exact on powers of two");
    // the digest that freezes the vectors must itself be right, or the vector check proves nothing
    { std::vector<uint8_t> empty, abc = { 'a', 'b', 'c' }; std::vector<uint8_t> mil(1000000, 'a');
      CHECK(tv::sha256_hex(empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256 of the empty string is wrong");
      CHECK(tv::sha256_hex(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 of \"abc\" is wrong");
      CHECK(tv::sha256_hex(mil) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "SHA-256 of a million 'a' is wrong"); }
    // compress_bound must stay an upper bound, and must not wrap
    CHECK(pe::compress_bound(SIZE_MAX) == SIZE_MAX, "compress_bound must saturate instead of wrapping");
    { std::vector<uint8_t> in = gen_bf16(4096, 31), o(4096);
      CHECK(pe_compress(nullptr, 100, in.data(), in.size()) == 0, "NULL dst must return 0");
      CHECK(pe_compress(o.data(), o.size(), nullptr, 64) == 0, "NULL src must return 0");
      CHECK(pe_decompress(nullptr, 64, in.data(), in.size()) == 0, "NULL dst must return 0 on decompress");
      CHECK(pe_decompress(o.data(), 64, nullptr, 10) == 0, "NULL src must return 0 on decompress");
      CHECK(pe_compress(o.data(), 0, in.data(), in.size()) == 0, "dst_cap 0 must return 0");
      CHECK(pe_compress(o.data(), o.size(), in.data(), 0) == 0, "n = 0 must return 0"); }
    // ---- 7. frozen stream vectors (required) ----
    printf("[7] stream vectors\n");
    if (write_vectors) {
        std::ofstream f("vectors.txt");
        f << "# plane-entropy frozen streams: <generator>:<n> <stream bytes> <sha256 of the stream> [full stream in hex, n <= " << VECTOR_BYTES_MAX << "]\n";
        for (auto& kv : vectors) f << kv.first << " " << kv.second << "\n";
        printf("    wrote %zu vectors to vectors.txt\n", vectors.size());
    } else {
        std::ifstream f("vectors.txt");
        CHECK((bool)f, "vectors.txt is missing: the frozen streams are part of the test, not optional");
        if (f) {
            std::set<std::string> seen; size_t cmp = 0, with_bytes = 0; std::string line;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::string k, sz, h, hex; size_t a = line.find(' ');
                if (a == std::string::npos) { CHECK(false, "vectors.txt: malformed line [%s]", line.c_str()); continue; }
                k = line.substr(0, a);
                std::string rest = line.substr(a + 1);
                size_t b = rest.find(' '); sz = rest.substr(0, b); rest = rest.substr(b + 1);
                size_t c = rest.find(' ');
                if (c == std::string::npos) h = rest; else { h = rest.substr(0, c); hex = rest.substr(c + 1); }
                auto it = vectors.find(k);
                CHECK(it != vectors.end(), "vectors.txt names %s, which this build does not produce", k.c_str());
                if (it == vectors.end()) continue;
                seen.insert(k);
                std::string want = sz + " " + h + (hex.empty() ? "" : " " + hex);
                CHECK(it->second == want, "vector %s differs from the frozen stream", k.c_str());
                cmp++; if (!hex.empty()) with_bytes++;
            }
            for (auto& kv : vectors) CHECK(seen.count(kv.first) == 1, "vector %s is produced by this build but absent from vectors.txt", kv.first.c_str());
            CHECK(cmp == vectors.size(), "vectors.txt has %zu entries, this build produces %zu", cmp, vectors.size());
            CHECK(with_bytes >= 30, "only %zu vectors carry full stream bytes (expected the small ones, >= 30)", with_bytes);
            printf("    %zu vectors compared (%zu byte-for-byte, the rest by size and hash)\n", cmp, with_bytes);
        }
    }
    printf("=== %d checks passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
