// test_stream.hpp - TEST-ONLY stream tools: a parser, a builder, an independent decoder and a
// SHA-256. Written from SPEC.md alone; it shares no code with plane_entropy.hpp, its reference, or
// huff0, so the assertions written in terms of it are not written in terms of the thing under test.
// Used by test_pe.cpp and fuzz_pe.cpp.
// Copyright 2025 Cranot. Licensed under the Apache License, Version 2.0. See LICENSE.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace tv {

// ---- varints (SPEC.md 2) --------------------------------------------------------------------
inline bool rd_varint(const uint8_t*& p, const uint8_t* e, uint64_t& v) {
    v = 0; int sh = 0;
    while (p < e && sh < 64) { uint8_t b = *p++; if (sh == 63 && (b & 0x7e) != 0) return false; v |= (uint64_t)(b & 0x7f) << sh; if (!(b & 0x80)) return true; sh += 7; }
    return false;
}
inline void wr_varint(std::vector<uint8_t>& o, uint64_t v) { while (v >= 0x80) { o.push_back((uint8_t)(v | 0x80)); v >>= 7; } o.push_back((uint8_t)v); }
inline size_t varint_len(uint64_t v) { size_t n = 1; while (v >= 0x80) { v >>= 7; n++; } return n; }

// ---- stream and segment structure (SPEC.md 3) -----------------------------------------------
struct Seg { uint64_t seg_n = 0; std::vector<uint8_t> body; };
struct Stream { uint64_t nseg = 0; std::vector<Seg> segs; std::vector<uint8_t> trailing; bool ok = false; };

inline Stream parse(const std::vector<uint8_t>& s) {
    Stream st; const uint8_t* p = s.data(); const uint8_t* e = p + s.size();
    if (!rd_varint(p, e, st.nseg)) return st;
    if (st.nseg == 0 || st.nseg > 1000000) return st;
    for (uint64_t i = 0; i < st.nseg; i++) {
        Seg g; uint64_t len = 0;
        if (!rd_varint(p, e, g.seg_n) || !rd_varint(p, e, len)) return st;
        if ((uint64_t)(e - p) < len) return st;
        g.body.assign(p, p + (size_t)len); p += len; st.segs.push_back(std::move(g));
    }
    st.trailing.assign(p, e); st.ok = true; return st;
}
inline std::vector<uint8_t> build(const Stream& st) {
    std::vector<uint8_t> o; wr_varint(o, st.nseg);
    for (const auto& g : st.segs) { wr_varint(o, g.seg_n); wr_varint(o, g.body.size()); o.insert(o.end(), g.body.begin(), g.body.end()); }
    o.insert(o.end(), st.trailing.begin(), st.trailing.end()); return o;
}
// body = k:u8 | flags:u8 | hflags:u8 | clen varint x k | payload x k | tail
struct Body { unsigned k = 0; uint8_t flags = 0, hflags = 0; std::vector<uint64_t> clen; std::vector<std::vector<uint8_t>> pay; std::vector<uint8_t> tail; bool ok = false; size_t header_len = 0; };
inline Body parse_body(const std::vector<uint8_t>& b, uint64_t seg_n) {
    Body y; if (b.size() < 3) return y;
    y.k = b[0]; y.flags = b[1]; y.hflags = b[2];
    if (y.k != 2 && y.k != 4) return y;
    const uint8_t* p = b.data() + 3; const uint8_t* e = b.data() + b.size();
    for (unsigned j = 0; j < y.k; j++) { uint64_t c; if (!rd_varint(p, e, c)) return y; y.clen.push_back(c); }
    y.header_len = (size_t)(p - b.data());
    for (unsigned j = 0; j < y.k; j++) { if ((uint64_t)(e - p) < y.clen[j]) return y; y.pay.push_back(std::vector<uint8_t>(p, p + (size_t)y.clen[j])); p += y.clen[j]; }
    y.tail.assign(p, e);
    y.ok = (y.tail.size() == seg_n - (seg_n / y.k) * y.k);
    return y;
}
inline std::vector<uint8_t> build_body(const Body& y) {
    std::vector<uint8_t> b; b.push_back((uint8_t)y.k); b.push_back(y.flags); b.push_back(y.hflags);
    for (uint64_t c : y.clen) wr_varint(b, c);
    for (const auto& p : y.pay) b.insert(b.end(), p.begin(), p.end());
    b.insert(b.end(), y.tail.begin(), y.tail.end()); return b;
}
// byte offset just past the last structural byte of a single-segment stream (0 if not applicable)
inline size_t structure_end(const std::vector<uint8_t>& s) {
    Stream st = parse(s); if (!st.ok || st.segs.size() != 1) return 0;
    Body y = parse_body(st.segs[0].body, st.segs[0].seg_n); if (!y.ok) return 0;
    return varint_len(st.nseg) + varint_len(st.segs[0].seg_n) + varint_len(st.segs[0].body.size()) + y.header_len;
}

// ---- an INDEPENDENT decoder for the streams that need no Huffman block ----------------------
// A plane is stored raw (clen == L) or, when huff0 found a single symbol, as its RLE form (clen == 1,
// the payload byte repeated L times; SPEC.md 2). Everything else needs huff0 and is not attempted.
// Returns {ok=false} for a stream this decoder does not handle, which is not a failure.
struct IndepResult { bool handled = false; bool ok = false; std::vector<uint8_t> out; };
inline IndepResult decode_independent(const std::vector<uint8_t>& s, size_t n) {
    IndepResult r;
    Stream st = parse(s);
    if (!st.ok || !st.trailing.empty()) return r;
    std::vector<uint8_t> out; out.reserve(n);
    uint64_t sum = 0;
    for (const auto& g : st.segs) {
        Body y = parse_body(g.body, g.seg_n);
        if (!y.ok || y.hflags != y.flags || (y.flags >> y.k) != 0) return r;
        const size_t L = (size_t)(g.seg_n / y.k);
        std::vector<std::vector<uint8_t>> plane(y.k);
        for (unsigned j = 0; j < y.k; j++) {
            const bool coded = (y.flags >> j) & 1;
            if (!coded) { if (y.clen[j] != L) return r; plane[j] = y.pay[j]; }
            else if (y.clen[j] == 1) plane[j].assign(L, y.pay[j][0]);   // huff0's single-symbol form
            else return r;                                             // a real Huffman block: not ours
        }
        size_t base = out.size(); out.resize(base + (size_t)g.seg_n);
        for (size_t i = 0; i < L; i++) for (unsigned j = 0; j < y.k; j++) out[base + i * y.k + j] = plane[j][i];
        if (!y.tail.empty()) std::memcpy(out.data() + base + L * y.k, y.tail.data(), y.tail.size());
        sum += g.seg_n;
    }
    r.handled = true;
    r.ok = (sum == n && out.size() == n);
    r.out = std::move(out);
    return r;
}

// ---- SHA-256 (FIPS 180-4), test-only, so the frozen vectors do not rest on a 64-bit hash -----
struct Sha256 {
    uint32_t h[8]; uint64_t len = 0; uint8_t buf[64]; size_t buflen = 0;
    Sha256() { static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}; std::memcpy(h, iv, sizeof h); }
    static uint32_t ror(uint32_t x, int k) { return (x >> k) | (x << (32 - k)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) | ((uint32_t)p[4*i+2] << 8) | p[4*i+3];
        for (int i = 16; i < 64; i++) { uint32_t s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3); uint32_t s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10); w[i] = w[i-16] + s0 + w[i-7] + s1; }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25), ch = (e & f) ^ (~e & g), t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22), mj = (a & b) ^ (a & c) ^ (b & c), t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const uint8_t* p, size_t n) {
        len += n;
        while (n) { size_t take = 64 - buflen; if (take > n) take = n; std::memcpy(buf + buflen, p, take); buflen += take; p += take; n -= take; if (buflen == 64) { block(buf); buflen = 0; } }
    }
    std::string hex() {
        uint64_t bits = len * 8; uint8_t pad = 0x80; update(&pad, 1);
        uint8_t z = 0; while (buflen != 56) update(&z, 1);
        uint8_t lb[8]; for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
        update(lb, 8);   // bits was captured before padding, so len is no longer read
        static const char* D = "0123456789abcdef"; std::string s;
        for (int i = 0; i < 8; i++) for (int b = 3; b >= 0; b--) { uint8_t v = (uint8_t)(h[i] >> (8 * b)); s.push_back(D[v >> 4]); s.push_back(D[v & 15]); }
        return s;
    }
};
inline std::string sha256_hex(const std::vector<uint8_t>& v) { Sha256 s; s.update(v.data(), v.size()); return s.hex(); }
inline std::string to_hex(const std::vector<uint8_t>& v) { static const char* D = "0123456789abcdef"; std::string s; s.reserve(v.size() * 2); for (uint8_t b : v) { s.push_back(D[b >> 4]); s.push_back(D[b & 15]); } return s; }

} // namespace tv
