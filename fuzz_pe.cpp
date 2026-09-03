// fuzz_pe.cpp - mutation fuzzer for the decoder. Build with sanitizers (make fuzz_pe), run:
//   ./fuzz_pe [seconds=60] [seed=1]
// Exit status is non-zero if any oracle below is violated; a crash or sanitizer report is a failure
// in the same way. Copyright 2025 Cranot. Licensed under the Apache License, Version 2.0.
//
// Oracles, checked on every mutated stream:
//   1. the optimised and the reference decoder must AGREE on accept/reject;
//   2. when both accept, they must produce the SAME n bytes (they are separate implementations of
//      SPEC.md, so a disagreement is a defect in one of them);
//   3. the C API must return n exactly when the C++ API accepts, and 0 exactly when it rejects;
//   4. a decode must never write outside dst[0..n) — the guard bytes around the buffer are checked
//      after every call (ASAN catches this too; the guard also catches it in a plain build).
// Mutations are of two kinds: unstructured (bit flips, replacement, truncation, insertion, deletion,
// swap, anywhere) and FIELD-AWARE — the stream is parsed by the test-only reader and one named field
// (nseg, seg_n, seg_len, k, flags, hflags, a clen, the tail length) is set to a boundary value, so the
// edges of every length rule are reached deliberately rather than by chance.
// Accepting a mutated stream is not itself a fault: a flip inside a raw plane or the tail is
// undetectable by design (the format carries no checksum; the container hashes the chunk).
#include "plane_entropy.h"
#include "plane_entropy.hpp"
#include "reference/plane_entropy_ref.hpp"
#include "test_stream.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Rng { uint64_t s; explicit Rng(uint64_t seed) : s(seed ? seed : 88172645463325252ULL) {} uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; } };

static const uint8_t GUARD = 0xA5;
static const size_t GUARD_N = 64;

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 60.0;
    Rng r(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1);
    struct Item { std::vector<uint8_t> in, enc; };
    std::vector<Item> corpus;
    auto add = [&](std::vector<uint8_t> in) { Item it; it.enc = pe::encode(in.data(), in.size()); if (!it.enc.empty()) { it.in = std::move(in); corpus.push_back(std::move(it)); } };
    for (size_t n : { 64, 65, 100, 4096, 4097, 65536, 65537, 262080, 262154, 524161 }) {
        std::vector<uint8_t> z(n, 0), c(n, 0x33), rnd(n), bf(n), f4(n), mx(n);
        for (size_t i = 0; i < n; i++) { rnd[i] = (uint8_t)r.next(); mx[i] = (i & 3) == 0 ? 7 : (i & 3) == 1 ? (uint8_t)(r.next() % 3) : (i & 3) == 2 ? (uint8_t)r.next() : (uint8_t)(0x80 + r.next() % 16); }
        for (size_t i = 0; i + 2 <= n; i += 2) { uint16_t e = (uint16_t)(0x3C00 + (r.next() % 0x300)); if (r.next() & 1) e |= 0x8000; bf[i] = e & 0xff; bf[i+1] = e >> 8; }
        for (size_t i = 0; i + 4 <= n; i += 4) { uint32_t x = (uint32_t)(r.next()) & 0x007fffff; x |= (uint32_t)(0x3c + r.next() % 6) << 23; if (r.next() & 1) x |= 0x80000000u; f4[i] = x & 0xff; f4[i+1] = (x >> 8) & 0xff; f4[i+2] = (x >> 16) & 0xff; f4[i+3] = x >> 24; }
        add(z); add(c); add(rnd); add(bf); add(f4); add(mx);
    }
    printf("corpus: %zu streams\n", corpus.size());
    size_t cap = (1u << 21);
    std::vector<uint8_t> buf(cap + 2 * GUARD_N, GUARD), buf2(cap + 2 * GUARD_N, GUARD);
    auto guards_intact = [&](const std::vector<uint8_t>& b, size_t n) {
        for (size_t i = 0; i < GUARD_N; i++) if (b[i] != GUARD) return false;
        for (size_t i = GUARD_N + n; i < b.size(); i++) if (b[i] != GUARD) return false;
        return true;
    };
    auto t0 = std::chrono::steady_clock::now();
    unsigned long iters = 0, ok = 0, rejected = 0, ok_changed = 0, bad = 0, structured_iters = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        const Item& it = corpus[r.next() % corpus.size()];
        std::vector<uint8_t> m = it.enc;
        bool structured = (r.next() % 4) == 0;
        if (structured) {
            tv::Stream st = tv::parse(m);
            if (st.ok && !st.segs.empty()) {
                size_t gi = (size_t)(r.next() % st.segs.size());
                tv::Body y = tv::parse_body(st.segs[gi].body, st.segs[gi].seg_n);
                const uint64_t L = y.ok && y.k ? st.segs[gi].seg_n / y.k : 0;
                static const uint64_t EDGE[] = { 0, 1, 2, 3, 63, 64, 65, 131071, 131072, 131073, 262143, 262144, (uint64_t)1 << 32, ~(uint64_t)0 };
                const uint64_t edge = EDGE[r.next() % (sizeof EDGE / sizeof EDGE[0])];
                switch (r.next() % 9) {
                    case 0: st.nseg = edge; break;
                    case 1: st.segs[gi].seg_n = edge; break;
                    case 2: if (!st.segs[gi].body.empty()) st.segs[gi].body.resize((size_t)(edge % (st.segs[gi].body.size() + 4))); break;
                    case 3: if (!st.segs[gi].body.empty()) st.segs[gi].body[0] = (uint8_t)(r.next() % 8); break;                    // k
                    case 4: if (st.segs[gi].body.size() > 2) st.segs[gi].body[1] = (uint8_t)r.next(); break;                        // flags
                    case 5: if (st.segs[gi].body.size() > 2) st.segs[gi].body[2] = (uint8_t)r.next(); break;                        // hflags
                    case 6: if (y.ok) { y.clen[r.next() % y.k] = (r.next() & 1) ? edge : (L ? L + (r.next() % 3) - 1 : edge); st.segs[gi].body = tv::build_body(y); } break;
                    case 8: {   // corrupt only the DECLARED seg_len, leaving the body as it is
                        std::vector<uint8_t> raw; tv::wr_varint(raw, st.nseg);
                        for (size_t q = 0; q < st.segs.size(); q++) {
                            tv::wr_varint(raw, st.segs[q].seg_n);
                            uint64_t declared = st.segs[q].body.size();
                            if (q == gi) declared = (r.next() & 1) ? edge : declared + 1 + (r.next() % 3);
                            tv::wr_varint(raw, declared);
                            raw.insert(raw.end(), st.segs[q].body.begin(), st.segs[q].body.end());
                        }
                        st.segs.clear(); st.nseg = 0; m = raw; structured_iters++; break;
                    }
                    default: if (y.ok) { size_t t = (size_t)(edge % 8); y.tail.assign(t, 0x11); st.segs[gi].body = tv::build_body(y); } break;
                }
                if (!st.segs.empty()) { m = tv::build(st); structured_iters++; }
            }
        }
        const unsigned nmut = structured ? 0u : 1 + (unsigned)(r.next() % 8);
        for (unsigned q = 0; q < nmut && !m.empty(); q++) {
            switch (r.next() % 6) {
                case 0: m[r.next() % m.size()] ^= (uint8_t)(1u << (r.next() % 8)); break;
                case 1: m[r.next() % m.size()] = (uint8_t)r.next(); break;
                case 2: m.resize(r.next() % m.size()); break;
                case 3: m.insert(m.begin() + (r.next() % m.size()), (uint8_t)r.next()); break;
                case 4: m.erase(m.begin() + (r.next() % m.size())); break;
                default: { size_t a = r.next() % m.size(), b = r.next() % m.size(); std::swap(m[a], m[b]); }
            }
        }
        size_t n = it.in.size();
        const unsigned pick = (unsigned)(r.next() % 16);
        if (pick == 0) n = 64 + r.next() % (1u << 20);          // wrong declared length
        else if (pick == 1) n = r.next() % 64;                  // below the minimum
        if (n + 2 * GUARD_N > buf.size()) { buf.assign(n + 2 * GUARD_N, GUARD); buf2.assign(n + 2 * GUARD_N, GUARD); }
        std::fill(buf.begin(), buf.end(), GUARD); std::fill(buf2.begin(), buf2.end(), GUARD);
        uint8_t* dst = buf.data() + GUARD_N;
        uint8_t* dst2 = buf2.data() + GUARD_N;

        bool a = pe::decode(dst, n, m.data(), m.size());
        bool b = pe_ref::decode(dst2, n, m.data(), m.size());
        size_t capi = pe_decompress(dst, n, m.data(), m.size());   // same buffer: it must rewrite it identically
        iters++;

        if (!guards_intact(buf, n) || !guards_intact(buf2, n)) {
            printf("FAIL: a decode wrote outside dst[0..%zu) (stream %zu B)\n", n, m.size()); bad++; break;
        }
        if (a != b) { printf("FAIL: optimised %s but reference %s (n=%zu, stream %zu B)\n", a ? "accepted" : "rejected", b ? "accepted" : "rejected", n, m.size()); bad++; break; }
        if (a && std::memcmp(dst, dst2, n) != 0) { printf("FAIL: both accepted but decoded differently (n=%zu)\n", n); bad++; break; }
        if (capi != (a ? n : 0)) { printf("FAIL: C API returned %zu, C++ API %s (n=%zu)\n", capi, a ? "accepted" : "rejected", n); bad++; break; }

        if (a) { ok++; if (n != it.in.size() || std::memcmp(dst, it.in.data(), n) != 0) ok_changed++; } else rejected++;
    }
    printf("iterations %lu (%lu field-aware): rejected %lu, accepted %lu (of which %lu decoded to different bytes -- expected for raw-plane mutations; there is no checksum by design)\n", iters, structured_iters, rejected, ok, ok_changed);
    if (bad) { printf("FUZZ_FAILED\n"); return 1; }
    printf("FUZZ_OK\n");
    return 0;
}
