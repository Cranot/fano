// bench_pe.cpp - in-process throughput of the coder on a file, chunked as a content store would.
//   ./bench_pe <file> [chunk=65536] [reps=5]
// Prints bytes, ratio, encode MB/s, decode MB/s (medians of reps, single thread), round-trip
// status, and the k / coded-plane statistics. Pin to a core (taskset) and report the load.
// Copyright 2025 Cranot. Licensed under the Apache License, Version 2.0. See LICENSE.
#include "plane_entropy.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static std::vector<uint8_t> rd(const char* p) { std::ifstream f(p, std::ios::binary | std::ios::ate); if (!f) return {}; size_t n = (size_t)f.tellg(); f.seekg(0); std::vector<uint8_t> v(n); f.read((char*)v.data(), n); return v; }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_pe <file> [chunk=65536] [reps=5]\n"); return 1; }
    auto in = rd(argv[1]); if (in.empty()) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    const size_t CS = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 65536; const int R = argc > 3 ? std::atoi(argv[3]) : 5; const size_t N = in.size();
    std::vector<double> te(R), td(R); size_t total = 0, raw_chunks = 0, k2 = 0, k4 = 0, coded_planes = 0, planes = 0; bool ok = true;
    std::vector<std::vector<uint8_t>> outs; outs.reserve(N / CS + 1);
    std::vector<uint8_t> dec(CS);
    for (int rep = 0; rep < R; rep++) {
        outs.clear(); total = 0; raw_chunks = k2 = k4 = coded_planes = planes = 0;
        double t0 = now();
        for (size_t i = 0; i < N; i += CS) { size_t n = std::min(CS, N - i); auto s = pe::encode(in.data() + i, n);
            if (s.empty() || s.size() >= n) { raw_chunks++; total += n; outs.emplace_back(); }   // store raw (scheme None), as a store would
            else { total += s.size(); outs.push_back(std::move(s)); } }
        te[rep] = now() - t0;
        t0 = now(); size_t c = 0;
        for (size_t i = 0; i < N; i += CS, c++) { size_t n = std::min(CS, N - i); if (outs[c].empty()) continue;
            if (!pe::decode(dec.data(), n, outs[c].data(), outs[c].size()) || std::memcmp(dec.data(), in.data() + i, n) != 0) ok = false; }
        td[rep] = now() - t0;
    }
    // statistics from the last rep's streams (parse: nseg varint, then per segment sn, len, k, flags)
    for (auto& s : outs) { if (s.empty()) continue; const uint8_t* p = s.data(); const uint8_t* e = p + s.size(); uint64_t nseg, sn, cl;
        if (!pe::get_varint(p, e, nseg)) continue;
        for (uint64_t g = 0; g < nseg && p < e; g++) { if (!pe::get_varint(p, e, sn) || !pe::get_varint(p, e, cl) || p + cl > e) break; unsigned k = p[0]; uint8_t flags = p[1]; (k == 2 ? k2 : k4)++; planes += k; for (unsigned j = 0; j < k; j++) coded_planes += (flags >> j) & 1; p += cl; } }
    std::sort(te.begin(), te.end()); std::sort(td.begin(), td.end());
    printf("%s: %zu B in %zu chunks of %zu\n", argv[1], N, (N + CS - 1) / CS, CS);
    printf("  bytes %zu  ratio %.4f  encode %.1f MB/s  decode %.1f MB/s  round-trip %s\n", total, (double)N / total, N / 1048576.0 / te[R / 2], N / 1048576.0 / td[R / 2], ok ? "OK" : "FAIL");
    printf("  chunks stored raw %zu; segments k=2 %zu, k=4 %zu; planes coded %zu of %zu\n", raw_chunks, k2, k4, coded_planes, planes);
    return ok ? 0 : 2;
}
