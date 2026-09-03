// bench_ref.cpp - interleaved A/B of the reference and optimised coders on one core, same data,
// alternating arms per rep so drift and thermal effects hit both equally. Asserts identical bytes.
//   ./bench_ref <file> [chunk=65536] [reps=5]
// Copyright 2025 Cranot. Apache-2.0.
#include "plane_entropy.hpp"
#include "reference/plane_entropy_ref.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
static std::vector<uint8_t> rd(const char* p) { std::ifstream f(p, std::ios::binary | std::ios::ate); size_t n = (size_t)f.tellg(); f.seekg(0); std::vector<uint8_t> v(n); f.read((char*)v.data(), n); return v; }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
    if (argc < 2) return 1;
    auto in = rd(argv[1]); const size_t CS = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 65536; const int R = argc > 3 ? std::atoi(argv[3]) : 5; const size_t N = in.size();
    std::vector<double> re(R), oe(R), rdec(R), odec(R);
    size_t rb = 0, ob = 0; bool same = true, rt = true;
    std::vector<std::vector<uint8_t>> outs; std::vector<uint8_t> dec(CS);
    for (int r = 0; r < R; r++) {
        // reference encode
        double t0 = now(); rb = 0; for (size_t i = 0; i < N; i += CS) { size_t n = std::min(CS, N - i); auto s = pe_ref::encode(in.data() + i, n); rb += s.size(); } re[r] = now() - t0;
        // optimised encode (keep the streams for the decode arms)
        outs.clear(); t0 = now(); ob = 0; for (size_t i = 0; i < N; i += CS) { size_t n = std::min(CS, N - i); auto s = pe::encode(in.data() + i, n); ob += s.size(); outs.push_back(std::move(s)); } oe[r] = now() - t0;
        // reference decode
        t0 = now(); { size_t c = 0; for (size_t i = 0; i < N; i += CS, c++) { size_t n = std::min(CS, N - i); auto d = pe_ref::decode(outs[c].data(), outs[c].size(), n); if (d.size() != n || std::memcmp(d.data(), in.data() + i, n)) rt = false; } } rdec[r] = now() - t0;
        // optimised decode
        t0 = now(); { size_t c = 0; for (size_t i = 0; i < N; i += CS, c++) { size_t n = std::min(CS, N - i); if (!pe::decode(dec.data(), n, outs[c].data(), outs[c].size()) || std::memcmp(dec.data(), in.data() + i, n)) rt = false; } } odec[r] = now() - t0;
        if (r == 0) { size_t c = 0; for (size_t i = 0; i < N; i += CS, c++) { size_t n = std::min(CS, N - i); if (pe_ref::encode(in.data() + i, n) != outs[c]) same = false; } }
    }
    for (auto* v : { &re, &oe, &rdec, &odec }) std::sort(v->begin(), v->end());
    const double MB = N / 1048576.0;
    printf("ref %7.1f -> opt %7.1f MB/s enc (%.2fx) | ref %7.1f -> opt %7.1f MB/s dec (%.2fx) | bytes %zu%s | rt %s\n",
           MB / re[R/2], MB / oe[R/2], re[R/2] / oe[R/2], MB / rdec[R/2], MB / odec[R/2], rdec[R/2] / odec[R/2],
           ob, (same && rb == ob) ? " IDENTICAL" : " *** DIFFER ***", rt ? "OK" : "FAIL");
    return (same && rt && rb == ob) ? 0 : 2;
}
