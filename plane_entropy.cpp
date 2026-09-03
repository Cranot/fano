// plane_entropy.cpp - C API implementation.
// Copyright 2025 Cranot. Licensed under the Apache License, Version 2.0. See LICENSE.
//
// The wrappers reject NULL buffers and turn an allocation failure into the documented 0 return, so
// every failure a caller can provoke through this ABI is a return value rather than a trap.
#include "plane_entropy.h"
#include "plane_entropy.hpp"
#include <new>

extern "C" size_t pe_compress_bound(size_t n) { return pe::compress_bound(n); }

extern "C" size_t pe_compress(void* dst, size_t dst_cap, const void* src, size_t n) {
    if (!src || (!dst && dst_cap)) return 0;
    try {
        std::vector<uint8_t> s = pe::encode((const uint8_t*)src, n);
        if (s.empty() || s.size() > dst_cap) return 0;
        std::memcpy(dst, s.data(), s.size());
        return s.size();
    } catch (const std::bad_alloc&) { return 0; }
}

extern "C" size_t pe_decompress(void* dst, size_t n, const void* src, size_t sz) {
    if (!src || (!dst && n)) return 0;
    try {
        return pe::decode((uint8_t*)dst, n, (const uint8_t*)src, sz) ? n : 0;
    } catch (const std::bad_alloc&) { return 0; }
}
