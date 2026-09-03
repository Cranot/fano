/* plane_entropy.h - C API for the byte-plane entropy coder (see plane_entropy.hpp, SPEC.md).
 *
 * Copyright 2025 Cranot
 * Licensed under the Apache License, Version 2.0. See LICENSE.
 */
#ifndef PLANE_ENTROPY_H
#define PLANE_ENTROPY_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Upper bound on the compressed size of n input bytes. */
size_t pe_compress_bound(size_t n);

/* Compress src[0..n) into dst (capacity dst_cap). Returns the stream size, or 0 on failure
 * (n < 64, dst_cap too small). The stream may be larger than n on incompressible input; the
 * caller decides whether to store it or store raw, as Xet's scheme selection does. */
size_t pe_compress(void* dst, size_t dst_cap, const void* src, size_t n);

/* Decompress a stream of sz bytes into dst[0..n); n must be the original length. Returns n on
 * success, 0 on any malformed input. Never reads outside src[0..sz) or writes outside dst[0..n). */
size_t pe_decompress(void* dst, size_t n, const void* src, size_t sz);

#ifdef __cplusplus
}
#endif
#endif
