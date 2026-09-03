# plane-entropy - Makefile. Copyright 2025 Cranot. Apache-2.0.
CXX      ?= g++
CC       ?= gcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
CFLAGS   ?= -O3 -Wall
FSE_DIR   = third_party/fse
FSE_SRC   = $(FSE_DIR)/huf_compress.c $(FSE_DIR)/huf_decompress.c $(FSE_DIR)/fse_compress.c $(FSE_DIR)/fse_decompress.c $(FSE_DIR)/entropy_common.c $(FSE_DIR)/hist.c $(FSE_DIR)/debug.c
FSE_OBJ   = $(FSE_SRC:.c=.o)
FSE_ASAN  = $(FSE_SRC:.c=.asan.o)
INC       = -I. -I$(FSE_DIR)
SAN       = -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer

all: libplane_entropy.a test_pe bench_pe

$(FSE_DIR)/%.o: $(FSE_DIR)/%.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(FSE_DIR)/%.asan.o: $(FSE_DIR)/%.c
	$(CC) $(SAN) $(INC) -c $< -o $@

plane_entropy.o: plane_entropy.cpp plane_entropy.hpp plane_entropy.h
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

libplane_entropy.a: plane_entropy.o $(FSE_OBJ)
	ar rcs $@ $^

test_pe: test_pe.cpp test_stream.hpp plane_entropy.hpp plane_entropy.h reference/plane_entropy_ref.hpp libplane_entropy.a
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< libplane_entropy.a

bench_pe: bench_pe.cpp plane_entropy.hpp libplane_entropy.a
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< libplane_entropy.a

fuzz_pe: fuzz_pe.cpp test_stream.hpp plane_entropy.h plane_entropy.cpp plane_entropy.hpp reference/plane_entropy_ref.hpp $(FSE_ASAN)
	$(CXX) $(SAN) -std=c++17 $(INC) -o $@ fuzz_pe.cpp plane_entropy.cpp $(FSE_ASAN)

test_pe_asan: test_pe.cpp test_stream.hpp plane_entropy.h plane_entropy.cpp plane_entropy.hpp reference/plane_entropy_ref.hpp $(FSE_ASAN)
	$(CXX) $(SAN) -std=c++17 $(INC) -o $@ test_pe.cpp plane_entropy.cpp $(FSE_ASAN)

# Portability check: EVERY SIMD path compiles out (PE_NO_SIMD also removes the SSE2 blocks, which
# -mno-avx does not) and the scalar fallbacks must give the same bytes, including the frozen vectors.
test_pe_nosimd: test_pe.cpp test_stream.hpp plane_entropy.h plane_entropy.hpp plane_entropy.cpp reference/plane_entropy_ref.hpp $(FSE_SRC)
	$(CXX) -O2 -std=c++17 -DPE_NO_SIMD -mno-avx2 -mno-avx $(INC) -o $@ test_pe.cpp plane_entropy.cpp $(FSE_SRC)
test_pe_noavx: test_pe_nosimd
	cp $< $@

check: test_pe
	./test_pe

check-asan: test_pe_asan
	./test_pe_asan

fuzz: fuzz_pe
	./fuzz_pe 60

check-nosimd: test_pe_nosimd
	./test_pe_nosimd

clean:
	rm -f *.o *.a test_pe test_pe_asan test_pe_nosimd test_pe_noavx bench_pe fuzz_pe $(FSE_DIR)/*.o

.PHONY: all check check-asan check-nosimd fuzz clean
