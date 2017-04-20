#!/bin/bash

if [ -f benchmark.out ]; then
    rm benchmark.out
fi
g++ -march=native -pg -DUSEAVX -O3 -std=c++11 benchmark.cpp gemm-bitserial.cpp roaring.c -o benchmark.out
./benchmark.out
