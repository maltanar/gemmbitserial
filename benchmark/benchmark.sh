#!/bin/bash

if [ -f benchmark.out ]; then
    rm benchmark.out
fi
g++ -march=native -O3 -std=c++11 -I.. benchmark.cpp -o benchmark.out
./benchmark.out
