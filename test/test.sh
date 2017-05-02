#!/bin/bash

if [ -f test.out ]; then
    rm test.out
fi
g++ -march=native -O3 -std=c++11 -I.. test.cpp ../*.cpp -o test.out
./test.out
