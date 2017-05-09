#!/bin/bash

g++ -std=c++11 -O3 -march=native *.cpp -fPIC -c
g++ -shared *.o -o libgemmbitserial.so
