#!/bin/sh

LHS="8 16 32 64 128 256 512 1024 2048 4096 8192"
RHS="8 16 32 64 128 256 512 1024 2048 4096 8192"
DEPTH="256 512 1024 2048 4096 8192 16384"

for lhs in $LHS; do
  for rhs in $RHS; do
    for depth in $DEPTH; do
      echo "$lhs $depth $rhs 1 1 0 0 5 -1" | ./benchmark.out 
    done
  done
done
