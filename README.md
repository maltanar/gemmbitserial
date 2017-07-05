# gemmbitserial

gemmbitserial is a simple, header-only C++ library for fast multiplication of few-bit integer matrices. It is primarily intended for running quantized neural network inference on CPUs, which require fast few-bit integer matrix multiplication.

Documentation is currently underway, and all contributions/suggestions are welcome.

## Preliminaries

It computes the product between two integer matrices A and B, either signed or unsigned. For 1-bit matrices, it supports both bipolar {-1, +1} and regular unsigned {0, 1} encoding. The input matrices must be first converted into bit-serial form via the importRegular function call.
Note that the right-hand-side matrix must be provided in transposed (column-major) form, and the result is also produced in transposed form.
A short paper regarding the underlying operation principle can be found [here](http://www.idi.ntnu.no/~yamanu/2017-cases-wip-quantizedmm-preprint.pdf).

## Quickstart
1) Import "gemmbitserial.h"
2) Instantiate a GEMMContext by using the allocateGEMMContext function.
2) Import left-hand-side and right-hand-side matrices by calling gemmcontext.lhs.importRegular and gemmcontext.rhs.importRegular.
3) Call the gemmBitSerial function with the gemmcontext as the argument.
4) Done! You can now read out the result from gemmcontext.res
5) Release the context by calling deallocGEMMContext.

## Running benchmarks
There is a little benchmarking tool to quickly evaluate performance on different matrix sizes and bitwidths. So far the code paths are only optimized for ARM and x86 to some extent. To build the benchmarking tool:

For Android: ndk-build in the root directory, then adb push to e.g. /data/local/tmp and run from there
For x86: cd benchmark; ./benchmark.sh

Once the interactive benchmark is running, it will ask for the following parameters from stdin:
rows depth columns lhs_bitwidth rhs_bitwidth lhs_signed rhs_signed number_of_seconds_to run

For instance, entering the following in stdin will run a 8x8192x8 binary unsigned matrix multiply for 20 seconds, and report the GOPS:
8 8192 8 1 1 0 0 20
