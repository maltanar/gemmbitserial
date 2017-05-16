#include <cassert>
#include <iostream>
#include <chrono>
#include "gemm-bitserial.h"

using namespace std;

void benchmark_unrolledpopcount(size_t numBits, float secs) {
  string bench_name = "popcount";
  double opcount = numBits;
  uint8_t * rnd_vec = new uint8_t[numBits/8];
  uint64_t * ppcnt = (uint64_t *) rnd_vec;
  uint64_t res_pcnt = 0;
  cout << "======================================================================" << endl;
  cout << numBits << "-bit popcount using __builtin_popcntll, for " << secs << " seconds..." << endl;
  unsigned int reps = 0;
  auto start = chrono::high_resolution_clock::now();
  auto end = chrono::high_resolution_clock::now();
  while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
    // =============== start of benchmark kernel =============
    res_pcnt = 0;
    for(unsigned int r = 0; r < numBits/64; r+=4) {
      res_pcnt += __builtin_popcountll(ppcnt[r]);
      res_pcnt += __builtin_popcountll(ppcnt[r+1]);
      res_pcnt += __builtin_popcountll(ppcnt[r+2]);
      res_pcnt += __builtin_popcountll(ppcnt[r+3]);
    }
    // =============== end of benchmark kernel ================
    reps += 1;
    end = chrono::high_resolution_clock::now();
  }
  cout << "Returned result: " << res_pcnt << endl;
  double nscount = chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / (double)reps;
  double perf = opcount / (nscount); // billion bit operations per second
  cout << "Time for a single " << bench_name << ": " << nscount << " nanoseconds" << endl;
  cout << "Performance for " << bench_name << ": " << perf << " GOPS per second" << endl;
  cout << (numBits/8) / (nscount) << " GB/s" << endl;
  delete [] rnd_vec;
}

void benchmark_andcardinality(size_t numBits, float secs) {
  string bench_name = "and_cardinality";
  double opcount = 2*numBits;
  MyBitVector a(numBits), b(numBits);
  cout << "======================================================================" << endl;
  cout << numBits << "-bit " << bench_name << " for " << secs << " seconds..." << endl;
  unsigned int reps = 0;
  uint64_t res = 0;
  auto start = chrono::high_resolution_clock::now();
  auto end = chrono::high_resolution_clock::now();
  while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
    // =============== start of benchmark kernel =============
    res = a.and_cardinality(b);
    // =============== end of benchmark kernel ================
    reps += 1;
    end = chrono::high_resolution_clock::now();
  }
  cout << "Returned result: " << res << endl;
  double nscount = chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / (double)reps;
  double perf = opcount / (nscount); // billion bit operations per second
  cout << "Time for a single " << bench_name << ": " << nscount << " nanoseconds" << endl;
  cout << "Performance for " << bench_name << ": " << perf << " GOPS per second" << endl;
}

void benchmark_matvec(size_t rows, size_t cols, size_t mbits, size_t vbits, float secs) {
  string bench_name = std::to_string(rows)+"x"+std::to_string(cols)+" "+std::to_string(mbits)+"-bit matrix "+std::to_string(vbits) +"-bits vector";
  double opcount = 2*rows*cols;
  uint8_t * rnd_mat = new uint8_t[rows*cols];
  uint8_t * rnd_vec = new uint8_t[cols];
  BitSerialVector bsv = toBitSerialVector(rnd_vec, cols, vbits);
  BitSerialMatrix bsm = toBitSerialMatrix(rnd_mat, rows, cols, mbits);
  AccumulateVector res;

  cout << "======================================================================" << endl;
  cout << bench_name << " for " << secs << " seconds..." << endl;
  unsigned int reps = 0;
  auto start = chrono::high_resolution_clock::now();
  auto end = chrono::high_resolution_clock::now();
  while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
    // =============== start of benchmark kernel =============
    res = bitSerialMatrixVector(bsm, bsv);
    // =============== end of benchmark kernel ================
    reps += 1;
    end = chrono::high_resolution_clock::now();
  }
  cout << "Size of returned result: " << res.size() << endl;
  double nscount = chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / (double)reps;
  double perf = opcount / (nscount); // billion bit operations per second
  cout << "Time for a single " << bench_name << ": " << nscount << " nanoseconds" << endl;
  cout << "Performance for " << bench_name << ": " << perf << " GOPS per second" << endl;
  delete [] rnd_mat;
  delete [] rnd_vec;
}

void benchmark_matmat(size_t rowsA, size_t colsA, size_t colsB, size_t abits, size_t bbits, float secs) {
  string bench_name = std::to_string(rowsA)+"x"+std::to_string(colsA)+" "+std::to_string(abits)+"-bit matrix times ";
  bench_name += std::to_string(colsA)+"x"+std::to_string(colsB)+" "+std::to_string(bbits)+"-bit matrix ";
  double opcount = 2*rowsA*colsA*colsB;
  uint8_t * rnd_matA = new uint8_t[rowsA*colsA];
  uint8_t * rnd_matB = new uint8_t[colsA*colsB];
  BitSerialMatrix bsmA = toBitSerialMatrix(rnd_matA, rowsA, colsA, abits);
  // bitSerialMatrixMatrix wants matrix B to be transposed, so rows and cols switched
  BitSerialMatrix bsmB = toBitSerialMatrix(rnd_matB, colsB, colsA, bbits);
  AccumulateMatrix res;

  cout << "======================================================================" << endl;
  cout << bench_name << " for " << secs << " seconds..." << endl;
  unsigned int reps = 0;
  auto start = chrono::high_resolution_clock::now();
  auto end = chrono::high_resolution_clock::now();
  while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
    // =============== start of benchmark kernel =============
    res = bitSerialMatrixMatrix(bsmA, bsmB);
    // =============== end of benchmark kernel ================
    reps += 1;
    end = chrono::high_resolution_clock::now();
  }
  cout << "Size of returned result: " << res.size() << endl;
  double nscount = chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / (double)reps;
  double perf = opcount / (nscount); // billion bit operations per second
  cout << "Time for a single " << bench_name << ": " << nscount << " nanoseconds" << endl;
  cout << "Performance for " << bench_name << ": " << perf << " GOPS per second" << endl;
}

void benchmark_caffenet(float secs) {
  string bench_name = "CaffeNet matrices";
  const int caffenet_gemm_sizes[] = {
      96, 363, 3025,
      256, 2400, 729,
      384, 2304, 169,
      384, 3456, 169,
      256, 3456, 169,
      4096, 9216, 1,
      4096, 4096, 1,
      1000, 4096, 1
  };
  double opcount = 0;
  size_t wbits = 2;
  size_t abits = 2;
  const std::size_t num_caffenet_gemms =
      sizeof(caffenet_gemm_sizes) / (3 * sizeof(caffenet_gemm_sizes[0]));
  // prepare workload
  std::vector<BitSerialMatrix> caffenet_gemms_A, caffenet_gemms_B;
  for (std::size_t i = 0; i < num_caffenet_gemms; i++) {
    size_t rows = caffenet_gemm_sizes[3 * i + 0];
    size_t depth = caffenet_gemm_sizes[3 * i + 1];
    size_t cols = caffenet_gemm_sizes[3 * i + 2];
    opcount += 2*rows*depth*cols;
    uint8_t * rnd_matA = new uint8_t[rows*depth];
    uint8_t * rnd_matB = new uint8_t[depth*cols];
    generateRandomVector(wbits, rows*depth, rnd_matA);
    generateRandomVector(abits, depth*cols, rnd_matB);
    caffenet_gemms_A.push_back(toBitSerialMatrix(rnd_matA, rows, depth, wbits));
    caffenet_gemms_B.push_back(toBitSerialMatrix(rnd_matB, cols, depth, abits));
    delete [] rnd_matA;
    delete [] rnd_matB;
  }
  AccumulateMatrix res;

  cout << "======================================================================" << endl;
  cout << bench_name << " for " << secs << " seconds..." << endl;
  unsigned int reps = 0;
  auto start = chrono::high_resolution_clock::now();
  auto end = chrono::high_resolution_clock::now();
  while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
    // =============== start of benchmark kernel =============
    for(size_t i = 0; i < num_caffenet_gemms; i++) {
      res = bitSerialMatrixMatrix(caffenet_gemms_A[i], caffenet_gemms_B[i]);
    }
    // =============== end of benchmark kernel ================
    reps += 1;
    end = chrono::high_resolution_clock::now();
    // ignore the first iteration, it's just for warmup
    if(reps == 1) {
      start = end;
    }
  }
  cout << "Completed " << reps << " iterations" << endl;
  cout << "Size of returned result: " << res.size() << endl;
  double nscount = chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / (double)reps;
  double perf = opcount / (nscount); // billion bit operations per second
  cout << "Time for a single " << bench_name << ": " << nscount << " nanoseconds" << endl;
  cout << "Performance for " << bench_name << ": " << perf << " GOPS per second" << endl;
}



int main(int argc, char const *argv[]) {
  vector<size_t> bits {1, 2, 3, 4};
  vector<size_t> mdims {256, 1024, 8192};
  vector<size_t> dims {256, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

  benchmark_caffenet(20);

  for(auto &d: dims) {
    benchmark_andcardinality(d, 5);
  }

  for(auto &d: dims) {
    benchmark_unrolledpopcount(d, 5);
  }

  for(auto &bm: bits) {
    for(auto &bv: bits) {
      for(auto &r: mdims) {
        for(auto &c: mdims) {
          benchmark_matvec(r, c, bm, bv, 5);
        }
      }
    }
  }
  for(auto &ba: bits) {
    for(auto &bb: bits) {
      for(auto &ra: mdims) {
        for(auto &ca: mdims) {
          for(auto &cb: mdims) {
            benchmark_matmat(ra, ca, cb, ba, bb, 5);
          }
        }
      }
    }
  }
  return 0;
}
