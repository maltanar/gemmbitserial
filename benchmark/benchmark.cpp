#include <cassert>
#include <iostream>
#include <chrono>
#include "gemm-bitserial.h"
#include "libpopcnt.h"

using namespace std;

void benchmark_unrolledpopcount(size_t numBits, float secs) {
  string bench_name = "popcount";
  double opcount = numBits;
  uint8_t * rnd_vec = new uint8_t[numBits/8];
  uint64_t * ppcnt = (uint64_t *) rnd_vec;
  uint64_t res_pcnt = 0;
  cout << "======================================================================" << endl;
  cout << numBits << "-bit popcount using libpopcnt::popcnt64, for " << secs << " seconds..." << endl;
  unsigned int reps = 0;
  auto start = chrono::high_resolution_clock::now();
  auto end = chrono::high_resolution_clock::now();
  while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
    // =============== start of benchmark kernel =============
    res_pcnt = 0;
    for(unsigned int r = 0; r < numBits/64; r+=4) {
      res_pcnt += popcnt64(ppcnt[r]);
      res_pcnt += popcnt64(ppcnt[r+1]);
      res_pcnt += popcnt64(ppcnt[r+2]);
      res_pcnt += popcnt64(ppcnt[r+3]);
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



int main(int argc, char const *argv[]) {

  vector<size_t> bits {1, 2, 3, 4};
  vector<size_t> mdims {256, 1024, 8192};

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

  for(auto &bm: bits) {
    for(auto &bv: bits) {
      for(auto &r: mdims) {
        for(auto &c: mdims) {
          benchmark_matvec(r, c, bm, bv, 5);
        }
      }
    }
  }

  vector<size_t> dims {256, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  for(auto &d: dims) {
    benchmark_andcardinality(d, 5);
    benchmark_unrolledpopcount(d, 5);
  }



  /*size_t b = 2;
  size_t d = 8192;
  size_t thres_levels = 3;
  size_t reps = 1000;
  size_t reps_mm = 10;
  uint8_t * rnd_mat = new uint8_t[d*d];
  uint8_t * rnd_vec = new uint8_t[d];
  assert(rnd_mat != 0 && rnd_vec != 0);
  generateRandomVector(b, d, rnd_vec);
  generateRandomVector(b, d*d, rnd_mat);
  BitSerialVector bsv = toBitSerialVector(rnd_vec, d, b);
  BitSerialVector bsv2 = toBitSerialVector(rnd_vec, d, b);
  BitSerialMatrix bsm = toBitSerialMatrix(rnd_mat, d, d, b);

  // unrolled builtin_popcount
  cout << "======================================================================" << endl;
  uint64_t res_pcnt = 0;
  cout << reps << " reps of " << d*8 << "-bit popcount using 8-unrolled popcnt64..." << endl << flush;
  start = chrono::high_resolution_clock::now();

  for(unsigned int i = 0; i < reps; i++) {
    res_pcnt = 0;
    for(unsigned int r = 0; r < d/8; r+=8) {
      res_pcnt += popcnt64(ppcnt[r]);
      res_pcnt += popcnt64(ppcnt[r+1]);
      res_pcnt += popcnt64(ppcnt[r+2]);
      res_pcnt += popcnt64(ppcnt[r+3]);
      res_pcnt += popcnt64(ppcnt[r+4]);
      res_pcnt += popcnt64(ppcnt[r+5]);
      res_pcnt += popcnt64(ppcnt[r+6]);
      res_pcnt += popcnt64(ppcnt[r+7]);
    }
  }
  end = chrono::high_resolution_clock::now();
  cout << "Returned result: " << res_pcnt << endl;
  nscount = chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / (double)reps;
  perf = (d*8) / (nscount); // billion bit operations per second
  cout << "Time for a single " << d*8 <<"-bit popcount: " << nscount << " nanoseconds" << endl;
  cout << "Performance for popcount: " << perf << " GOPS per second" << endl;
  cout << d / (nscount) << " GB/s" << endl;

  // and_cardinality
  cout << "======================================================================" << endl;
  start = chrono::high_resolution_clock::now();
  uint64_t res_card;
  for(unsigned int i = 0; i < reps; i++)
    res_card = bsv[0].and_cardinality(bsv2[0]);
  end = chrono::high_resolution_clock::now();
  float uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  perf = 1000000 * (d*2 / uscount);
  cout << "Time for and_cardinality: " << uscount << " microseconds" << endl;
  cout << "Performance for and_cardinality: " << perf/1000000000.0 << " GOPS per second" << endl;
  // matrix-vector
  cout << "======================================================================" << endl;
  AccumulateVector resvec;
  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < reps; i++)
    resvec = bitSerialMatrixVector(bsm, bsv);
  end = chrono::high_resolution_clock::now();
  uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  perf = 1000000 * (d*d*b*b*2 / uscount);
  cout << "Time for single matrix-vector: " << uscount << " microseconds" << endl;
  cout << "Performance for matrix-vector: " << perf/1000000000.0 << " binary GOPS per second" << endl;
  cout << "Performance for matrix-vector: " << perf/(b*b*1000000000.0) << " lowprec (" << b << " bit) GOPS per second" << endl;
  // matrix-matrix
  cout << "======================================================================" << endl;
  start = chrono::high_resolution_clock::now();
  AccumulateMatrix resmat;
  for(unsigned int i = 0; i < reps_mm; i++)
    resmat = bitSerialMatrixMatrix(bsm, bsm);
  end = chrono::high_resolution_clock::now();
  uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps_mm;
  perf = 1000000 * (d*d*d*b*b*2 / uscount);
  cout << "Time for single matrix-matrix: " << uscount << " microseconds" << endl;
  cout << "Performance for matrix-matrix: " << perf/1000000000.0 << " binary GOPS per second" << endl;
  cout << "Performance for matrix-matrix: " << perf/(b*b*1000000000.0) << " lowprec (" << b << " bit) GOPS per second" << endl;
  // matrix-vector-threshold
  ThresholdMatrix T;

  for(size_t l = 0; l < thres_levels; l++) {
    AccumulateVector currentT;
    for(size_t t = 0; t < d; t++) {
      currentT.push_back(rand() % d);
    }
    T.push_back(currentT);
  }
  ResultVector thres_resvec;
  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < reps; i++)
    thres_resvec = bitSerialMatrixVectorThreshold(bsm, bsv, T);
  end = chrono::high_resolution_clock::now();
  uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  perf = 1000000 * (d*d*b*b*2 / uscount);
  cout << "Time for single matrix-vector-threshold: " << uscount << " microseconds" << endl;
  cout << "Performance for matrix-vector-threshold, not counting threshold ops: " << perf/1000000000.0 << " binary GOPS per second" << endl;
  cout << "Performance for matrix-vector-threshold, not counting threshold ops: " << perf/(b*b*1000000000.0) << " lowprec (" << b << " bit) GOPS per second" << endl;

  delete [] rnd_mat;
  delete [] rnd_vec;*/
  return 0;
}
