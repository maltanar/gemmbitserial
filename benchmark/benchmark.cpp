#include <cassert>
#include <iostream>
#include <chrono>
#include <vector>
#include "gemmbitserial.hpp"

using namespace std;
using namespace gemmbitserial;

/**
* Generate a random vector with given dimension and number of bits <= 8
*/
template <typename T>
void generateRandomVector(size_t bits, size_t dim, T* ret) {
  uint8_t minVal = 0;
  uint8_t maxVal = (1 << bits);
  for(size_t i = 0; i < dim; i++) {
    ret[i] = (T) (rand() % maxVal);
  }
}

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

void benchmark_gemm_interactive() {
  while(1) {
    int rows, depth, cols, lhsbits, rhsbits, lhssigned, rhssigned;
    float secs;
    cout << "Enter rows depth cols, 0 for next benchmark, -1 to exit " << endl;
    cin >> rows;
    if(rows == 0) {
      break;
    } else if (rows < 0) {
      exit(0);
    }
    cin >> depth >> cols;
    cout << "Enter lhs and rhs bits: " << endl;
    cin >> lhsbits >> rhsbits;
    cout << "Enter signedness (1 or 0) for lhs and rhs: " << endl;
    cin >> lhssigned >> rhssigned;
    cout << "Enter number of seconds to benchmark: " << endl;
    cin >> secs;
    // prepare workload
    uint8_t * rnd_matA = new uint8_t[rows*depth];
    uint8_t * rnd_matB = new uint8_t[depth*cols];
    int32_t * res = new int32_t[rows*cols];
    generateRandomVector(lhsbits, rows*depth, rnd_matA);
    generateRandomVector(rhsbits, depth*cols, rnd_matB);

    GEMMContext ctx = allocGEMMContext(rows, depth, cols, lhsbits, rhsbits, (bool) lhssigned, (bool) rhssigned);
    ctx.lhs.importRegular(rnd_matA);
    ctx.rhs.importRegular(rnd_matB);
    ctx.printSummary();


    delete [] rnd_matA;
    delete [] rnd_matB;
    cout << "======================================================================" << endl;
    char bench_name[1024];
    sprintf(bench_name, "gemm-%d x %d x %d (%d bit x %d bit)", rows, depth, cols, lhsbits, rhsbits);
    cout << "Running " << bench_name << " for " << secs << " seconds..." << endl;
    unsigned int reps = 0;
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
      // =============== start of benchmark kernel =============
      gemmBitSerial(ctx);
      // =============== end of benchmark kernel ================
      reps += 1;
      end = chrono::high_resolution_clock::now();
      // ignore the first iteration, it's just for warmup
      if(reps == 1) {
        start = end;
      }
    }
    cout << "Completed " << reps << " iterations" << endl;
    float opcount = 2.0*(float)rows*(float)depth*(float)cols;
    float nscount = chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / (float)reps;
    float perf = opcount / nscount; // billion bit operations per second
    cout << "Time for a single " << bench_name << ": " << nscount << " nanoseconds" << endl;
    cout << "Performance for " << bench_name << ": " << perf << " GOPS per second" << endl;

    deallocGEMMContext(ctx);
    delete [] res;
  }
}

void benchmark_import_interactive() {
  string bench_name = "Regular-to-bitserial conversion";
  while(1) {
    int rows, cols, nbits;
    float secs;
    cout << "Enter rows cols nbits, 0 for next benchmark, -1 to exit " << endl;
    cin >> rows;
    if(rows == 0) {
      break;
    } else if (rows < 0) {
      exit(0);
    }
    cin >> cols >> nbits;
    int nthres = (1 << nbits) - 1;
    cout << "Enter 0 for regular import, 1 for thresholding import: " << endl;
    int use_thres;
    cin >> use_thres;
    if(use_thres) {
      cout << "Benchmark will use " << nthres << " thresholds" << endl;
    }
    cout << "Enter 0 for no transpose, 1 for tranposed import: " << endl;
    int do_transpose;
    cin >> do_transpose;
    cout << "Enter number of seconds to benchmark: " << endl;
    cin >> secs;
    BitSerialMatrix bsm = BitSerialMatrix::alloc(nbits, rows, cols, false);
    uint8_t * rand_mat = new uint8_t[rows*cols];
    uint8_t * rand_thres = new uint8_t[rows*nthres];
    generateRandomVector(nbits, rows*cols, rand_mat);
    generateRandomVector(nbits, rows*nthres, rand_thres);
    unsigned int reps = 0;
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
      // =============== start of benchmark kernel =============
      if(use_thres) {
        bsm.importRegularAndQuantize(rand_mat, rand_thres, nthres, (bool) do_transpose);
      } else {
        bsm.importRegular(rand_mat, (bool) do_transpose);
      }
      // =============== end of benchmark kernel ================
      reps += 1;
      end = chrono::high_resolution_clock::now();
      // ignore the first iteration, it's just for warmup
      if(reps == 1) {
        start = end;
      }
    }
    float mscount = chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / (float)reps;
    cout << "Completed " << reps << " iterations, " << mscount << " ms per iteration" << endl;
    BitSerialMatrix::dealloc(bsm);
    delete [] rand_mat;
    delete [] rand_thres;
  }
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
  vector<GEMMContext> caffenet_gemms;
  for (std::size_t i = 0; i < num_caffenet_gemms; i++) {
    size_t rows = caffenet_gemm_sizes[3 * i + 0];
    size_t depth = caffenet_gemm_sizes[3 * i + 1];
    size_t cols = caffenet_gemm_sizes[3 * i + 2];
    opcount += 2*rows*depth*cols;
    uint8_t * rnd_matA = new uint8_t[rows*depth];
    uint8_t * rnd_matB = new uint8_t[depth*cols];
    generateRandomVector(wbits, rows*depth, rnd_matA);
    generateRandomVector(abits, depth*cols, rnd_matB);
    GEMMContext g = allocGEMMContext(rows, depth, cols, wbits, abits, false, false);
    g.lhs.importRegular(rnd_matA);
    g.rhs.importRegular(rnd_matB);
    caffenet_gemms.push_back(g);
    delete [] rnd_matA;
    delete [] rnd_matB;
  }

  cout << "======================================================================" << endl;
  cout << bench_name << " for " << secs << " seconds..." << endl;
  unsigned int reps = 0;
  auto start = chrono::high_resolution_clock::now();
  auto end = chrono::high_resolution_clock::now();
  while (chrono::duration_cast<std::chrono::seconds>(end-start).count() < secs) {
    // =============== start of benchmark kernel =============
    for(size_t i = 0; i < num_caffenet_gemms; i++) {
      gemmBitSerial(caffenet_gemms[i]);
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
  double nscount = chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / (double)reps;
  double perf = opcount / (nscount); // billion bit operations per second
  cout << "Time for a single " << bench_name << ": " << nscount << " nanoseconds" << endl;
  cout << "Performance for " << bench_name << ": " << perf << " GOPS per second" << endl;

  for (std::size_t i = 0; i < num_caffenet_gemms; i++) {
    deallocGEMMContext(caffenet_gemms[i]);
  }
}

int main(int argc, char const *argv[]) {
  benchmark_gemm_interactive();
  benchmark_import_interactive();
  benchmark_caffenet(20);

  vector<size_t> dims {256, 512, 1024, 2048, 4096, 8192, 16384};
  for(auto &d: dims) {
    benchmark_unrolledpopcount(d, 5);
  }

  return 0;
}
