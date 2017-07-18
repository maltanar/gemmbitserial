#include <cassert>
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <vector>
#include <deque>
#include "gemmbitserial.hpp"
#include "mnistdata.h"

using namespace std;
using namespace gemmbitserial;

#define VERBOSE_TEST(x) ;
//#define VERBOSE_TEST(x) x

// Generate a random vector of -1 and +1 values of given dimension
template <typename T>
void generateRandomVector_Bipolar(size_t dim, T * ret) {
  for(size_t i = 0; i < dim; i++) {
    ret[i] = (rand() % 2 == 0) ? 1 : -1;
  }
}

/**
* Generate a random vector with given dimension and number of bits
*/
template <typename T>
void generateRandomVector(size_t bits, size_t dim, T * ret, bool allowNeg = false) {
  assert(bits <= (sizeof(T) * 8));
  if(bits == 1 && allowNeg) {
    // generate bipolar values
    generateRandomVector_Bipolar(dim, ret);
    return;
  }
  int32_t minVal = 0;
  int32_t maxVal = (1 << bits);
  for(size_t i = 0; i < dim; i++) {
    ret[i] = (rand() % maxVal) - (allowNeg ? maxVal/2 : 0);
  }
}

template <typename LHSType, typename RHSType>
void naive_int_gemm(LHSType * lhs, RHSType * rhs, int32_t * res, int rows, int depth, int cols) {
  for(int k = 0; k < cols; k++) {
    for(int i = 0; i < rows; i++) {
      int32_t acc = 0;
      for(int j = 0; j < depth; j++) {
        acc += lhs[i * depth + j] * rhs[k * depth + j];
      }
      res[k * rows + i] = acc;
    }
  }
}

template <typename T>
void naive_sum_rows(T * m, int32_t * res, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    int32_t acc = 0;
    for(int k = 0; k < cols; k++) {
      acc += m[i * cols + k];
    }
    res[i] = acc;
  }
}

template <typename T>
void printmatrix(T * mat, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      cout << (int) mat[i * cols + j] << " ";
    }
    cout << endl;
  }
  cout << endl;
}

template <typename T>
void printmatrixdiff(const T * mat1, const T * mat2, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      if(mat1[i * cols + j] != mat2[i * cols + j]) {
        cout << "Difference at (i,j) = " << i << " " << j << " Mat1: " << (int)mat1[i * cols + j] << " Mat2: " << mat2[i * cols + j] << endl;
      }
    }
  }
  cout << endl;
}

void printBitSerialMatrix(BitSerialMatrix * bsm) {
  cout << "BitSerialMatrix with bits " << bsm->nbits << " rows " << bsm->nrows << " cols " << bsm->ncols << endl;
  for(int b = 0; b < bsm->nbits; b++) {
    cout << "bit " << b << ":" << endl;
    for(int r = 0; r < bsm->nrows; r++) {
      for(int c = 0; c < bsm->ncols; c++) {
        cout << (bsm->get(b,r,c) ? 1 : 0) << " ";
      }
      cout << endl << endl;
    }
  }
}

bool test_rowwise_sum() {
  vector<size_t> param_bits {1, 2, 3, 4};
  vector<size_t> param_dims {4, 16, 17, 32, 77, 100, 1023};
  vector<int> param_signed {1, 0};
  unsigned int numConfigs = 0, ok = 0, nok = 0;
  for(auto & b: param_bits) {
    for(auto & d: param_dims) {
      for(auto & sgnd: param_signed) {
        bool isSigned = (bool) sgnd;
        int8_t * rnd_mat = new int8_t[d*d];
        int32_t * res_ret = new int32_t[d];
        int32_t * res_golden = new int32_t[d];
        generateRandomVector(b, d*d, rnd_mat, isSigned);
        // TODO add aligned version of BitSerialMatrix::alloc
        GEMMContext ctx = allocGEMMContext(
          d, d, 1, b, 1, isSigned, false
        );
        BitSerialMatrix bsm = ctx.lhs;
        bsm.importRegular(rnd_mat);
        sumRows(bsm, res_ret);
        naive_sum_rows(rnd_mat, res_golden, d, d);
        int res = memcmp(res_ret, res_golden, d);
        if(res == 0) {
          ok++;
        } else {
          nok++;
        }
        /*printmatrix(rnd_mat, d, d);
        printmatrix(res_golden, d, 1);
        printmatrix(res_ret, d, 1);*/
        deallocGEMMContext(ctx);
        delete [] rnd_mat;
        delete [] res_golden;
        delete [] res_ret;
        numConfigs++;
        VERBOSE_TEST(cout << "Bits = " << b << " dim = " << d << " result = " << res << endl);
      }
    }
  }
  cout << "Row-wise sum tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

bool test_conversions() {
  vector<size_t> param_bits {1, 2, 3, 7};
  vector<size_t> param_dims {4, 16, 17, 32, 77, 100, 1023};
  vector<int> param_signed {1, 0};
  unsigned int numConfigs = 0, ok = 0, nok = 0;

  for(auto & b: param_bits) {
    for(auto & d: param_dims) {
      for(auto & sgnd: param_signed) {
        int8_t * res_chk = new int8_t[d*d];
        int8_t * rnd_vec = new int8_t[d*d];
        assert(res_chk != 0 && rnd_vec != 0);
        generateRandomVector(b, d*d, rnd_vec, (bool) sgnd);

        BitSerialMatrix bsm = BitSerialMatrix::alloc(b, d, d, (bool) sgnd);
        bsm.importRegular(rnd_vec);
        bsm.exportRegular(res_chk);
        //printmatrix(rnd_vec, d, d);
        //printmatrix(res_chk, d, d);
        int res = memcmp(rnd_vec, res_chk, d);
        if(res == 0) {
          ok++;
        } else {
          nok++;
        }
        delete [] rnd_vec;
        delete [] res_chk;
        BitSerialMatrix::dealloc(bsm);
        numConfigs++;
        VERBOSE_TEST(cout << "Bits = " << b << " dim = " << d << " result = " << res << endl);
      }
    }
  }
  cout << "Conversion tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

bool test_matrix_matrix() {
  vector<size_t> param_bits {2, 3, 4};
  vector<size_t> param_dims {3, 5, 7, 16, 17, 18, 30, 31, 32, 100, 177, 256};
  vector<int> do_matrix_vector {0, 1};

  unsigned int numConfigs = 0, ok = 0, nok = 0;
  for(auto & b: param_bits) {
    for(auto & d: param_dims) {
      for(auto & mv: do_matrix_vector) {
        uint8_t * rnd_mat_a = new uint8_t[d*d*2];
        uint8_t * rnd_mat_b = new uint8_t[2*d*(mv ? 1 : d*3)];
        int32_t * res_mat_golden = new int32_t[d*(mv ? 1 : d*3)];
        generateRandomVector(b, d*d*2, rnd_mat_a);
        generateRandomVector(b, 2*d*(mv ? 1 : d*3), rnd_mat_b);
        naive_int_gemm(rnd_mat_a, rnd_mat_b, res_mat_golden, d, 2*d, (mv ? 1 : d*3));
        GEMMContext ctx = allocGEMMContext(d, 2*d, (mv ? 1 : d*3), b, b, false, false);
        ctx.lhs.importRegular(rnd_mat_a);
        ctx.rhs.importRegular(rnd_mat_b);

        gemmBitSerial(ctx);
        //ctx.printSummary();
        //printmatrix(rnd_mat_a, d, d*2);
        //printmatrix(rnd_mat_b, d*3, d*2);
        //printmatrix(res_mat_golden, d*3, d);
        //printmatrix(ctx.res, d*3, d);

        int rbytes = d*(mv ? 1 : d*3)*sizeof(int32_t);
        int res = memcmp(ctx.res, res_mat_golden, rbytes);
        if(res == 0) {
          ok++;
        } else {
          nok++;
          //printmatrixdiff(res_mat, res_mat_golden, 3*d, d);
        }
        delete [] rnd_mat_a;
        delete [] rnd_mat_b;
        delete [] res_mat_golden;
        deallocGEMMContext(ctx);
        numConfigs++;
        VERBOSE_TEST(cout << "Bits = " << b << " dim = " << d << " result = " << res << endl);
      }
    }
  }
  cout << "Matrix matrix multiplication tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

bool test_mnist() {
  // test bit serial gemm using real-life matrix data from a MNIST neural net
  GEMMContext ctx = allocGEMMContext(
    MNIST_OUT, MNIST_IN, 1, MNIST_WBITS, MNIST_ABITS, MNIST_WSIGN, MNIST_ASIGN
  );
  ctx.lhs.importRegular(mnist_weights);
  ctx.rhs.importRegular(mnist_in);
  gemmBitSerial(ctx);
  int res = memcmp(ctx.res, mnist_res_golden, MNIST_OUT*sizeof(int32_t));
  cout << "MNIST matrix-vector using bipolar times regular: " << (res == 0 ? "OK" : "NOK") << endl;
  if(res != 0) {
    printmatrixdiff(ctx.res, mnist_res_golden, 1, MNIST_OUT);
  }
  deallocGEMMContext(ctx);
  return res == 0;
}

bool test_bipolar_times_regular() {
  vector<size_t> param_regularmatrix_bits {2, 3, 4};
  vector<size_t> param_dims {3, 5, 7, 16, 17, 18, 30, 31, 32, 100, 177, 256};
  vector<int> param_signed {1, 0};
  vector<int> param_switch_lhsrhs {0, 1};

  unsigned int numConfigs = 0, ok = 0, nok = 0;
  // TODO when bipolar times bipolar is covered, merge into matrix matrix
  for(auto & sw_lhsrhs: param_switch_lhsrhs) {
    for(auto & rhs_bits: param_regularmatrix_bits) {
      for(auto & d: param_dims) {
        for(auto & sgnd: param_signed) {
          const size_t lhs_bits = 1;
          const bool lhs_sign = true;
          const bool rhs_sign = (bool) sgnd;
          int8_t * bipolar_mat = new int8_t[d*d];
          int8_t * regular_mat = new int8_t[d*d];
          int32_t * res_golden = new int32_t[d*d];
          int32_t * res_chk = new int32_t[d*d];
          generateRandomVector_Bipolar(d*d, bipolar_mat);
          generateRandomVector(rhs_bits, d*d, regular_mat, rhs_sign);
          GEMMContext ctx;
          if(sw_lhsrhs) {
            ctx = allocGEMMContext(
              d, d, d, rhs_bits, lhs_bits, rhs_sign, lhs_sign
            );
            ctx.rhs.importRegular(bipolar_mat);
            ctx.lhs.importRegular(regular_mat);
            naive_int_gemm(regular_mat, bipolar_mat, res_golden, d, d, d);
          } else {
            ctx = allocGEMMContext(
              d, d, d, lhs_bits, rhs_bits, lhs_sign, rhs_sign
            );
            ctx.lhs.importRegular(bipolar_mat);
            ctx.rhs.importRegular(regular_mat);
            naive_int_gemm(bipolar_mat, regular_mat, res_golden, d, d, d);
          }
          gemmBitSerial(ctx);
          //printmatrix(bipolar_mat, d, d);
          //printmatrix(regular_mat, d, d);
          //printmatrix(res_golden, d, d);
          //printmatrix(ctx.res, d, d);
          int res = memcmp(res_golden, ctx.res, sizeof(int32_t)*d*d);
          if(res == 0) {
            ok++;
          } else {
            nok++;
          }
          numConfigs++;
          delete [] bipolar_mat;
          delete [] regular_mat;
          delete [] res_golden;
          delete [] res_chk;
          deallocGEMMContext(ctx);
          VERBOSE_TEST(cout << "Bits = " << rhs_bits << " dim = " << d << " result = " << res << endl);
        }
      }
    }
  }
  cout << "Bipolar times regular tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

bool test_bipolar_times_bipolar() {
  vector<size_t> param_dims {3, 5, 7, 16, 17, 18, 30, 31, 32, 100, 177, 256};
  vector<int> do_matrix_vector {0, 1};
  unsigned int numConfigs = 0, ok = 0, nok = 0;

  for(auto & d: param_dims) {
    for(auto & mv: do_matrix_vector) {
      int8_t * lhs_mat = new int8_t[d*d];
      int8_t * rhs_mat = new int8_t[mv ? d : d*d];
      int32_t * res_golden = new int32_t[mv ? d : d*d];
      int32_t * res_chk = new int32_t[mv ? d : d*d];
      GEMMContext ctx = allocGEMMContext(
        d, d, mv ? 1 : d, 1, 1, true, true
      );
      generateRandomVector_Bipolar(d*d, lhs_mat);
      generateRandomVector_Bipolar(mv ? d : d*d, rhs_mat);
      ctx.lhs.importRegular(lhs_mat);
      ctx.rhs.importRegular(rhs_mat);
      gemmBitSerial(ctx);
      naive_int_gemm(lhs_mat, rhs_mat, res_golden, d, d, mv ? 1 : d);
      //printmatrix(lhs_mat, d, d);
      //printmatrix(rhs_mat, d, d);
      //printmatrix(res_golden, d, d);
      //printmatrix(ctx.res, d, d);
      int res = memcmp(res_golden, ctx.res, sizeof(int32_t)*d*(mv ? 1 : d));
      if(res == 0) {
        ok++;
      } else {
        nok++;
      }
      numConfigs++;
      delete [] lhs_mat;
      delete [] rhs_mat;
      delete [] res_golden;
      delete [] res_chk;
      deallocGEMMContext(ctx);
    }
  }
  cout << "Bipolar times bipolar tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

int main(int argc, char const *argv[]) {
  srand(time(NULL));
  bool all_ok = true;
  all_ok &= test_conversions();
  all_ok &= test_rowwise_sum();
  all_ok &= test_mnist();
  all_ok &= test_matrix_matrix();
  all_ok &= test_bipolar_times_regular();
  all_ok &= test_bipolar_times_bipolar();

  if(all_ok) {
    cout << "All tests completed successfully" << endl;
  } else {
    cout << "Some tests failed" << endl;
  }
  return 0;
}
