#include <cassert>
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <vector>
#include <deque>
#include "gemmbitserial.hpp"

using namespace std;
using namespace gemmbitserial;

#define VERBOSE_TEST(x) ;
//#define VERBOSE_TEST(x) x

/**
* Generate a random vector with given dimension and number of bits <= 8
*/
void generateRandomVector(size_t bits, size_t dim, uint8_t * ret) {
  uint8_t minVal = 0;
  uint8_t maxVal = (1 << bits);
  for(size_t i = 0; i < dim; i++) {
    ret[i] = rand() % maxVal;
  }
}

void naive_int_gemm(uint8_t * lhs, uint8_t * rhs, int32_t * res, int rows, int depth, int cols) {
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
void printmatrixdiff(T * mat1, T* mat2, int rows, int cols) {
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

bool test_conversions() {
  vector<size_t> param_bits {1, 2, 3, 4};
  vector<size_t> param_dims {16, 17, 32, 77, 100, 1024, 4096};
  unsigned int numConfigs = 0, ok = 0, nok = 0;

  for(auto & b: param_bits) {
    for(auto & d: param_dims) {
      uint8_t * res_chk = new uint8_t[d*d];
      uint8_t * rnd_vec = new uint8_t[d*d];
      assert(res_chk != 0 && rnd_vec != 0);
      generateRandomVector(b, d*d, rnd_vec);

      BitSerialMatrix bsm = BitSerialMatrix::alloc(b, d, d, false);
      bsm.importRegular(rnd_vec);
      bsm.exportRegular(res_chk);
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
  cout << "Conversion tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}


bool test_matrix_matrix() {
  vector<size_t> param_bits {1, 2, 3, 4};
  vector<size_t> param_dims {16, 17, 18, 30, 31, 32, 100, 177, 256};

  deque<bool> param_allow_neg {false, true};
  unsigned int numConfigs = 0, ok = 0, nok = 0;
  for(auto & b: param_bits) {
    for(auto & d: param_dims) {
      uint8_t * rnd_mat_a = new uint8_t[d*d*2];
      uint8_t * rnd_mat_b = new uint8_t[2*d*d*3];
      int32_t * res_mat_golden = new int32_t[d*d*3];
      generateRandomVector(b, d*d*2, rnd_mat_a);
      generateRandomVector(b, 2*d*d*3, rnd_mat_b);
      naive_int_gemm(rnd_mat_a, rnd_mat_b, res_mat_golden, d, 2*d, d*3);
      GEMMContext ctx = allocGEMMContext(d, 2*d, 3*d, b, b, false, false);
      ctx.lhs.importRegular(rnd_mat_a);
      ctx.rhs.importRegular(rnd_mat_b);

      gemmBitSerial(ctx);
      //printmatrix(rnd_mat_a, d, d*2);
      //printmatrix(rnd_mat_b, d*3, d*2);
      //printmatrix(res_mat_golden, d*3, d);
      //printmatrix(ctx.res, d*3, d);

      int rbytes = d*d*3*sizeof(int32_t);
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
  cout << "Matrix matrix multiplication tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

int main(int argc, char const *argv[]) {
  srand(time(NULL));
  bool all_ok = true;
  //all_ok &= test_conversions();
  all_ok &= test_matrix_matrix();

  if(all_ok) {
    cout << "All tests completed successfully" << endl;
  } else {
    cout << "Some tests failed" << endl;
  }
  return 0;
}
