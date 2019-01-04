#include <cassert>
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <vector>
#include <deque>
#include "gemmbitserial.hpp"
#include "convbitserial.hpp"
#include "mnistdata.h"
#include "testhelpers.hpp"

using namespace std;
using namespace gemmbitserial;

#define VERBOSE_TEST(x) ;
//#define VERBOSE_TEST(x) x

void printBitSerialMatrix(BitSerialMatrix * bsm) {
  std::cout << "BitSerialMatrix with bits " << bsm->nbits << " rows " << bsm->nrows << " cols " << bsm->ncols << std::endl;
  for(int b = 0; b < bsm->nbits; b++) {
    std::cout << "bit " << b << ":" << std::endl;
    for(int r = 0; r < bsm->nrows; r++) {
      for(int c = 0; c < bsm->ncols; c++) {
        std::cout << (bsm->get(b,r,c) ? 1 : 0) << " ";
      }
      std::cout << std::endl << std::endl;
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

bool test_conv_lowering() {
  // some notes on this test:
  // * IFM must be multiple of 64, since we don't support chan padding in direct lowering
  // * the datatype must support zero (i.e. no bipolar)
  // * OFM = 1 has a known bug so is not included
  vector<int> ibits_sweep {1, 2, 3};
  vector<int> ifm_sweep {64, 128};
  vector<int> ofm_sweep {2, 4};
  vector<int> idim_sweep {15, 16};
  vector<int> k_sweep {2, 3};
  vector<int> stride_sweep {1, 2};
  vector<int> pad_sweep {0, 1};
  unsigned int numConfigs = 0, ok = 0, nok = 0;

  for(auto & ibits: ibits_sweep) {
  for(auto & idim: idim_sweep) {
  for(auto & k: k_sweep) {
  for(auto & stride: stride_sweep) {
  for(auto & pad: pad_sweep) {
  for(auto & ifm: ifm_sweep) {
  for(auto & ofm: ofm_sweep) {
    // calculate some sizes
    int depth = k * k * ifm;
    int odim = (((idim + 2*pad) - k) / stride) + 1;
    // allocate input image and weight
    //uint8_t * w = new uint8_t[ofm * depth];
    uint8_t * a = new uint8_t[ifm * idim * idim];
    uint8_t * a_lowered = new uint8_t[odim * odim * depth];
    //int32_t * res_golden = new int32_t[odim * odim * ofm];
    // random initialization
    //generateRandomVector(wbits, ofm*depth, w, wsigned);
    generateRandomVector(ibits, ifm*idim*idim, a, false);
    //memset(res_golden, 0, sizeof(int32_t)*odim*odim*ofm);
    // allocate conv context
    ConvBitSerialContext ctx = allocConvBitSerialContext(
    ifm, ofm, idim, k, stride, pad, ibits, 2, false, false
    );
    //ctx.importWeights(w);
    ctx.importActivations(a);
    // produce golden and compare
    im2row(a, ifm, idim, idim, k, stride, pad, a_lowered);
    uint8_t * a_lowered_produced = new uint8_t[odim * odim * depth];
    ctx.gemmctx.lhs.exportRegular(a_lowered_produced);
    //ctx.printSummary();
    int res = memcmp(a_lowered, a_lowered_produced, sizeof(uint8_t)*odim * odim * depth);
    if(res == 0) {
    ok++;
    } else {
    nok++;
    //VERBOSE_TEST(printmatrix(a_lowered, odim*odim, depth));
    //VERBOSE_TEST(printmatrix(a_lowered_produced, odim*odim, depth));
    }
    numConfigs++;
    VERBOSE_TEST(cout << "ibits " << ibits << " ifm " << ifm << " ofm " << ofm << " idim " << idim << " k " << k << " result = " << res << endl);
    // cleanup
    deallocConvBitSerialContext(ctx);
    //delete [] res_golden;
    delete [] a_lowered;
    delete [] a_lowered_produced;
    //delete [] w;
    delete [] a;
  }
  }
  }
  }
  }
  }
  }
  cout << "Convolution lowering tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

bool test_conv() {
  // TODO add bipolar times bipolar conv test cases
  vector<int> ibits_sweep {2, 3};
  vector<int> wbits_sweep {1, 2, 3};
  vector<int> ifm_sweep {1, 2, 3};
  vector<int> ofm_sweep {2, 4};
  deque<bool> isigned_sweep {true, false};
  deque<bool> wsigned_sweep {true, false};
  vector<int> idim_sweep {15, 16};
  vector<int> k_sweep {2, 3};
  vector<int> stride_sweep {1, 2};
  vector<int> pad_sweep {0, 1};
  unsigned int numConfigs = 0, ok = 0, nok = 0;

  for(auto & ibits: ibits_sweep) {
    for(auto & wbits: wbits_sweep) {
      for(auto & idim: idim_sweep) {
        for(auto & k: k_sweep) {
          for(auto & stride: stride_sweep) {
            for(auto & pad: pad_sweep) {
              for(auto & isigned: isigned_sweep) {
                for(auto & wsigned: wsigned_sweep) {
                  for(auto & ifm: ifm_sweep) {
                    for(auto & ofm: ofm_sweep) {
                      // calculate some sizes
                      int depth = k * k * ifm;
                      int odim = (((idim + 2*pad) - k) / stride) + 1;
                      // allocate input image and weight
                      int8_t * w = new int8_t[ofm * depth];
                      int8_t * a = new int8_t[ifm * idim * idim];
                      int8_t * a_lowered = new int8_t[odim * odim * depth];
                      int32_t * res_golden = new int32_t[odim * odim * ofm];
                      // random initialization
                      generateRandomVector(wbits, ofm*depth, w, wsigned);
                      generateRandomVector(ibits, ifm*idim*idim, a, isigned);
                      memset(res_golden, 0, sizeof(int32_t)*odim*odim*ofm);
                      // allocate conv context
                      ConvBitSerialContext ctx = allocConvBitSerialContext(
                        ifm, ofm, idim, k, stride, pad, ibits, wbits, isigned, wsigned
                      );
                      ctx.importWeights(w);
                      ctx.importActivations(a);
                      gemmBitSerial(ctx.gemmctx);
                      // produce golden and compare
                      im2row(a, ifm, idim, idim, k, stride, pad, a_lowered);
                      naive_int_gemm(a_lowered, w, res_golden, odim*odim, depth, ofm);

                      //ctx.printSummary();
                      int res = memcmp(res_golden, ctx.gemmctx.res, sizeof(int32_t)*odim * odim * ofm);
                      if(res == 0) {
                        ok++;
                      } else {
                        nok++;
                        //VERBOSE_TEST(printmatrix(a_lowered, odim*odim, depth));
                        //VERBOSE_TEST(printmatrix(a_lowered_produced, odim*odim, depth));
                      }
                      numConfigs++;
                      VERBOSE_TEST(cout << "ifm " << ifm << " ofm " << ofm << " idim " << idim << " k " << k << " result = " << res << endl);
                      // cleanup
                      deallocConvBitSerialContext(ctx);
                      delete [] res_golden;
                      delete [] a_lowered;
                      delete [] w;
                      delete [] a;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  cout << "Convolution tests: " << ok << " OK, " << nok << " NOK" << endl;
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
  all_ok &= test_conv_lowering();
  all_ok &= test_conv();

  if(all_ok) {
    cout << "All tests completed successfully" << endl;
  } else {
    cout << "Some tests failed" << endl;
  }
  return 0;
}
