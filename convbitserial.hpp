#ifndef CONVBITSERIAL_H
#define CONVBITSERIAL_H
#include "gemmbitserial.hpp"

namespace gemmbitserial {

// im2row on interleaved channels inspired from DarkNet
template <typename Dtype>
inline Dtype im2row_get_pixel(const Dtype *im, const int height, const int width, const int channels,
                        const int row, const int col, const int channel, const int pad)
{
    const int prow = row - pad;
    const int pcol = col - pad;

    if (prow < 0 || pcol < 0 ||
        prow >= height || pcol >= width) return 0;
    // indexing according to [height][width][channel]
    return im[channel + channels*(pcol + width*prow)];
}
template <typename Dtype, typename DtypeOut>
inline void im2row(const Dtype* data_im,
     const int channels, const int height, const int width,
     const int ksize, const int stride, const int pad, DtypeOut* data_col)
{
  int c,h,w;
  const int height_col = (height + 2*pad - ksize) / stride + 1;
  const int width_col = (width + 2*pad - ksize) / stride + 1;
  const int k2 = ksize * ksize;

  int channels_col = channels * ksize * ksize;
  for (h = 0; h < height_col; ++h) {
    for (w = 0; w < width_col; ++w) {
      for(int ky = 0; ky < ksize; ky++) {
        const int im_row = ky + h*stride /*src row*/;
        for(int kx = 0; kx < ksize; kx++) {
          const int im_col = kx + w*stride /*src col*/;
          for(int c = 0; c < channels; c++) {
            const int im_chan = c /*src chan*/;
            *data_col = (DtypeOut) im2row_get_pixel(
              data_im, height, width, channels,
              im_row, im_col, im_chan, pad
            );
            data_col++;
          }
        }
      }
    }
  }
}

class ConvBitSerialContext {
public:
  uint64_t ifm;         // channels in input
  uint64_t ofm;         // channels in output
  uint64_t in_dim;      // input dimension (assumed to be square)
  uint64_t k;           // convolution kernel size
  uint64_t stride;      // convolution kernel stride
  uint64_t pad;         // padded pixels on each edge

  GEMMContext gemmctx;    // GEMM context
  BitSerialMatrix abuf;   // buffer for converted activations (prior to lowering)
  uint64_t aligned_ifm;   // input channels aligned to packing size
  uint64_t packed_ifm;    // input channels after packing
  uint64_t padded_idim;   // padded input dimension (independent from alignment)
  uint64_t out_dim;       // output dimension (assumed to be square)

  void printSummary() {
    std::cout << "========================" << std::endl;
    std::cout << "ConvBitSerialContext" << std::endl;
    std::cout << "Input channels x dim: " << ifm << " x " << in_dim << std::endl;
    std::cout << "Input channels aligned to packing: " << aligned_ifm << std::endl;
    std::cout << "Output channels x dim: " << ofm << " x " << out_dim << std::endl;
    std::cout << "ksize, stride, pad: " << k << ", " << stride << ", " << pad << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Activation buffer summary: ";
    abuf.printSummary();
    std::cout << "========================" << std::endl;
    std::cout << "Lowered activation matrix (LHS) summary: ";
    gemmctx.lhs.printSummary();
    std::cout << "========================" << std::endl;
    std::cout << "Weight matrix (RHS) summary: ";
    gemmctx.rhs.printSummary();
    std::cout << "========================" << std::endl;
  }

  template <typename T>
  void importWeights(T * buf) {
    // temporarily allocate a new matrix to perform channel-padded input
    // this is the same trick we use while importing the activations
    BitSerialMatrix bsm = BitSerialMatrix::alloc(
      gemmctx.rhs.nbits, gemmctx.rhs.nrows * k * k, ifm, gemmctx.rhs.issigned, 1, sizeof(uint64_t)*8
    );
    bsm.importRegular(buf);
    gemmctx.rhs.copyFrom_IgnoreSpatialMismatch(bsm);
    BitSerialMatrix::dealloc(bsm);
  }

  template <typename T>
  void importActivations(T * buf) {
    // import into the activation buffer. the rows/cols here are set up s.t.
    // a regular import is able to handle the channel padding.
    abuf.importRegular(buf);
    // call the sliding window (im2row) operator to generate lhs matrix
    for(int b = 0; b < gemmctx.lhs.nbits; b++) {
      im2row(
        abuf.bitplaneptr(b), packed_ifm, in_dim, in_dim, k, stride, pad, gemmctx.lhs.bitplaneptr(b)
      );
    }
  }
};

inline ConvBitSerialContext allocConvBitSerialContext(
  const uint64_t ifm,         // channels in input
  const uint64_t ofm,         // channels in output
  const uint64_t in_dim,      // input dimension (assumed to be square)
  const uint64_t k,           // convolution kernel size
  const uint64_t stride,      // convolution kernel stride
  const uint64_t pad,         // padded pixels on each edge
  const uint64_t ibits,       // bits per input
  const uint64_t wbits,       // bits per weight
  const bool isigned,         // whether inputs are signed
  const bool wsigned          // whether weights are signed
) {
  // there's currently a bug with ofm=1 configs, assert until resolved
  assert(ofm != 1);
  ConvBitSerialContext ctx;
  ctx.ifm = ifm;
  ctx.ofm = ofm;
  ctx.in_dim = in_dim;
  ctx.k = k;
  ctx.stride = stride;
  ctx.pad = pad;
  const uint64_t pack_bits = sizeof(uint64_t) * 8;
  // determine the output dimension based on input
  ctx.padded_idim = ctx.in_dim + 2*ctx.pad;
  ctx.out_dim = ((ctx.padded_idim - ctx.k) / ctx.stride) + 1;
  // round up number of input channels to be divisible by packing size
  ctx.aligned_ifm = alignTo(ctx.ifm, pack_bits);
  // number of channels after packing
  ctx.packed_ifm = ctx.aligned_ifm / pack_bits;
  // determine the matrix sizes for the lowered convolution
  // lhs is inputs, rhs is weights
  const uint64_t lhs_rows = ctx.out_dim * ctx.out_dim;
  const uint64_t depth = ctx.aligned_ifm * ctx.k * ctx.k;
  const uint64_t rhs_rows = ofm;
  // allocate the context
  ctx.gemmctx = allocGEMMContext(
    lhs_rows, depth, rhs_rows, ibits, wbits, isigned, wsigned
  );
  // allocate the buffer for converted activations
  ctx.abuf = BitSerialMatrix::alloc(
    ibits, ctx.in_dim * ctx.in_dim, ctx.ifm, isigned, 1, pack_bits
  );
  // bipolar-bipolar convs not yet supported due to padding issues
  assert(!(ctx.gemmctx.lhs.isBipolar() && ctx.gemmctx.rhs.isBipolar()));
  return ctx;
}

inline void deallocConvBitSerialContext(ConvBitSerialContext ctx) {
  deallocGEMMContext(ctx.gemmctx);
  BitSerialMatrix::dealloc(ctx.abuf);
}
}
#endif /* end of include guard: CONVBITSERIAL_H */
