#include "tensors/gpu/cuda_helpers.h"
#include "ctc/cudnn_ctc.h"

namespace marian {

#ifdef CUDNN
#include <cudnn.h>

#define CUDA_CALL(x)                  \
  do {                                \
    if((x) != cudaSuccess) {          \
      printf("Error (%s) at %s:%d\n", \
             cudaGetErrorString(x),   \
             __FILE__,                \
             __LINE__);               \
    }                                 \
  } while(0)


#define CUDNN_CALL(x)                 \
  do {                                \
    if((x) != CUDNN_STATUS_SUCCESS) { \
      printf("Error (%s) at %s:%d\n", \
             cudnnGetErrorString(x),  \
             __FILE__,                \
             __LINE__);               \
    }                                 \
  } while(0)

CUDNNCTCWrapper::CUDNNCTCWrapper(int blankTokenIndex) : CTCWrapper(blankTokenIndex) {
  CUDNN_CALL(cudnnCreate(&cudnnHandle_));
  setCTCLossDescriptor();
}

void CUDNNCTCWrapper::setCTCLossDescriptor() {
  CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctcDesc_));
  CUDNN_CALL(cudnnSetCTCLossDescriptor_v8(ctcDesc_,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_LOSS_NORMALIZATION_SOFTMAX,
                                          CUDNN_NOT_PROPAGATE_NAN,
                                          256));
}

void CUDNNCTCWrapper::compute(Tensor loss,
			      Tensor grads,
			      Tensor logits,
			      Tensor flatLabels,
			      Tensor labelLengths,
			      Tensor inputLengths,
			      const Ptr<ExpressionGraph> graph) {
  CUDA_CHECK(cudaSetDevice(loss->getDeviceId().no));

  Shape logitsShape = logits->shape();

  // in the first dimension is 1.
  ABORT_IF(logitsShape[0] != 1, "First dimension must be 1");
  int time = logitsShape[1];
  int batch = logitsShape[2];
  int vocab = logitsShape[3];

  // BLANK TOKEN NEEDS TO BE ON INDEX ZERO FOR CUDNN
  // need to swap values in the according columns, then move back
  float* logitsData = logits->data();

  // for every i in TIME, for every j in BATCH,

  // switch logitsData[:, :, :, blankTokenIndex_] with logitsData[:, :, :, 0].

  // axis = -1
  Tensor blankLogits;
  Tensor indices;
  Select(blankLogits, logits, indices, -1);



  const int dims[] = {time, batch, vocab};
  const int strides[] = {batch * vocab, vocab, 1};

  cudnnTensorDescriptor_t logitsDesc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&logitsDesc));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(logitsDesc,
                                        CUDNN_DATA_FLOAT,
                                        3,
                                        dims,
                                        strides));

  cudnnTensorDescriptor_t gradsDesc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&gradsDesc));
  CUDNN_CALL(cudnnSetTensorNdDescriptor(gradsDesc,
                                        CUDNN_DATA_FLOAT,
                                        3,
                                        dims,
                                        strides));

  // TODO here, supply flat labels in CPU memory for CuDNN 7,
  // or flat labels in GPU memory for CuDNN 8.

  size_t gpuWorkspaceSize;
  CUDNN_CALL(cudnnGetCTCLossWorkspaceSize_v8(cudnnHandle_,
                                             CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC,
                                             ctcDesc_,
                                             logitsDesc,
                                             gradsDesc,
                                             &gpuWorkspaceSize));

  //void* gpuWorkspace;
  MemoryPiece::PtrType gpuWorkspace = graph->allocator()->alloc(gpuWorkspaceSize);
  //CUDA_CHECK(cudaMalloc(&gpuWorkspace, gpuWorkspaceSize));

  int* labels = flatLabels->data<int>();
  int* labelLens = labelLengths->data<int>();
  int *inputLens = inputLengths->data<int>();
  void *costs = loss->data();
  void *gradsdata = grads->data();

  cudnnStatus_t status = cudnnCTCLoss_v8(cudnnHandle_,
                                         CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC,
                                         ctcDesc_,
                                         logitsDesc,
                                         logitsData,
                                         labels,
                                         labelLens,
                                         inputLens,
                                         costs,
                                         gradsDesc,
                                         gradsdata,
                                         gpuWorkspaceSize,
                                         gpuWorkspace->data<void>());

  switch(status) {
    case CUDNN_STATUS_SUCCESS:
      break;
    case CUDNN_STATUS_BAD_PARAM:
      if (time > 256) {  // TODO not time, but label lengths!
          printf("Too many labels (%d). Limit is 256.", time);
      }
      break;
    case CUDNN_STATUS_NOT_SUPPORTED:
    case CUDNN_STATUS_EXECUTION_FAILED:
      printf("Error in CTC loss computation: %s", cudnnGetErrorString(status));
      break;
  }

  //cudaFree(gpuWorkspace);
  graph->allocator()->free(gpuWorkspace);

  CUDNN_CALL(cudnnDestroyTensorDescriptor(logitsDesc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(gradsDesc));
}

CUDNNCTCWrapper::~CUDNNCTCWrapper() {
  CUDNN_CALL(cudnnDestroyCTCLossDescriptor(ctcDesc_));
  CUDNN_CALL(cudnnDestroy(cudnnHandle_));
}

#else  // CUDNN

CUDNNCTCWrapper::CUDNNCTCWrapper(int blankTokenIndex) {
  ABORT(
    "To use CUDNN CTC, recompile with CUDNN (cmake flag "
    "-DUSE_CUDNN=on)");
}

CUDNNCTCWrapper::~CUDNNCTCWrapper() {}

void CUDNNCTCWrapper::setCTCLossDescriptor() {
  ABORT(
    "To use CUDNN CTC, recompile with CUDNN (cmake flag "
    "-DUSE_CUDNN=on)");
}

void CUDNNCTCWrapper::compute(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
			      const Ptr<ExpressionGraph>) {
  ABORT(
    "To use CUDNN CTC, recompile with CUDNN (cmake flag "
    "-DUSE_CUDNN=on)");
}

#endif

} // namespace marian
