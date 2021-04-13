#pragma once

#include <iostream>

#include "common/shape.h"
#include "tensors/tensor.h"
#include "graph/expression_graph.h"
#include "ctc/ctc_wrapper.h"

#ifdef CUDNN
#include <cudnn.h>
#endif

namespace marian {

class CUDNNCTCWrapper : public CTCWrapper {

public:
  CUDNNCTCWrapper(int blankTokenIndex);

  virtual void compute(Tensor loss,
		       Tensor grads,
		       Tensor logits,
		       Tensor flatLabels,
		       Tensor labelLengths,
		       Tensor inputLenghts,
		       const Ptr<ExpressionGraph> graph) override;

  virtual ~CUDNNCTCWrapper();

protected:
  void setCTCLossDescriptor();

#ifdef CUDNN
  cudnnCTCLossDescriptor_t ctcDesc_;
  cudnnHandle_t cudnnHandle_;
#endif
};

}  // namespace marian
