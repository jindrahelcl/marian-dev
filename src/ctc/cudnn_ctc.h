#pragma once

#include <iostream>

#include "common/shape.h"
#include "tensors/tensor.h"
#include "graph/expression_graph.h"
#include "tensors/gpu/cudnn_wrappers.h"

#ifdef CUDNN
#include <cudnn.h>

namespace marian {

class CTCWrapper : public CUDNNWrapper {
public:
  CTCWrapper();

  void compute(Tensor loss,
               Tensor grads,
               Tensor logits,
               Tensor flatLabels,
               Tensor labelLengths,
               const Ptr<ExpressionGraph> graph);

  virtual ~CTCWrapper();

protected:
  void setCTCLossDescriptor();

  cudnnCTCLossDescriptor_t ctcDesc_;
};

}  // namespace marian

#else

class CTCWrapper : public CUDNNWrapper {
public:
  CTCWrapper();

  void compute(Tensor loss,
               Tensor grads,
               Tensor logits,
               Tensor flatLabels,
               Tensor labelLengths,
               const Ptr<ExpressionGraph> graph);

  virtual ~CTCWrapper();

protected:
  void setCTCLossDescriptor();
};

}  // namespace marian

#endif
