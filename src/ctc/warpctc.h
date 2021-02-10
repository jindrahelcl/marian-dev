#pragma once

#include "tensors/tensor.h"
#include "graph/expression_graph.h"

#include <ctc.h>

namespace marian {

#define WARP_CALL(x)                  \
 do {                                 \
    if((x) != CTC_STATUS_SUCCESS) {   \
      printf("Error (%s) at %s:%d\n", \
             ctcGetStatusString(x),   \
             __FILE__,                \
             __LINE__);               \
    }                                 \
  } while(0)

class WarpCTCWrapper {
private:
  const int blankLabelIdx_;

public:
  WarpCTCWrapper(int blankLabel) : blankLabelIdx_(blankLabel) { }

  void compute(Tensor loss,
               Tensor grads,
               Tensor logits,
               Tensor flatLabels,
               Tensor labelLengths,
               const Ptr<ExpressionGraph> graph);
};
} // namespace marian