#pragma once

#include "tensors/tensor.h"
#include "graph/expression_graph.h"
#include "ctc/ctc_wrapper.h"

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

class WarpCTCWrapper : public CTCWrapper {

public:
  virtual void compute(Tensor loss,
		       Tensor grads,
		       Tensor logits,
		       Tensor flatLabels,
		       Tensor labelLengths,
		       Tensor inputLengths,
		       const Ptr<ExpressionGraph> graph) override;
};
} // namespace marian
