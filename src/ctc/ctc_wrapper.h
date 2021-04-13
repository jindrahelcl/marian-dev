#pragma once

#include "tensors/tensor.h"
#include "graph/expression_graph.h"
#include "ctc/ctc_wrapper.h"

#include <ctc.h>

namespace marian {

class CTCWrapper {
protected:
  const int blankLabelIdx_;

public:
  CTCWrapper(int blankLabel) : blankLabelIdx_(blankLabel) { }

  virtual void compute(Tensor loss,
		       Tensor grads,
		       Tensor logits,
		       Tensor flatLabels,
		       Tensor labelLengths,
		       Tensor inputLengths,
		       const Ptr<ExpressionGraph> graph) = 0;
};

} // namespace marian
