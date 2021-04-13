#pragma once

#include "marian.h"

#ifdef CUDNN
#include "ctc/cudnn_ctc.h"
#else
#include "ctc/warpctc.h"
#endif

namespace marian {

Expr ctc_loss(Expr logits, Expr flatLabels, Expr labelLengths, Expr inputLengths);

class CTCNodeOp : public NaryNodeOp {
private:
  Expr grads_;

  #ifdef CUDNN
  CUDNNCTCWrapper ctc_;
  #else
  WarpCTCWrapper ctc_; // TODO this sets blank label ID
  #endif

public:
  CTCNodeOp(Expr logits, Expr flatLabels, Expr labelLengths, Expr inputLengths);

  Shape newShape(Expr a);

  NodeOps forwardOps() override;
  NodeOps backwardOps() override;

  const std::string type() override { return "ctc"; }
};

} // namespace marian
