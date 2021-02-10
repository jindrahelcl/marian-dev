#pragma once

#include "marian.h"
#include "ctc/warpctc.h"

namespace marian {

Expr ctc_loss(Expr logits, Expr flatLabels, Expr labelLengths);

class CTCNodeOp : public NaryNodeOp {
private:
  Expr grads_;
  //CTCWrapper ctc_;
  WarpCTCWrapper ctc_; // TODO this sets blank label ID

public:
  CTCNodeOp(Expr logits, Expr flatLabels, Expr labelLengths);

  Shape newShape(Expr a);

  NodeOps forwardOps() override;
  NodeOps backwardOps() override;

  const std::string type() override { return "ctc"; }
};

} // namespace marian