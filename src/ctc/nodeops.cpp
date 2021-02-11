#include "ctc/nodeops.h"

namespace marian {

Expr ctc_loss(Expr logits, Expr flatLabels, Expr labelLengths, Expr inputLengths) {
  return Expression<CTCNodeOp>(logits, flatLabels, labelLengths, inputLengths);
}

Shape CTCNodeOp::newShape(Expr a) {
  Shape shape1 = a->shape();
  shape1.set(a->shape().size() - 1, 1);  // will reduce the logit dimension
  shape1.set(a->shape().size() - 3, 1);  // will reduce the time dimension
  return shape1;
}

CTCNodeOp::CTCNodeOp(Expr logits, Expr flatLabels, Expr labelLengths, Expr inputLengths)
  : NaryNodeOp({logits, flatLabels, labelLengths, inputLengths}, newShape(logits), logits->value_type()),
    grads_(graph()->zeros(logits->shape())),
    ctc_(3) {

    setMemoize(false);
  //  {
  //matchOrAbort<IndexType>(flatLabels->value_type());

  //int rows   = logits->shape().elements() / logits->shape()[-1];
  //int labels = labelLengths->shape().elements();

  //ABORT_IF(rows != labels, "Number of examples and labels does not match: {} != {}", rows, labels);

  // output shape of this op should be 1, batch, 1.
  // input lengths correspond to the time dimension of logits
}

NodeOps CTCNodeOp::forwardOps() {
  return {NodeOp(
      ctc_.compute(
        /* loss= */         val_,
        /* gradients= */    grads_->val(),
        /* logits= */       child(0)->val(),
        /* flatLabels= */   child(1)->val(),
        /* labelLengths= */ child(2)->val(),
        /* inputLengths= */ child(3)->val(),
        /* graph= */        graph());
      )};
}

NodeOps CTCNodeOp::backwardOps() {
  using namespace functional;
  // TODO Dont know whether to use adj_ or not.
  return {NodeOp(
    // child(0)->grad is initialized with zeros, so it should
    // suffice to just add the grads->val() to it.

    Add(_1 * _2, child(0)->grad(), grads_->val(), adj_)
  )};
}

} // namespace marian