#include "graph/expression_operators.h"
#include "ctc/nodeops.h"


namespace marian {

Expr ctc_loss(Expr logits, Expr flatLabels, Expr labelLengths, Expr inputLengths) {

  #ifdef CUDNN
  // TODO VERY UGLY HACK

  // cudnn implementation needs blank token on index 0. swapping here, cost
  // does not care, and grads get backpropagated and swapped back, hopefully.
  int vocab_size = logits->shape()[-1];
  std::vector<IndexType> range(vocab_size);
  std::iota(range.begin(), range.end(), 0);

  // this is fixed for blank token on index 3
  range[0] = 3;
  range[3] = 0;

  // this assumes 0 (EOS) is not in flat labels. if it is, things will get
  // messy. Let's try.

  return Expression<CTCNodeOp>(index_select(logits, -1, range), flatLabels, labelLengths, inputLengths);
  #else

  return Expression<CTCNodeOp>(logits, flatLabels, labelLengths, inputLengths);
  #endif

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
