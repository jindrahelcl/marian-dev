#pragma once

#include "marian.h"
#include "models/transformer.h"

namespace marian {

class CTCDecoder : public Transformer<EncoderDecoderLayerBase> {
  typedef Transformer<EncoderDecoderLayerBase> Base;
  using Base::Base;

protected:
  using Base::options_;
  using Base::inference_;
  using Base::batchIndex_;
  using Base::graph_;
  // const std::string prefix_{"ctc-decoder"};
  // const bool inference_{false};
  // const size_t batchIndex_{0};
  const int splitFactor_{3};
  //Ptr<ExpressionGraph> graph_;

public:
  CTCDecoder(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : Base(graph, options, "ctc-decoder", /*batchIndex=*/1,
             options->get<float>("dropout-trg", 0.0f),
             options->get<bool>("embedding-fix-trg", false)),
        splitFactor_(options->get<size_t>("ctc-split-factor", 3)) {}

  Expr apply(Ptr<ExpressionGraph> graph,
             Ptr<data::CorpusBatch> batch,
             const std::vector<Ptr<EncoderState>>& encoderStates);

  virtual ~CTCDecoder() {}

protected:
  Expr expandMask(Expr mask) {
    // input shape: (beam, batch, time, 1)

    int dimModel = mask->shape()[-1];
    int dimSteps = mask->shape()[-2];
    int dimBatch = mask->shape()[-3];
    int dimBeam  = mask->shape()[-4];

    ABORT_IF(dimModel != 1, "model dim in mask must be 1");

    auto rep = repeat(mask, splitFactor_, -1);
    // shape (beam, batch, time, splitFactor)

    return reshape(rep, {dimBeam, dimBatch, dimSteps * splitFactor_, 1});
  }

  Expr splitStates(Expr input) {
    // input shape: (beam, batch, time, dim)

    // do linear projection from dim to 3xdim, split into 3 time steps
    int dimModel = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimBatch = input->shape()[-3];
    int dimBeam  = input->shape()[-4];

    int dimSplit = dimModel * splitFactor_;

    auto projected = denseInline(
      input, "ctc-statesplit", "1", dimSplit);
    // (beam, batch, time, dim * splitFactor_)

    auto reshaped = reshape(
      projected, {dimBeam, dimBatch, dimSteps * splitFactor_, dimModel});
    // (beam, batch, time * splitFactor_, dim)

    return reshaped;
  }
};

} // namespace marian