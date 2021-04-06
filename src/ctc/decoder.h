#pragma once

#include "marian.h"
#include "data/shortlist.h"
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

  const int splitFactor_{3};
  Ptr<data::Shortlist> shortlist_;

public:
  CTCDecoder(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : Base(graph, options, "ctc-decoder", /*batchIndex=*/1,
             options->get<float>("dropout-trg", 0.0f),
             options->get<bool>("embedding-fix-trg", false)),
        splitFactor_(options->get<size_t>("ctc-split-factor", 3)) {}

  Logits apply(Ptr<ExpressionGraph> graph,
	       Ptr<data::CorpusBatch> batch,
	       const std::vector<Ptr<EncoderState>>& encoderStates);

  void clear();

  Ptr<data::Shortlist> getShortlist() { return shortlist_; }

  void setShortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
  }

  virtual ~CTCDecoder() {}

protected:
  Expr expandMask(Expr mask);

  Expr splitStates(Expr input);

private:
  Ptr<mlp::Output> output_;

  void lazyCreateOutputLayer();
};

} // namespace marian
