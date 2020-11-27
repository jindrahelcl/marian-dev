#pragma once

#include "marian.h"
#include "translator/history.h"
#include "translator/scorers.h"


namespace marian {

class CTCBeamSearch {

  Ptr<Options> options_;
  std::vector<Ptr<Scorer>> scorers_;
  size_t beamSize_;
  Ptr<const Vocab> trgVocab_;

public:
  CTCBeamSearch(Ptr<Options> options, const std::vector<Ptr<Scorer>>& scorers, const Ptr<const Vocab> trgVocab)
      : options_(options), scorers_(scorers), beamSize_(options_->get<size_t>("beam-size")), trgVocab_(trgVocab)
  {}

  // main decoding function
  Histories search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch);
}

} // namespace marian