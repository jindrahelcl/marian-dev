#pragma once

#include "marian.h"
#include "data/types.h"

#include "ctc/nat_scorer.h"


namespace marian {

class CTCSearchResult {
private:
  size_t lineNo_;
  Words words_;
  float pathScore_;

public:
  CTCSearchResult(size_t lineNo, const Words& words, float pathScore)
    : lineNo_(lineNo),
      words_(words),
      pathScore_(pathScore) {};

  // // returns n best hypotheses from the beam
  // NBestList nBest(size_t n, bool skipEmpty = false) const;

  // these are copied from marian::History
  size_t getLineNum() const { return lineNo_ ; };

  const Words& getWords() const { return words_; };

  float getPathScore() const { return pathScore_; };

  // Result top() const {
  //   const NBestList& nbest = nBest(1);
  //   ABORT_IF(nbest.empty(), "No hypotheses in n-best list");
  //   return nBest[0];
  // }

};

typedef std::vector<Ptr<CTCSearchResult>> CTCResults;

class CTCBeamSearch {

  Ptr<Options> options_;
  Ptr<NATScorer> scorer_;
  size_t beamSize_;
  Ptr<Vocab> trgVocab_;

public:
  CTCBeamSearch(Ptr<Options> options, Ptr<NATScorer> scorer, Ptr<Vocab> trgVocab)
    : options_(options), scorer_(scorer), beamSize_(options_->get<size_t>("beam-size")), trgVocab_(trgVocab)
  {}

  // main decoding function
  CTCResults search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch);
};

} // namespace marian