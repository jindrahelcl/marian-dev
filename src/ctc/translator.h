#pragma once

#include <string>
#include <vector>

#include "marian.h"
#include "models/model_task.h"
#include "ctc/nat_scorer.h"


namespace marian {

class NARTranslate : public ModelTask {
private:
  Ptr<Options> options_;

  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<Ptr<NATScorer>> scorers_; // per-device scorers (no ensembles)

  Ptr<data::Corpus> corpus_;
  Ptr<Vocab> trgVocab_;
  // Ptr<const data::ShortlistGenerator> shortlistGenerator_;

  size_t numDevices_;


public:
  NARTranslate(Ptr<Options> options);

  void run() override;

};


}
