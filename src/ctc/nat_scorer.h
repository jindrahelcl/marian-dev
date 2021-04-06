#pragma once

#include <string>
#include <vector>

#include "marian.h"
#include "data/shortlist.h"
#include "ctc/model.h"


namespace marian {

class NATScorer {
protected:
  std::string name_;
  float weight_;

  Ptr<EncoderCTCDecoder> model_;

  std::string fname_;

public:
  NATScorer(Ptr<models::IModel> model, const std::string& name, float weight, const std::string& fname)
    : name_(name),
      weight_(weight),
      model_(std::static_pointer_cast<EncoderCTCDecoder>(model)),
      fname_(fname) {}

  std::string getName() { return name_; }
  float getWeight() { return weight_; }

  void clear(Ptr<ExpressionGraph> graph);

  Logits score(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch);

  void init(Ptr<ExpressionGraph> graph);

  void setShortlistGenerator(Ptr<const data::ShortlistGenerator> shortlistGenerator);
  Ptr<data::Shortlist> getShortlist();

};


Ptr<NATScorer> natScorerByType(const std::string& fname,
			       float weight,
			       const std::string& model,
			       Ptr<Options> options);

Ptr<NATScorer> createNATScorer(Ptr<Options> options);


} // namespace marian
