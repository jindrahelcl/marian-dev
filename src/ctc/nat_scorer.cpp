#include "ctc/nat_scorer.h"
#include "common/io.h"
#include "models/model_factory.h"


namespace marian {

void NATScorer::init(Ptr<ExpressionGraph> graph) {
  graph->switchParams(getName());
  model_->load(graph, fname_);
}

void NATScorer::clear(Ptr<ExpressionGraph> graph) {
  graph->switchParams(getName());
  model_->clear(graph);
}

Logits NATScorer::score(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
  graph->switchParams(getName());
  return model_->build(graph, batch);
}

Ptr<NATScorer> natScorerByType(const std::string& fname,
			       float weight,
			       const std::string& model,
			       Ptr<Options> options) {
  options->set("inference", true);
  auto nat_model = models::createModelFromOptions(options, models::usage::raw);

  LOG(info, "Loading NAT scorer as feature {}", fname);

  return New<NATScorer>(nat_model, fname, weight, model);
}

Ptr<NATScorer> createNATScorer(Ptr<Options> options) {
  std::vector<Ptr<NATScorer>> scorers;

  auto models = options->get<std::vector<std::string>>("models");
  ABORT_IF(models.size() != 1, "no ensembles supported");

  auto model = models[0];
  float weight = 1.f;

  if(options->hasAndNotEmpty("weights"))
    weight = options->get<std::vector<float>>("weights")[0];

  std::string fname = "F0";

  // load options specific for the scorer
  auto modelOptions = New<Options>(options->clone());
  try {
    if(!options->get<bool>("ignore-model-config")) {
      YAML::Node modelYaml;
      io::getYamlFromModel(modelYaml, "special:model.yml", model);
        modelOptions->merge(modelYaml, true);
    }
  } catch(std::runtime_error&) {
    LOG(warn, "No model settings found in model file");
  }

  // TODO something like l2r and r2l can be done in CTC as well, but
  // let's not for now..

  ABORT_IF(modelOptions->get<bool>("right-left"), "Cannot do r2l, sorry");

  return natScorerByType(fname, weight, model, modelOptions);
}

void NATScorer::setShortlistGenerator(Ptr<const data::ShortlistGenerator> shortlistGenerator) {
  model_->setShortlistGenerator(shortlistGenerator);
}

Ptr<data::Shortlist> NATScorer::getShortlist() {
  return model_->getShortlist();
}

}
