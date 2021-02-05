#include "ctc/model.h"

/// MOST OF THE CODE COPIED FROM ENCODER_CLASSIFIER


namespace marian {

Config::YamlNode EncoderCTCDecoder::getModelParameters() {
  Config::YamlNode modelParams;
  auto clone = options_->cloneToYamlNode();
  for(auto& key : modelFeatures_)
    modelParams[key] = clone[key];

  if(options_->has("original-type"))
    modelParams["type"] = clone["original-type"];

  modelParams["version"] = buildVersion();
  return modelParams;
}

std::string EncoderCTCDecoder::getModelParametersAsString() {
  auto yaml = getModelParameters();
  YAML::Emitter out;
  cli::OutputYaml(yaml, out);
  return std::string(out.c_str());
}

// @TODO: lots of code-duplication with EncoderDecoder
EncoderCTCDecoder::EncoderCTCDecoder(Ptr<Options> options)
    : options_(options),
      prefix_(options->get<std::string>("prefix", "")),
      inference_(options->get<bool>("inference", false)) {
  modelFeatures_ = {"type",
                    "dim-vocabs",
                    "dim-emb",
                    "dim-rnn",
                    "enc-cell",
                    "enc-type",
                    "enc-cell-depth",
                    "enc-depth",
                    "dec-depth",
                    "dec-cell",
                    "dec-cell-base-depth",
                    "dec-cell-high-depth",
                    "skip",
                    "layer-normalization",
                    "right-left",
                    "input-types",
                    "special-vocab",
                    "tied-embeddings",
                    "tied-embeddings-src",
                    "tied-embeddings-all"};

  modelFeatures_.insert("transformer-heads");
  modelFeatures_.insert("transformer-no-projection");
  modelFeatures_.insert("transformer-dim-ffn");
  modelFeatures_.insert("transformer-ffn-depth");
  modelFeatures_.insert("transformer-ffn-activation");
  modelFeatures_.insert("transformer-dim-aan");
  modelFeatures_.insert("transformer-aan-depth");
  modelFeatures_.insert("transformer-aan-activation");
  modelFeatures_.insert("transformer-aan-nogate");
  modelFeatures_.insert("transformer-preprocess");
  modelFeatures_.insert("transformer-postprocess");
  modelFeatures_.insert("transformer-postprocess-emb");
  modelFeatures_.insert("transformer-postprocess-top");
  modelFeatures_.insert("transformer-decoder-autoreg");
  modelFeatures_.insert("transformer-tied-layers");
  modelFeatures_.insert("transformer-guided-alignment-layer");
  modelFeatures_.insert("transformer-train-position-embeddings");

  modelFeatures_.insert("bert-train-type-embeddings");
  modelFeatures_.insert("bert-type-vocab-size");

  modelFeatures_.insert("ulr");
  modelFeatures_.insert("ulr-trainable-transformation");
  modelFeatures_.insert("ulr-dim-emb");
  modelFeatures_.insert("lemma-dim-emb");
}

void EncoderCTCDecoder::clear(Ptr<ExpressionGraph> graph) {
  graph->clear();

  for(auto& enc : encoders_)
    enc->clear();

  // TODO ??
  //ctcDecoder_.clear();
}

Logits EncoderCTCDecoder::build(Ptr<ExpressionGraph> graph,
                                Ptr<data::CorpusBatch> batch,
                                bool clearGraph) {
  if(clearGraph)
    clear(graph);

  std::vector<Ptr<EncoderState>> encoderStates;
  for(auto& encoder : encoders_)
      encoderStates.push_back(encoder->build(graph, batch));

  return Logits(ctcDecoder_->apply(graph, batch, encoderStates));
}

Logits EncoderCTCDecoder::build(Ptr<ExpressionGraph> graph,
                                Ptr<data::Batch> batch,
                                bool clearGraph) {
  auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
  return build(graph, corpusBatch, clearGraph);
}


} // namespace marian