#pragma once

#include "marian.h"

#include "models/encoder.h"
#include "models/model_base.h"
#include "models/states.h"
#include "ctc/decoder.h"

namespace marian {

class EncoderCTCDecoder : public models::IModel {
protected:
  Ptr<Options> options_;
  std::string prefix_;
  std::vector<Ptr<EncoderBase>> encoders_;
  Ptr<CTCDecoder> ctcDecoder_;
  bool inference_{false};
  std::set<std::string> modelFeatures_;

  Config::YamlNode getModelParameters();

  std::string getModelParametersAsString();

public:
  typedef data::Corpus dataset_type;

  EncoderCTCDecoder(Ptr<Options> options);

  virtual ~EncoderCTCDecoder() {}

  Logits build(Ptr<ExpressionGraph> graph,
               Ptr<data::CorpusBatch> batch,
               bool clearGraph = true);

  Logits build(Ptr<ExpressionGraph> graph,
               Ptr<data::Batch> batch,
               bool clearGraph = true) override;

  Ptr<Options> getOptions() { return options_; }

  std::vector<Ptr<EncoderBase>>& getEncoders() { return encoders_; }
  Ptr<CTCDecoder>& getDecoder() { return ctcDecoder_; }

  void push_back(Ptr<EncoderBase> encoder) { encoders_.push_back(encoder); }
  void setDecoder(Ptr<CTCDecoder> decoder) { ctcDecoder_ = decoder; };

  void load(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool markedReloaded = true) override {
    graph->load(name, markedReloaded && !opt<bool>("ignore-model-config", false));
  }

  void save(Ptr<ExpressionGraph> graph,
            const std::string& name,
            bool /*saveModelConfig*/) override {
    LOG(info, "Saving model weights and runtime parameters to {}", name);
    graph->save(name , getModelParametersAsString());
  }
  void clear(Ptr<ExpressionGraph> graph) override;

  void mmap(Ptr<ExpressionGraph> graph,
            const void* ptr,
            bool markedReloaded = true) {
    graph->mmap(ptr, markedReloaded && !opt<bool>("ignore-model-config", false));
  }

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string& key, const T& def) {
    return options_->get<T>(key, def);
  }

  template <typename T>
  void set(std::string key, T value) {
    options_->set(key, value);
  }

};

}  // namespace marian
