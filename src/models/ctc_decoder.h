#pragma once

#include "marian.h"
#include "models/transformer.h"
#include "models/encoder_classifier.h"

namespace marian {

class CTCDecoderResult : public ClassifierState {
  using ClassifierState::ClassifierState;

private:
  std::vector<size_t> targetLengths_;

public:
  virtual ~CTCDecoderResult() {}

  virtual const std::vector<size_t>& getTargetLengths() const { return targetLengths_; };
  virtual void setTargetLengths(const std::vector<size_t> targetLengths) { targetLengths_ = targetLengths; }
};


class CTCDecoder : public LayerBase {
  using LayerBase::LayerBase;

protected:
  const std::string prefix_{"ctc-decoder"};
  const bool inference_{false};
  const size_t batchIndex_{0};

public:
  CTCDecoder(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : LayerBase(graph, options),
        prefix_(options->get<std::string>("prefix", "ctc-decoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 1)) {} // assume that training input has batch index 0 and labels has 1

  Ptr<CTCDecoderResult> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) {
    // based on bert.h which is generic enough. Also, the fact that
    // classifierstate is stateful comes in handy.

    ABORT_IF(encoderStates.size() != 1, "Multiple encoders not supported");

    auto context = encoderStates[0]->getContext();  // the actual values of encoder states
    // shape is [beam depth=1, max length, batch size, vector dim]

    // int dimModel = context->shape()[-1];
    // int dimBatch = context->shape()[-2];
    //int dimTime  = context->shape()[-3];

    //int dimModel = context->shape()[-1];
    int dimTrgCls = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    // TODO perhaps add hidden layer?
    auto output = mlp::mlp()
                    .push_back(mlp::output(
                                 "dim", dimTrgCls,
                                 "prefix", prefix_ + "ff_logit_out")
                                 .tieTransposed("Wemb"))
                    .construct(graph);

    auto logits = output->apply(context);
    // shape is [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab or shortlist dim]

    auto result = New<CTCDecoderResult>();
    result->setLogProbs(logits);

    const auto& flatLabels = (*batch)[batchIndex_]->flatData();
    result->setTargetWords(flatLabels);

    //int dimTrgTime = classLabels.size() / dimBatch;

    const auto& labelLengths = (*batch)[batchIndex_]->sentenceLengths();
    result->setTargetLengths(labelLengths);

    //const auto& mask = (*batch)[batchIndex_]->mask();
    //auto batchMask = graph->constant({dimTrgTime, dimBatch, 1}, inits::fromVector(mask));
    //result->setTargetMask(batchMask);

    return result;
  }

  virtual ~CTCDecoder() {}
};

} // namespace marian





  // // TODO this function is copied from transformer.h
  // void lazyCreateOutputLayer() {
  //   if(output_) // create it lazily
  //     return;

  //   int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

  //   auto outputFactory = mlp::OutputFactory(
  //       "prefix", prefix_ + "_ff_logit_out",
  //       "dim", dimTrgVoc,
  //       "vocab", opt<std::vector<std::string>>("vocabs")[batchIndex_], // for factored outputs
  //       "output-omit-bias", opt<bool>("output-omit-bias", false),
  //       "output-approx-knn", opt<std::vector<int>>("output-approx-knn", {}),
  //       "lemma-dim-emb", opt<int>("lemma-dim-emb", 0)); // for factored outputs

  //   if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all"))
  //     outputFactory.tieTransposed(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src") ? "Wemb" : prefix_ + "_Wemb");

  //   output_ = std::dynamic_pointer_cast<mlp::Output>(outputFactory.construct(graph_)); // (construct() returns only the underlying interface)
  // }