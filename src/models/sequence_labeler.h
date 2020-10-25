#pragma once

#include "marian.h"
#include "models/transformer.h"
#include "models/encoder_classifier.h"

namespace marian {

class SequenceLabeler : public ClassifierBase {
  using ClassifierBase::ClassifierBase;

public:

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

  Ptr<ClassifierState> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    // based on bert.h which is generic enough. Also, the fact that
    // classifierstate is stateful comes in handy.

    ABORT_IF(encoderStates.size() != 1, "Multiple encoders not supported");

    auto context = encoderStates[0]->getContext();  // the actual values of encoder states
    // shape is [beam depth=1, max length, batch size, vector dim]

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

    auto state = New<ClassifierState>();
    state->setLogProbs(logits);

    return state;
  }

  virtual void clear() override {}

};

} // namespace marian