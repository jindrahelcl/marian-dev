#include "ctc/cost.h"

#include "ctc/nodeops.h"
#include "ctc/model.h"
#include "ctc/decoder.h"

namespace marian {
namespace models {

Ptr<MultiRationalLoss> CTCCost::apply(Ptr<IModel> model,
                                      Ptr<ExpressionGraph> graph,
                                      Ptr<data::Batch> batch,
                                      bool clearGraph) {
  auto encdec = std::static_pointer_cast<EncoderCTCDecoder>(model);
  auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);

  auto logits = encdec->build(graph, corpusBatch, clearGraph);

  Ptr<MultiRationalLoss> multiLoss = newMultiLoss(options_);

  const auto& flatLabelsData = corpusBatch->back()->flatData();
  const auto& labelLengthsData = corpusBatch->back()->sentenceLengths();


  auto flatLabels = graph->constant(
    {(int)flatLabelsData.size()},
    inits::fromVector(toWordIndexVector(flatLabelsData)));

  auto labelLengths = graph->constant(
    {(int)labelLengthsData.size()},
    inits::fromVector(std::vector<float>(
      labelLengthsData.begin(),
      labelLengthsData.end())));

  flatLabels->set_name("flat-labels");
  labelLengths->set_name("label-lengths");

  // calculate CTC costs per sentence and sum costs along the batch axis
  Expr loss = cast(ctc_loss(logits.getLogits(), flatLabels, labelLengths), Type::float32);
  loss = sum(loss, -2);

  multiLoss->push_back(RationalLoss(loss, labelLengths->shape().elements()));
  return multiLoss;
}

} // namespace models
} // namespace marian