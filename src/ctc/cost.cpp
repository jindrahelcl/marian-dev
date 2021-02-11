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
  auto labelLengthsData = corpusBatch->back()->sentenceLengths();
  auto inputLengthsData = corpusBatch->front()->sentenceLengths();

  // TODO make these integer tensors, now there are conversions from size_t to float and then to int.
  auto flatLabels = graph->constant(
    {(int)flatLabelsData.size()},
    inits::fromVector(toWordIndexVector(flatLabelsData)));

  auto labelLengths = graph->constant(
    {(int)labelLengthsData.size()},
    inits::fromVector(std::vector<float>(labelLengthsData.begin(), labelLengthsData.end())));

  auto inputLengths = graph->constant(
    {(int)labelLengthsData.size()},
    inits::fromVector(std::vector<float>(inputLengthsData.begin(), inputLengthsData.end())));

  flatLabels->set_name("flat-labels");
  labelLengths->set_name("label-lengths");

  // Input lengths to the CTC loss.
  // Computed from the input mask and the split factor.
  int split_factor = options_->get<int>("ctc-split-factor", 3);
  Expr inputLengthsAfterSplit = inputLengths * split_factor;

  // calculate CTC costs per sentence and sum costs along the batch axis
  Expr loss = cast(ctc_loss(logits.getLogits(), flatLabels, labelLengths, inputLengthsAfterSplit), Type::float32);
  loss = sum(loss, -2);

  multiLoss->push_back(RationalLoss(loss, labelLengths->shape().elements()));
  return multiLoss;
}

} // namespace models
} // namespace marian