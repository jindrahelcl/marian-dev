#include "ctc/decoder.h"

namespace marian {

Expr CTCDecoder::apply(Ptr<ExpressionGraph> graph,
                       Ptr<data::CorpusBatch> batch,
                       const std::vector<Ptr<EncoderState>>& encoderStates) {
  // based on bert.h which is generic enough. Also, the fact that
  // classifierstate is stateful comes in handy.
  graph_ = graph;
  ABORT_IF(encoderStates.size() != 1, "Multiple encoders not supported");

  // move to batch-major mode
  auto encoderOutput = transposeTimeBatch(encoderStates[0]->getContext());
  auto encoderMask = transposeTimeBatch(encoderStates[0]->getMask());

  //int dimBeam = encoderOutput->shape()[-4];
  int dimBatch = encoderOutput->shape()[-3];
  int dimSrcWords = encoderOutput->shape()[-2];
  //int dimModel = encoderOutput->shape()[-1];

  auto decoderMask = expandMask(encoderMask);
  auto decoderInput = splitStates(encoderOutput);
  auto layer = decoderInput;
  auto prevLayer = layer;

  auto opsEmb = opt<std::string>("transformer-postprocess-emb");
  float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");

  layer = preProcess(prefix_ + "_emb", opsEmb, layer, dropProb);

  // LayerAttention expects mask in a different layout
  auto layerMask = reshape(decoderMask, {1, dimBatch, 1, dimSrcWords * splitFactor_});
  // [1, batch size, 1, max length]
  layerMask = transposedLogMask(layerMask);
  // [batch size, num heads broadcast=1, max length broadcast=1, max length]

  // apply layers
  auto depth = opt<int>("dec-depth");
  for(int i = 1; i <= depth; ++i) {
    layer = LayerAttention(prefix_ + "_l" + std::to_string(i) + "_self",
      layer, layer, layer, layerMask, opt<int>("transformer-heads"));

    layer = LayerAttention(prefix_ + "_l" + std::to_string(i) + "_cross",
      layer, decoderInput, decoderInput, layerMask, opt<int>("transformer-heads"));

    layer = LayerFFN(prefix_ + "_l" + std::to_string(i) + "_ffn", layer);
    checkpoint(layer); // sets a manually specified checkpoint if gradient checkpointing is enabled, does nothing otherwise.
  }

  // this allows to run a final layernorm operation after going through the transformer layer stack.
  // By default the operations are empty, but with prenorm (--transformer-preprocess n --transformer-postprocess da)
  // it is recommended to normalize here. Can also be used to add a skip connection from the very bottom if requested.
  auto opsTop = opt<std::string>("transformer-postprocess-top", "");
  layer = postProcess(prefix_ + "_top", opsTop, layer, prevLayer, dropProb);

  // move back to time major mode
  auto context = transposeTimeBatch(layer);

  // END TRANSFORMER BLOCK

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

  return logits;
}

} // namespace marian