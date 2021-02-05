#include "ctc/factory.h"

namespace marian {
namespace models {

Ptr<CTCDecoder> CTCDecoderFactory::construct(Ptr<ExpressionGraph> graph) {
  return New<CTCDecoder>(graph, options_);
}


Ptr<IModel> EncoderCTCDecoderFactory::construct(Ptr<ExpressionGraph> graph) {
  Ptr<EncoderCTCDecoder> model;
  model = New<EncoderCTCDecoder>(options_);

  for(auto& ef : encoders_)
    model->push_back(ef(options_).construct(graph));

  model->setDecoder(decoders_[0](options_).construct(graph));

  return model;
}

} // namespace models
} // namespace marian