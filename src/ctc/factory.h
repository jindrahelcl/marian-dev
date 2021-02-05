#pragma once

#include "marian.h"
#include "layers/factory.h"
#include "models/model_factory.h"

#include "ctc/model.h"
#include "ctc/decoder.h"

namespace marian {
namespace models {

class CTCDecoderFactory : public Factory {
  using Factory::Factory;
public:
  virtual Ptr<CTCDecoder> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<CTCDecoderFactory> ctc_decoder;

class EncoderCTCDecoderFactory : public Factory {
  using Factory::Factory;
private:
  std::vector<encoder> encoders_;
  std::vector<ctc_decoder> decoders_;

public:
  Accumulator<EncoderCTCDecoderFactory> push_back(encoder enc) {
    encoders_.push_back(enc);
    return Accumulator<EncoderCTCDecoderFactory>(*this);
  }

  Accumulator<EncoderCTCDecoderFactory> push_back(ctc_decoder dec) {
    ABORT_IF(decoders_.size() > 0, "CTC Decoder already set");
    decoders_.push_back(dec);
    return Accumulator<EncoderCTCDecoderFactory>(*this);
  }

  virtual Ptr<IModel> construct(Ptr<ExpressionGraph> graph);
};

typedef Accumulator<EncoderCTCDecoderFactory> encoder_ctc_decoder;

} // namespace models
} // namespace marian