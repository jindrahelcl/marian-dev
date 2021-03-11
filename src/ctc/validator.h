#pragma once

#include "marian.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "training/validator.h"
#include "training/training_state.h"
#include "ctc/model.h"

#include <vector>

namespace marian {

/**
 * NATSacreBleuValidator
 *
 * much of the code has been copied from SacreBleuValidator, but adapted to
 * work with NATScorer, CTCBeamSearch, and CTCOutputPrinter.
 */
class NATSacreBleuValidator : public SacreBleuValidator {

public:
  NATSacreBleuValidator(std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options, const std::string& metric)
    : SacreBleuValidator(vocabs, options, metric) {}
  virtual ~NATSacreBleuValidator() {}

  virtual float validate(const std::vector<Ptr<ExpressionGraph>>& graphs,
			 Ptr<const TrainingState> state) override;
};

}
