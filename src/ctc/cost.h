#pragma once

#include "marian.h"
#include "models/costs.h"
#include "models/model_base.h"

namespace marian {
namespace models {

class CTCCost : public ICost {
protected:
  Ptr<Options> options_;
  const bool inference_{false};

public:
  CTCCost(Ptr<Options> options)
      : options_(options), inference_(options->get<bool>("inference", false)) { }

  virtual ~CTCCost() {}

  Ptr<MultiRationalLoss> apply(Ptr<IModel> model,
             Ptr<ExpressionGraph> graph,
             Ptr<data::Batch> batch,
             bool clearGraph = true) override;
};

} // namespace models
} // namespace marian