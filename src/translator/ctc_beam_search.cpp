#include "translator/ctc_beam_search.h"

Histories CTCBeamSearch::search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {

  // We will use the prefix "origBatch..." whenever we refer to batch dimensions of the original batch. These do not change during search.
  // We will use the prefix "currentBatch.." whenever we refer to batch dimension that can change due to batch-pruning.
  const int origDimBatch = (int)batch->size();
  const auto trgEosId = trgVocab_->getEosId();
  const auto trgUnkId = trgVocab_->getUnkId();
  const auto trgBlankId = trgVocab_->getBlankId(); // TODO TODO

  auto getNBestList = createGetNBestListFn(beamSize_, origDimBatch, graph->getDeviceId());


  for(auto scorer : scorers_) {
    scorer->clear(graph);
  }


  // a history object for each sentence in batch
  Histories histories(origDimBatch);
  for(int i = 0; i < origDimBatch; ++i) {
    size_t sentId = batch->getSentenceIds()[i];
    histories[i] = New<History>(sentId)
       // no word penalty and/or normalization
  }

  // start states become the whole computation
  std::vector<Ptr<ScorerState>> states;
  for(auto scorer : scorers_) {
    states.push_back(scorer->startState(graph, batch));
  }




}