#if __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "ctcdecode.h"
#undef LOG

#if __GNUC__
#pragma GCC diagnostic pop
#endif

#include "ctc/beam_search.h"

namespace marian {

CTCResults CTCBeamSearch::search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {

  // const auto trgEosId = trgVocab_->getEosId();
  // const auto trgUnkId = trgVocab_->getUnkId();
  // const auto trgBlankId = trgVocab_->getBlankId(); // TODO TODO

  scorer_->clear(graph);

  auto logits = scorer_->score(graph, batch);

  // get probabilities
  Expr probs = softmax(logits.getLogits());

  // NN computation
  graph->forward();

  int max_length = probs->shape()[1];
  int batch_size = probs->shape()[2];
  int vocab_size = probs->shape()[3];

  std::vector<float> probs_flat;
  probs->val()->get(probs_flat);

  // probs in shape 1(beam), time, batch, vocab
  // need to convert to vectors of (batch, time, vocab)

  // This initializes a 3D vector
  std::vector<std::vector<std::vector<double>>> probs_seq(
    batch_size, std::vector<std::vector<double>>(
      max_length, std::vector<double>(vocab_size)));

  // TODO I am sure this can be done more efficiently
  for(int b = 0; b < batch_size; ++b) {
    for(int t = 0; t < max_length; ++t) {
      for(int i = 0; i < vocab_size; ++i) {
        probs_seq[b][t][i] = (double)probs_flat[t * b + i];
      }
    }
  }

  // CALL ctcdecode-cpp library function
  auto batch_res = ctcdecode::ctc_beam_search_decoder_batch(/*probs_seq=*/ probs_seq,
						 /*vocabulary=*/ vocab_size,
						 /*beam_size=*/ beamSize_,
						 /*num_processes=*/ 1,
						 /*cutoff_prob=*/ 1.0,
						 /*cutoff_top_n=*/ 40,
						 /*blank_id=*/ 3, // TODO
						 /*log_input=*/ 0,
						 /*ext_scorer=*/ nullptr);

  // that returned vector<vector<std::pair<double, Output>>>, where
  // Output has vector<int> tokens and vector<int> timesteps

  // dims: batch, beam. the dim inside Output is time which is sparse but in correct order

  // the output printer could might as well take the batch_res as-is, but let's parse it.
  CTCResults results(batch_size);

  for(int i = 0; i < batch_size; ++i) {
    size_t sentId = batch->getSentenceIds()[i];

    auto best = batch_res[i][0];
    double score = std::get<0>(best);
    auto output = std::get<1>(best);

    auto tokens = output.tokens;
    //auto timesteps = output.timesteps;

    Words decoded;

    for(int token : tokens) {
      decoded.push_back(Word::fromWordIndex(token));
    }

    results[i] = New<CTCSearchResult>(sentId, decoded, score);
  }

  return results;
}

} // namespace marian