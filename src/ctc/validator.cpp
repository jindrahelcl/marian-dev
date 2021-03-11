#include "ctc/validator.h"
#include "ctc/nat_scorer.h"
#include "ctc/beam_search.h"
#include "ctc/output_printer.h"


namespace marian {

float NATSacreBleuValidator::validate(const std::vector<Ptr<ExpressionGraph>>& graphs,
				      Ptr<const TrainingState> state) {
  using namespace data;

  // Generate batches
  batchGenerator_->prepare();

  // Create scorer
  auto model = options_->get<std::string>("model");

  // @TODO: check if required - Temporary options for translation
  auto mopts = New<Options>();
  mopts->merge(options_);
  mopts->set("inference", true);

  std::vector<Ptr<NATScorer>> scorers;
  for(auto graph : graphs) {
    auto builder = models::createModelFromOptions(options_, models::usage::raw);
    Ptr<NATScorer> scorer = New<NATScorer>(builder, "", 1.0f, model);
    scorers.push_back(scorer);
  }

  for(auto graph : graphs)
    graph->setInference(true);

  if(!quiet_)
    LOG(info, "Translating validation set...");

  // For BLEU
  // 0: 1-grams matched, 1: 1-grams cand total, 2: 1-grams ref total (used in ChrF)
  // ...,
  // n: reference length (used in BLEU)
  std::vector<float> stats(statsPerOrder * order_ + 1, 0.f);

  timer::Timer timer;
  {
    auto printer = New<CTCOutputPrinter>(options_, vocabs_.back());

    Ptr<OutputCollector> collector;
    if(options_->hasAndNotEmpty("valid-translation-output")) {
      auto fileName = options_->get<std::string>("valid-translation-output");
      // fileName can be a template with fields for training state parameters:
      fileName = state->fillTemplate(fileName);
      collector = New<OutputCollector>(fileName);  // for debugging
    } else {
      collector = New<OutputCollector>(/* null */);  // don't print, but log
    }

    if(quiet_)
      collector->setPrintingStrategy(New<QuietPrinting>());
    else
      collector->setPrintingStrategy(New<GeometricPrinting>());

    std::deque<Ptr<ExpressionGraph>> graphQueue(graphs.begin(), graphs.end());
    std::deque<Ptr<NATScorer>> scorerQueue(scorers.begin(), scorers.end());
    auto task = [=, &stats, &graphQueue, &scorerQueue](BatchPtr batch) {
      thread_local Ptr<ExpressionGraph> graph;
      thread_local Ptr<NATScorer> scorer;

      if(!graph) {
        std::unique_lock<std::mutex> lock(mutex_);
        ABORT_IF(graphQueue.empty(), "Asking for graph, but none left on queue");
        graph = graphQueue.front();
        graphQueue.pop_front();

        ABORT_IF(scorerQueue.empty(), "Asking for scorer, but none left on queue");
        scorer = scorerQueue.front();
        scorerQueue.pop_front();
      }

      auto search = New<CTCBeamSearch>(options_, scorer, vocabs_.back());
      auto results = search->search(graph, batch);

      size_t no = 0;
      std::lock_guard<std::mutex> statsLock(mutex_);
      for(auto result : results) {
        const auto& words = result->getWords();
        updateStats(stats, words, batch, no);

        std::stringstream best1;
        printer->print(result, best1);
        collector->Write((long)result->getLineNum(),
                         best1.str(),
                         "",
                         /*nbest=*/false);
        no++;
      }
    };

    threadPool_.reserve(graphs.size());
    TaskBarrier taskBarrier;
    for(auto batch : *batchGenerator_)
      taskBarrier.push_back(threadPool_.enqueue(task, batch));
    // ~TaskBarrier waits until all are done
  }

  //if(!quiet_)
  LOG(info, "Total translation time: {:.5f}s", timer.elapsed());

  for(auto graph : graphs)
    graph->setInference(false);

  float val = computeChrF_ ? calcChrF(stats) : calcBLEU(stats);
  updateStalled(graphs, val);

  return val;
}


} // namespace marian
