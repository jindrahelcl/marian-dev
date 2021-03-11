#include "ctc/translator.h"
#include "ctc/beam_search.h"
#include "ctc/output_printer.h"

#include "data/corpus.h"
#include "data/text_input.h"
#include "translator/output_collector.h"

#include "3rd_party/threadpool.h"


namespace marian {

NARTranslate::NARTranslate(Ptr<Options> options) : options_(New<Options>(options->clone())) {
  options->set("inference", true, "shuffle", "none");

  // std::string type = options->get<std::string>("type");
  // ABORT_IF(type != "transformer-nat", "Only use with NAT Transformer for now");

  corpus_ = New<data::Corpus>(options_, true);

  auto vocabs = options_->get<std::vector<std::string>>("vocabs");
  trgVocab_ = New<Vocab>(options_, vocabs.size() - 1);
  trgVocab_->load(vocabs.back());
  auto srcVocab = corpus_->getVocabs()[0];

  // TODO shortlist initialization here

  auto devices = Config::getDevices(options_);
  numDevices_ = devices.size();

  ThreadPool threadPool(numDevices_, numDevices_);
  scorers_.resize(numDevices_);
  graphs_.resize(numDevices_);

  size_t id = 0;
  for(auto device : devices) {

    auto task = [&](DeviceId device, size_t id) {
      auto graph = New<ExpressionGraph>(true);
      auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});

      graph->setDefaultElementType(typeFromString(prec[0]));
      graph->setDevice(device);
      //graph->getBackend()->setClip(options_->get<float>("clip-gemm"));
      //if (device.type == DeviceType::cpu) {
      //  graph->getBackend()->setOptimized(options_->get<bool>("optimize"));
      //}
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_[id] = graph;

      auto scorer = createNATScorer(options_);
      scorer->init(graph);

      // TODO init shortlist generator for scorer

      scorers_[id] = scorer;
      graph->forward();
    };

    threadPool.enqueue(task, device, id++);
  }

  ABORT_IF(options_->get<bool>("output-sampling", false), "output sampling not supported (yet?)");
}

void NARTranslate::run() {
  data::BatchGenerator<data::Corpus> bg(corpus_, options_);

  ThreadPool threadPool(numDevices_, numDevices_);

  size_t batchId = 0;
  auto collector = New<OutputCollector>(options_->get<std::string>("output"));
  auto printer = New<CTCOutputPrinter>(options_, trgVocab_);
  if(options_->get<bool>("quiet-translation"))
    collector->setPrintingStrategy(New<QuietPrinting>());

  bg.prepare();

  bool doNbest = options_->get<bool>("n-best");
  ABORT_IF(doNbest, "n best not supported at the moment");

  for(auto batch : bg) {
    auto task = [=](size_t id) {
      thread_local Ptr<ExpressionGraph> graph;
      thread_local Ptr<NATScorer> scorer;

      if(!graph) {
        graph = graphs_[id % numDevices_];
        scorer = scorers_[id % numDevices_];
      }

      auto search = New<CTCBeamSearch>(options_, scorer, trgVocab_);
      auto results = search->search(graph, batch);

      for(auto result : results) { // loop over batch
        std::stringstream best1;
        printer->print(result, best1);
        collector->Write((long)result->getLineNum(), best1.str(), "", false);
      }
    };

    threadPool.enqueue(task, batchId++);
  }
}

} // namespace marian
