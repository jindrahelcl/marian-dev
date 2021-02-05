#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#include "ctc/translator.h"
#include "common/timer.h"
#ifdef _WIN32
#include <Windows.h>
#endif

int main(int argc, char** argv) {
  using namespace marian;
  auto options = parseOptions(argc, argv, cli::mode::translation);

  Ptr<ModelTask> task;
  if(options->get<std::string>("type") == "transformer-nat")
    task = New<NARTranslate>(options);
  else
    task = New<Translate<BeamSearch>>(options);

  timer::Timer timer;
  task->run();
  LOG(info, "Total time: {:.5f}s wall", timer.elapsed());

  return 0;
}
