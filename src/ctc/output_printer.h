#include <sstream>

#include "marian.h"
#include "ctc/beam_search.h"

namespace marian {

class CTCOutputPrinter {
private:
  Ptr<Vocab const> vocab_;

public:
  CTCOutputPrinter(Ptr<const Options> options, Ptr<const Vocab> vocab)
    : vocab_(vocab) {}


  template <class OStream>
  void print(Ptr<const CTCSearchResult> result, OStream& best) {

    auto words = result->getWords();
    std::string translation = vocab_->decode(words);

    best << translation;
    best << std::flush;
  }
};

} // namespace marian