#include "ctc/warpctc.h"
#include "common/shape.h"
#include "tensors/tensor.h"
#include "graph/expression_graph.h"

#include <ctc.h>

namespace marian {

void WarpCTCWrapper::compute(Tensor loss,
                             Tensor grads,
                             Tensor logits,
                             Tensor flatLabels,
                             Tensor labelLengths,
                             const Ptr<ExpressionGraph> graph) {

  Shape logitsShape = logits->shape();

  // in the first dimension is 1.
  ABORT_IF(logitsShape[0] != 1, "First dimension must be 1");
  int time = logitsShape[1];
  int batch = logitsShape[2];
  int vocab = logitsShape[3];

  std::vector<int> input_lengths;
  for(int i = 0; i < batch; i++) {
    input_lengths.push_back(time);
  }

  std::vector<float> activations;
  logits->get(activations);

  std::vector<float> flat_labels;
  flatLabels->get(flat_labels);

  std::vector<int> int_flat_labels;
  for(auto item : flat_labels) {
    int_flat_labels.push_back(static_cast<int>(item));
  }

  std::vector<float> label_lengths;
  labelLengths->get(label_lengths);

  std::vector<int> int_label_lengths;
  for(auto item : label_lengths) {
    int_label_lengths.push_back(static_cast<int>(item));
  }

  int alphabet_size = vocab;
  int mini_batch = batch;

  std::vector<float> costs;
  costs.resize(loss->size());

  std::vector<float> gradients;
  gradients.resize(grads->size());

  ctcOptions options;
  options.loc = CTC_CPU;
  options.num_threads = 16;
  options.blank_label = 3; // TODO

  size_t workspace_size;
  WARP_CALL(get_workspace_size(
    int_label_lengths.data(), //const
    input_lengths.data(), //const
    alphabet_size, //const
    mini_batch,  //const
    options,
    &workspace_size));

  //MemoryPiece::PtrType workspace = graph->allocator()->alloc(gpuWorkspaceSize);

  void *workspace = malloc(workspace_size);

  WARP_CALL(compute_ctc_loss(
    activations.data(), //const
    gradients.data(),
    int_flat_labels.data(), //const
    int_label_lengths.data(), //const
    input_lengths.data(), //const
    alphabet_size, //const
    mini_batch,
    costs.data(),
    workspace,
    options));

  free(workspace);

  loss->set(costs);
  grads->set(gradients);
}

} //namespace marian