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
                             Tensor inputLengths,
                             const Ptr<ExpressionGraph> graph) {

  Shape logitsShape = logits->shape();

  // in the first dimension is 1.
  ABORT_IF(logitsShape[0] != 1, "First dimension must be 1");
  // int time = logitsShape[1];
  int batch = logitsShape[2];
  int vocab = logitsShape[3];

  // Move activations and gradients to CPU memory (only for CPU warpctc)
  std::vector<float> activations;
  std::vector<float> gradients;

  logits->get(activations);
  gradients.resize(grads->size());

  // Costs are always on CPU
  std::vector<float> costs;
  costs.resize(loss->size());

  // Move flat labels and label lengths to CPU memory (always on CPU)
  std::vector<float> flat_labels;
  std::vector<float> label_lengths;
  std::vector<float> input_lengths;

  flatLabels->get(flat_labels);
  labelLengths->get(label_lengths);
  inputLengths->get(input_lengths);

  std::vector<int> int_flat_labels(flat_labels.begin(), flat_labels.end());
  std::vector<int> int_label_lengths(label_lengths.begin(), label_lengths.end());
  std::vector<int> int_input_lengths(input_lengths.begin(), input_lengths.end());

  ctcOptions options;
  options.blank_label = blankLabelIdx_;

  // GPU options
  //options.loc = CTC_GPU;
  //options.stream = CU_STREAM_LEGACY;

  // CPU options
  options.loc = CTC_CPU;
  options.num_threads = 16;

  size_t workspace_size;
  WARP_CALL(get_workspace_size(
    int_label_lengths.data(), //const
    int_input_lengths.data(), //const
    vocab, //const
    batch,  //const
    options,
    &workspace_size));

  //MemoryPiece::PtrType workspace = graph->allocator()->alloc(workspace_size);

  void *workspace = malloc(workspace_size);

  WARP_CALL(compute_ctc_loss(
    activations.data(), //const
    //logits->data(),
    gradients.data(),
    //grads->data(),
    int_flat_labels.data(), //const
    int_label_lengths.data(), //const
    int_input_lengths.data(), //const
    vocab, //const
    batch,
    costs.data(),
    workspace,
    //workspace->data<void>(),
    options));

  //graph->allocator()->free(workspace);
  free(workspace);

  // move loss back to GPU memory (always)
  loss->set(costs);

  // move gradients back to GPU memory (only with CPU warpctc)
  grads->set(gradients);
}

} //namespace marian