#include "ctc/warpctc.h"
#include "common/shape.h"
#include "tensors/tensor.h"
#include "graph/expression_graph.h"

#include <ctc.h>

#define WARPCTC_GPU 1

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
#ifndef WARPCTC_GPU
  std::vector<float> activations;
  logits->get(activations);
#endif

  std::vector<float> gradients;
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

#ifdef WARPCTC_GPU
  // GPU options
  options.loc = CTC_GPU;
  options.stream = CU_STREAM_LEGACY;

#else
  // CPU options
  options.loc = CTC_CPU;
  options.num_threads = 16;
#endif

  size_t workspace_size;
  WARP_CALL(get_workspace_size(
    int_label_lengths.data(), //const
    int_input_lengths.data(), //const
    vocab, //const
    batch,  //const
    options,
    &workspace_size));

#ifdef WARPCTC_GPU
  MemoryPiece::PtrType workspace = graph->allocator()->alloc(workspace_size);

  WARP_CALL(compute_ctc_loss(
    logits->data(),
    grads->data(),
    int_flat_labels.data(), //const
    int_label_lengths.data(), //const
    int_input_lengths.data(), //const
    vocab, //const
    batch,
    costs.data(),
    workspace->data<void>(),
    options));

  graph->allocator()->free(workspace);
#else
  void *workspace = malloc(workspace_size);

  WARP_CALL(compute_ctc_loss(
    activations.data(), //const
    gradients.data(),
    int_flat_labels.data(), //const
    int_label_lengths.data(), //const
    int_input_lengths.data(), //const
    vocab, //const
    batch,
    costs.data(),
    workspace,
    options));

  free(workspace);

  // move gradients back to GPU memory (only with CPU warpctc)
  grads->set(gradients);
#endif

    // move loss back to GPU memory (always)
  loss->set(costs);
}

} //namespace marian
