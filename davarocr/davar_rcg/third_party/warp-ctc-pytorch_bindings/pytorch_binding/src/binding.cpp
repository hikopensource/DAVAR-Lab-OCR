#include <iostream>
#include <vector>

#include <numeric>

#include <torch/extension.h>

#ifdef WARPCTC_ENABLE_GPU
	#include "ATen/cuda/CUDAContext.h"
	#include <c10/cuda/CUDAGuard.h>
	#include "ATen/cuda/CUDAEvent.h"

    #include "THC.h"
    extern THCState* state;
#endif

#include "ctc.h"

int cpu_ctc(torch::Tensor probs,
            torch::Tensor grads,
            torch::Tensor labels,
            torch::Tensor label_sizes,
            torch::Tensor sizes,
            int minibatch_size,
            torch::Tensor costs,
            int blank_label)
{
    float* probs_ptr       = (float*)probs.data_ptr();
    float* grads_ptr       = grads.storage() ? (float*)grads.data_ptr() : NULL;
    int*   sizes_ptr       = (int*)sizes.data_ptr();
    int*   labels_ptr      = (int*)labels.data_ptr();
    int*   label_sizes_ptr = (int*)label_sizes.data_ptr();
    float* costs_ptr       = (float*)costs.data_ptr();

    const int probs_size = probs.size(2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads
    options.blank_label = blank_label;

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       probs_size, minibatch_size,
                       options, &cpu_size_bytes);

    float* cpu_workspace = new float[cpu_size_bytes / sizeof(float)];

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs_size,
                     minibatch_size, costs_ptr,
                     cpu_workspace, options);

    delete[] cpu_workspace;
    return 1;
}

#ifdef WARPCTC_ENABLE_GPU
int gpu_ctc(torch::Tensor probs,
            torch::Tensor grads,
            torch::Tensor labels,
            torch::Tensor label_sizes,
            torch::Tensor sizes,
            int minibatch_size,
            torch::Tensor costs,
            int blank_label)
{
    float* probs_ptr       = (float*)probs.data_ptr();
    float* grads_ptr       = grads.storage() ? (float*)grads.data_ptr() : NULL;
    int*   sizes_ptr       = (int*)sizes.data_ptr();
    int*   labels_ptr      = (int*)labels.data_ptr();
    int*   label_sizes_ptr = (int*)label_sizes.data_ptr();
    float* costs_ptr       = (float*)costs.data_ptr();

    const int probs_size = probs.size(2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_GPU;
    options.blank_label = blank_label;
    options.stream = at::cuda::getCurrentCUDAStream();

    size_t gpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       probs_size, minibatch_size,
                       options, &gpu_size_bytes);

    void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs_size,
                     minibatch_size, costs_ptr,
                     gpu_workspace, options);

    THCudaFree(state, (void *) gpu_workspace);
    return 1;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_ctc", &cpu_ctc, "CTC Loss function with cpu");
#ifdef WARPCTC_ENABLE_GPU
  m.def("gpu_ctc", &gpu_ctc, "CTC Loss function with gpu");
#endif
}
