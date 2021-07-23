#pragma once

#include <stdexcept>
#include <vector>
#include <limits>
#include <numeric>


#include <ctc.h>

inline void throw_on_error(ctcStatus_t status, const char* message) {
    if (status != CTC_STATUS_SUCCESS) {
        throw std::runtime_error(message + (", stat = " + 
                                            std::string(ctcGetStatusString(status))));
    }
}

#ifdef __CUDACC__
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

inline void throw_on_error(cudaError_t error, const char* message) {
    if (error) {
        throw thrust::system_error(error, thrust::cuda_category(), message);
    }
}

#endif

std::vector<float> genActs(int size);
std::vector<int> genLabels(int alphabet_size, int L);

float rel_diff(const std::vector<float>& grad,
               const std::vector<float>& num_grad) {
    float diff = 0.;
    float tot = 0.;
    for(size_t idx = 0; idx < grad.size(); ++idx) {
        diff += (grad[idx] - num_grad[idx]) * (grad[idx] - num_grad[idx]);
        tot += grad[idx] * grad[idx];
    }

    return diff / tot;
}

// Numerically stable softmax for a minibatch of 1
void softmax(const float* const acts,
             int alphabet_size, int T,
             float *probs) {

    for (int t = 0; t < T; ++t) {

        float max_activation =
            -std::numeric_limits<float>::infinity();

        for (int a = 0; a < alphabet_size; ++a)
            max_activation =
               std::max(max_activation, acts[t*alphabet_size + a]);

        float denom = 0;
        for (int a = 0; a < alphabet_size; ++a)
            denom += std::exp(acts[t*alphabet_size + a] - max_activation);

        for (int a = 0; a < alphabet_size; ++a)
            probs[t*alphabet_size + a] =
               std::exp(acts[t*alphabet_size + a] - max_activation) / denom;
    }
}
