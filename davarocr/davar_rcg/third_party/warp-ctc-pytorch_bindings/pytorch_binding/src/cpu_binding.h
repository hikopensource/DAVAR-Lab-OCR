/*
int cpu_ctc(THFloatTensor *probs,
                        THFloatTensor *grads,
                        THIntTensor *labels_ptr,
                        THIntTensor *label_sizes_ptr,
                        THIntTensor *sizes,
                        int minibatch_size,
                        THFloatTensor *costs,
                        int blank_label);
*/

int cpu_ctc(torch::Tensor probs,
            torch::Tensor grads,
            torch::Tensor labels,
            torch::Tensor label_sizes,
            torch::Tensor sizes,
            int minibatch_size,
            torch::Tensor costs,
            int blank_label); 
