import torch
import warpctc_pytorch as warp_ctc
import pytest


def test_simple():
    probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
    grads = torch.zeros(probs.size())
    labels = torch.IntTensor([1, 2])
    label_sizes = torch.IntTensor([2])
    sizes = torch.IntTensor(probs.size(1)).fill_(probs.size(0))
    minibatch_size = probs.size(1)
    costs = torch.zeros(minibatch_size)
    warp_ctc.cpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs,
                     0)
    print('CPU_cost: %f' % costs.sum())


@pytest.mark.parametrize("multiplier", [1.0, 200.0])
def test_medium(multiplier):
    probs = torch.FloatTensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    ]).contiguous() * multiplier

    grads = torch.zeros(probs.size())
    labels = torch.IntTensor([1, 2, 1, 2])
    label_sizes = torch.IntTensor([2, 2])
    sizes = torch.IntTensor([2, 2])
    minibatch_size = probs.size(1)
    costs = torch.zeros(minibatch_size)
    warp_ctc.cpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs,
                     0)
    print('CPU_cost: %f' % costs.sum())


def test_empty_label():
    probs = torch.FloatTensor([
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
    ]).contiguous()

    grads = torch.zeros(probs.size())
    labels = torch.IntTensor([1, 2])
    label_sizes = torch.IntTensor([2, 0])
    sizes = torch.IntTensor([2, 2])
    minibatch_size = probs.size(1)
    costs = torch.zeros(minibatch_size)
    warp_ctc.cpu_ctc(probs,
                     grads,
                     labels,
                     label_sizes,
                     sizes,
                     minibatch_size,
                     costs,
                     0)
    print('CPU_cost: %f' % costs.sum())


def test_CTCLoss():
    probs = torch.FloatTensor([[
        [0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]
    ]]).transpose(0, 1).contiguous()
    labels = torch.IntTensor([1, 2])
    label_sizes = torch.IntTensor([2])
    probs_sizes = torch.IntTensor([2])
    probs.requires_grad_(True)

    ctc_loss = warp_ctc.CTCLoss()
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    cost.backward()


if __name__ == '__main__':
    pytest.main([__file__])
