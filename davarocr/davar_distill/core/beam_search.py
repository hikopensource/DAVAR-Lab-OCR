"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    beam_search.py
# Abstract       :    Beam search for attention decode

# Current Version:    1.0.0
# Date           :    2022-07-07
##################################################################################################
"""
import torch
from queue import PriorityQueue


class BeamSearchNode(object):
    """ Beam search node class """
    def __init__(self, previous_node, char_id, logProb, length):
        """
        Args:
            previous_node (obj:`BeamSearchNode`): node in queue
            char_id (dict): character id
            logProb (float): word probability
            length (int): word length
        """
        self.prev_node = previous_node
        self.char_id = char_id
        self.logp = logProb
        self.leng = length

    def eval(self):
        """ Calculate beam search path score

        Returns:
            float: beam search path score
        """
        return self.logp / float(self.leng - 1 + 1e-6)

    def __lt__(self, other):
        """
        Args:
            self (obj:`BeamSearchNode`): beam search node
            other (obj:`BeamSearchNode`): beam search node
        """
        if self.eval() < other.eval():
            return False
        else:
            return True


def beam_decode(encoder_outputs, beam_width=5, topk=1):
    """ Beam search decode

    Args:
        encoder_outputs (Tensor): encoder outputs tensor of shape [B, T, C]
            where B is the batch size and T is the maximum length of the output sentence
        beam_width (int): beam search width
        topk (int): select top-k beam search result

    Returns:
        list(list(Tensor)): beam search decoded path
    """
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(encoder_outputs.size(0)):
        # Start with the start of the sentence token
        decoder_input = torch.tensor([[0]], device=encoder_outputs.device).long()

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  previous node, char id, logp, length
        node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put(node)
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000:
                break

            # fetch the best node
            priority_node = nodes.get()
            decoder_input = priority_node.char_id

            if priority_node.char_id.item() == 1 and priority_node.prev_node != None:
                endnodes.append(priority_node)
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(encoder_outputs[idx][priority_node.leng-1], beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[new_k].view(1, -1)
                log_p = log_prob[new_k].item()

                node = BeamSearchNode(priority_node, decoded_t, priority_node.logp + log_p, priority_node.leng + 1)
                nextnodes.append(node)

            # put them into queue
            for i in range(len(nextnodes)):
                nextnode = nextnodes[i]
                nodes.put(nextnode)
            # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for endnode in sorted(endnodes, key=lambda x: x.eval()):
            utterance = []
            utterance.append(endnode.char_id)
            # back trace
            while endnode.prev_node != None:
                endnode = endnode.prev_node
                utterance.append(endnode.char_id)

            utterance = utterance[::-1]
            utterances.append(utterance)

        stack_utterances = []
        for path_id in range(len(utterances)):
            stack_utterances.append(torch.stack(utterances[path_id], dim=-1).squeeze(0).squeeze(0))
        decoded_batch.append(stack_utterances)

    return decoded_batch
