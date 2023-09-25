"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tree_f1_score.py
# Abstract       :    Table understanding evaluation metrics.

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""
import re
import numpy as np
from collections import deque
from apted import APTED
from apted.helpers import Tree


def evaluate_tree_f1(preds, gts, eval_type='hard'):
    """Evaluate the prediction with similarity.

    Args:
        preds (list(list(list(int)))): ground-truth of relations. like [list(N_1 x N_1), ..., list(N_C x N_C)].
            where C is the number of tables and N_i is the cells number of i-th table.
        gts (list(list(list(int)))): predicted relations. like [list(N_1 x N_1), ..., list(N_C x N_C)].
            where C is the number of tables and N_i is the cells number of i-th table.
        eval_type (str): evaluation type. Optional values are 'hard' and 'soft'.

    Returns:
        float: tree based recall of basic information trees.
    Returns:
        float: tree based precision of basic information trees.
    Returns:
        float: tree based f1-score of basic information trees.
    """
    recall, precision, f1 = 0, 0, 0
    num_samples = 0
    for pred, gt in zip(preds, gts):
        pred_tree = ajacent_to_tree(pred)
        gt_tree = ajacent_to_tree(gt)
        r_cur, p_cur, f1_cur = cal_tree_f1_score(gt_tree, pred_tree, eval_type=eval_type)
        recall += r_cur
        precision += p_cur
        f1 += f1_cur
        num_samples += 1

    recall /= num_samples
    precision /= num_samples
    f1 /= num_samples
    print('precition = {}, recall = {}, f1 = {}'.format(precision, recall, f1))

    return recall, precision, f1


def ajacent_to_tree(ajacency, leftlabel=2, rightlabel=1):
    """Convert ajacent matrix to tree.

    Args:
        ajacency (list(list(int): ajacent matrix of a table.
        leftlabel (int): the corresponding categories in the adjacency matrix represent the left edge
        rightlabel (int): the corresponding categories in the adjacency matrix represent the right edge

    Returns:
        list(dict): A list of tree. Each tree describing a table items in table.
    """
    trees = []

    # Find the root node from ajacenct matrix. (in-degree == 0, out-degree > 0)
    ajacency_np = np.array(ajacency)
    root_nodes = [idx for idx in range(len(ajacency)) if ajacency_np[:, idx].sum() == 0 and ajacency_np[idx, :].sum()]

    for root in root_nodes:
        top, left = [], []
        for node, edge in enumerate(ajacency[root]):
            if edge == leftlabel:
                left.append(node)
            elif edge == rightlabel:
                top.append(node)
            else:
                continue
        top_child_list = [get_child_tree(ajacency, node) for node in top]
        left_child_list = [get_child_tree(ajacency, node) for node in left]

        tree_dict = {'idx': root, 'top': top_child_list, 'left': left_child_list}
        trees.append(tree_dict)

    return trees


def get_child_tree(ajacency, node):
    """Generate left tree and top tree for a node.

    Args:
        ajacency (list(list(int): ajacent matrix of a table..
        node (int): node number in table.
    Returns:
        dict: left tree or top tree for a node.
    """
    tree_dict = {'idx': node}
    que = deque()
    que.append(tree_dict)
    vis = [False for _ in range(len(ajacency))]
    while len(que) != 0:
        q_len = len(que)
        for i in range(q_len):
            u_dict = que.pop()
            u = u_dict['idx']
            vis[u] = True
            u_list = []
            for v in range(len(ajacency[u])):
                if vis[v]:
                    continue
                if ajacency[u][v] == 0:
                    continue
                v_dict = {'idx': v}
                que.append(v_dict)
                u_list.append(v_dict)
            u_dict['children'] = u_list

    return tree_dict


def cal_tree_f1_score(gt, pred, eval_type='hard'):
    """ Calculate tree-base precision, recall and f1 score.

    Args:
        gt (list(dict)）: A list of tree. Each tree describing a table items in table.
        pred (list(dict)）: A list of tree. Each tree describing a table items in table.
        eval_type (str): evaluation type. Optional values are 'hard' and 'soft'.

    Returns:
        float: tree based recall of a table.
    Returns:
        float: tree based precision of a table.
    Returns:
        float: tree based f1-score of a table.
    """
    if len(pred) == 0 and len(gt):
        return 1, 0, 0
    elif len(pred) and len(gt) == 0:
        return 0, 1, 0

    if eval_type == 'hard':
        gt_check = [1 if per_gt in pred else 0 for per_gt in gt]
        r = sum(gt_check) / len(gt_check)
        pred_check = [1 if per_pred in gt else 0 for per_pred in pred]
        p = sum(pred_check) / len(pred_check)
    elif eval_type == 'soft':
        r = cal_tree_recall(gt, pred)
        p = cal_tree_precision(gt, pred)
    else:
        raise ValueError('eval_type must be hard or soft...')

    f1 = 2 * r * p / (r + p) if p and r else 0

    return r, p, f1


def cal_tree_recall(gt, pred):
    """ Calculate soft tree-base recall.

    Args:
        gt (list(dict)）: A list of tree. Each tree describing a table items in table.
        pred (list(dict)）: A list of tree. Each tree describing a table items in table.
    Returns:
        float: tree based recall of a table.
    """
    total_scores = 0.
    for y_true in gt:
        idx = y_true['idx']
        pred_match_check = [1 if per_pred['idx'] == idx else 0 for per_pred in pred]
        assert sum(pred_match_check) <= 1
        if sum(pred_match_check) == 1:
            matched_pred = pred[pred_match_check.index(1)]
            teds_score = compute_teds_score(y_true, matched_pred)
        else:
            teds_score = 0.
        total_scores += teds_score

    recall = total_scores / len(gt)

    return recall


def cal_tree_precision(gt, pred):
    """ Calculate soft tree-base precision.

    Args:
        gt (list(dict)）: A list of tree. Each tree describing a table items in table.
        pred (list(dict)）: A list of tree. Each tree describing a table items in table.
    Returns:
        float: tree based precision of a table.
    """
    total_scores = 0.
    for y_pred in pred:
        idx = y_pred['idx']
        gt_match_check = [1 if per_gt['idx'] == idx else 0 for per_gt in gt]
        assert sum(gt_match_check) <= 1
        if sum(gt_match_check) == 1:
            matched_gt = gt[gt_match_check.index(1)]
            teds_score = compute_teds_score(matched_gt, y_pred)
        else:
            teds_score = 0.
        total_scores += teds_score

    precision = total_scores / len(pred)

    return precision


def compute_teds_score(tree_gt, tree_pred):
    """Calculate the teds-score between two trees.
    Args:
        tree_gt (string）: Ground-truth tree.
        tree_pred (string）: Predicted tree.
    Returns:
        float: teds-score between two trees.
    """
    tree_gt = cvt_tree2str(tree_gt)
    tree_pred = cvt_tree2str(tree_pred)

    # apted
    len_gt = len(re.split('[\{\}*]+', tree_gt)[1:-1])
    tree_gt = Tree.from_text(tree_gt)
    tree_pred = Tree.from_text(tree_pred)
    distance = APTED(tree_gt, tree_pred).compute_edit_distance()

    teds = 1 - distance / (len_gt - 1)  # The root node must be the same. Remove it.
    teds = teds if teds > 0 else 0

    return teds


def cvt_tree2str(tree):
    """Convert tree to string.
    Args:
        tree (dict）: Tree stored as a dict.
    Returns:
        string: Tree stored as a string.
    """

    def dfs(tree_ori):
        """
        Args:
            tree_ori (dict）: sub-tree stored as a dict.
        Returns:
            string: sub-tree stored as a string.
        """
        tree_string = '{' + str(tree_ori['idx'])
        for child_node in tree_ori['children']:
            tree_string += dfs(child_node)
        tree_string += '}'

        return tree_string

    root = tree['idx']
    tree_str = '{' + str(root)  # start

    for tt_d in tree['top']:
        tree_str += '{top' + str(tt_d['idx'])
        for child in tt_d['children']:
            tree_str += dfs(child)
        tree_str += '}'

    for ll_d in tree['left']:
        tree_str += '{left' + str(ll_d['idx'])
        for child in ll_d['children']:
            tree_str += dfs(child)
        tree_str += '}'

    tree_str += '}'  # end

    return tree_str
