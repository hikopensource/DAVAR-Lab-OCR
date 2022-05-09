"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ner_metric.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
from collections import Counter
from seqeval.scheme import Tokens, IOBES


def gt_label2entity(gt_infos):
    """ Get all entities from ground truth infos.

    Args:
        gt_infos (list[dict]): Ground-truth infomation contains text and label, eg:[{'token_labels':['B-', 'I-']}]

    Returns:
        list[list]: Original labeled entities in groundtruth.
                    [[category,start_position,end_position]]
    """
    gt_entities = []
    for gt_info in gt_infos:
        label = gt_info['token_labels']
        if label and isinstance(label[0],(tuple,list)):
            #assert len(label[0]) == 3
            entities = label
        else:
            entities = [[t.to_tuple()[1], t.to_tuple()[2], t.to_tuple()[3]] for t in Tokens(label, IOBES).entities]
        gt_entities.append(entities)
    return gt_entities


def _compute_f1(origin, found, right):
    """Calculate recall, precision, f1-score.

    Args:
        origin (int): Original entities in groundtruth.
        found (int): Predicted entities from model.
        right (int): Predicted entities that
                        can match to the original annotation.
    Returns:
        float: Metric of recall.
    Returns:
        float: Metric of precision.
    Returns:
        float: Metric of f1-score.
    """
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (
        precision + recall)
    return recall, precision, f1


def compute_f1_all(pred_entities, gt_entities):
    """Calculate precision, recall and F1-score for all categories.

    Args:
        pred_entities: The predicted entities from model.
        gt_entities: The entities of ground truth file.

    Returns:
        dict: precision,recall, f1-score in total
                        and each categories.
    """
    origins = []
    founds = []
    rights = []
    for i, _ in enumerate(pred_entities):
        origins.extend(gt_entities[i])
        founds.extend(pred_entities[i])
        rights.extend([
            pre_entity for pre_entity in pred_entities[i]
            if pre_entity in gt_entities[i]
        ])

    class_info = {}
    origin_counter = Counter([x[0] for x in origins])
    found_counter = Counter([x[0] for x in founds])
    right_counter = Counter([x[0] for x in rights])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = _compute_f1(origin, found, right)
        class_info[type_] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    recall, precision, f1 = _compute_f1(origin, found, right)
    class_info['all'] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }
    return class_info


def eval_ner_f1(results, gt_infos):
    """Evaluate for ner task.

    Args:
        results (list): Predict results of entities.
        gt_infos (list[dict]): Ground-truth infomation which contains
                            text and label.
    Returns:
        dict: precision,recall, f1-score of total
                            and each catogory.
    """
    assert len(results) == len(gt_infos)
    gt_entities = gt_label2entity(gt_infos)
    pred_entities = []
    for i in range(len(gt_infos)):
        line_entities = []
        for result in results[i]:
            line_entities.append(result)
        pred_entities.append(line_entities)
    assert len(pred_entities) == len(gt_entities)
    class_info = compute_f1_all(pred_entities, gt_entities)
    class_info['hmean'] = class_info['all']['f1-score']
    return class_info
