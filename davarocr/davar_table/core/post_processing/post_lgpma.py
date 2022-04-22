"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    post_lgpma.py
# Abstract       :    Post processing of lgpma detector. Get the format html output.

# Current Version:    1.0.1
# Date           :    2022-03-09
# Current Version:    1.0.0
# Date           :    2021-09-23
##################################################################################################
"""

import numpy as np
from math import ceil
from networkx import Graph, find_cliques
from davarocr.davar_common.core import POSTPROCESS
from davarocr.davar_det.core.post_processing.post_detector_base import BasePostDetector
from davarocr.davar_table.core.bbox.bbox_process import nms_inter_classes, bbox2adj
from .generate_html import area_to_html, format_html


def adj_to_cell(adj, bboxes, mod):
    """Calculating start and end row / column of each cell according to row / column adjacent relationships

    Args:
        adj(np.array): (n x n). row / column adjacent relationships of non-empty aligned cells
        bboxes(np.array): (n x 4). bboxes of non-empty aligned cells
        mod(str): 'row' or 'col'

    Returns:
        list(np.array): start and end row of each cell if mod is 'row' / start and end col of each cell if mod is 'col'
    """

    assert mod in ('row', 'col')

    # generate graph of each non-empty aligned cells
    nodenum = adj.shape[0]
    edge_temp = np.where(adj != 0)
    edge = list(zip(edge_temp[0], edge_temp[1]))
    table_graph = Graph()
    table_graph.add_nodes_from(list(range(nodenum)))
    table_graph.add_edges_from(edge)

    # Find maximal clique in the graph
    clique_list = list(find_cliques(table_graph))

    # Sorting the maximal cliques
    coord = []
    times = np.zeros(nodenum)
    for clique in clique_list:
        for node in clique:
            times[node] += 1
    for ind, clique in enumerate(clique_list):
        # The nodes that only belong to this maximal clique will be selected to order,
        # unless all nodes in this maximal clique belong to multi maximal clique
        nodes_nospan = [node for node in clique if times[node] == 1]
        nodes_select = nodes_nospan if len(nodes_nospan) else clique
        coord_mean = [ind, (bboxes[nodes_select, 1] + bboxes[nodes_select, 3]).mean()] if mod == 'row' \
            else [ind, (bboxes[nodes_select, 0] + bboxes[nodes_select, 2]).mean()]
        coord.append(coord_mean)
    coord = np.array(coord, dtype='int')
    coord = coord[coord[:, 1].argsort()]  # Sorting the maximal cliques according to coordinate mean of nodes_select

    # Start and end row of each cell if mod is 'row' / start and end col of each cell if mod is 'col'
    listcell = [[] for _ in range(nodenum)]
    for ind, coo in enumerate(coord[:, 0]):
        for node in clique_list[coo]:
            listcell[node] = np.append(listcell[node], ind)

    return listcell


def softmasks_refine_bboxes(bboxes, texts_masks, soft_masks):
    """Calculating start and end row / column of each cell according to row / column adjacent relationships

    Args:
        bboxes(np.array): (n x n). row / column adjacent relationships of non-empty aligned cells
        texts_masks(np.array): (n x 4). bboxes of non-empty aligned cells
        soft_masks(str): 'row' or 'col'

    Returns:
        list(np.array): start and end row of each cell if mod is 'row' / start and end col of each cell if mod is 'col'
    """

    def n2sum(n):
        """Calculate the sum of 1 square, 2 square, ...n square.

        Args:
            n(int): end of summation sequence.

        Returns:
            int: summation result.
        """

        s = (2 * n) * (2 * n + 1) * (2 * n + 2) / 24
        return s

    def nsum(n1, n2):
        """Calculate the sum of n1, n1+1, .... n2.

        Args:
            n1(int): start of summation sequence.
            n2(int): end of summation sequence.

        Returns:
            int: summation result.
        """

        s = (n2 - n1 + 1) * (n1 + n2) / 2
        return s

    def get_matrix(xmin, xmax, ymin, ymax):
        """Calculate the solved matrix for the least squares method.

        Args:
            xmin(int): left boundary of original aligned bboxes.
            xmax(int): right boundary of original aligned bboxes.
            ymin(int): top boundary of original aligned bboxes.
            ymax(int): lower boundary of original aligned bboxes.

        Returns:
            np.matrix: (3 x 3). the solved matrix for the least squares method.
        """

        a_sum = np.matrix(np.zeros((3, 3)))
        a_sum[0, 0] = (n2sum(xmax) - n2sum(xmin - 1)) * (ymax - ymin + 1)
        a_sum[1, 1] = (n2sum(ymax) - n2sum(ymin - 1)) * (xmax - xmin + 1)
        a_sum[2, 2] = (xmax - xmin + 1) * (ymax - ymin + 1)
        a_sum[1, 0] = nsum(xmin, xmax) * nsum(ymin, ymax)
        a_sum[2, 0] = nsum(xmin, xmax) * (ymax - ymin + 1)
        a_sum[2, 1] = nsum(ymin, ymax) * (xmax - xmin + 1)
        a_sum[0, 1], a_sum[0, 2], a_sum[1, 2] = a_sum[1, 0], a_sum[2, 0], a_sum[2, 1]

        return a_sum

    def get_vector(xmin, xmax, ymin, ymax, fxy):
        """Calculate the solved vector for the least squares method.

        Args:
            xmin(int): left boundary of original aligned bboxes.
            xmax(int): right boundary of original aligned bboxes.
            ymin(int): top boundary of original aligned bboxes.
            ymax(int): lower boundary of original aligned bboxes.
            fxy(float): the value of pyramid masks in point(x, y).

        Returns:
            np.matrix: (3 x 1). the solved vector for the least squares method.
        """

        z_sum = np.matrix(np.zeros((3, 1)))
        z_sum[2] = fxy[ymin: ymax + 1, xmin: xmax + 1].sum()
        fsum_x = fxy[ymin: ymax + 1, xmin: xmax + 1].sum(axis=0)
        z_sum[0] = (np.arange(xmin, xmax + 1) * fsum_x).sum()
        fsum_y = fxy[ymin: ymax + 1, xmin: xmax + 1].sum(axis=1)
        z_sum[1] = (np.arange(ymin, ymax + 1) * fsum_y).sum()

        return z_sum

    def refine_x(xmin, xmax, ymin, ymax):
        """Refining left boundary or right boundary.

        Args:
            xmin(int): left boundary of original aligned bboxes.
            xmax(int): right boundary of original aligned bboxes.
            ymin(int): top boundary of original aligned bboxes.
            ymax(int): lower boundary of original aligned bboxes.

        Returns:
            int: the refined boundary.
        """

        a_sum = get_matrix(xmin, xmax, ymin, ymax)
        z_sum = get_vector(xmin, xmax, ymin, ymax, soft_mask[0])
        try:
            (a, b, c) = np.dot(a_sum.I, z_sum)
        except:
            return -1
        y_mean = (ymax + ymin) / 2
        x_refine = int((-1 * c / a - y_mean * b / a) + 0.5)

        return x_refine

    def refine_y(xmin, xmax, ymin, ymax):
        """Refining left boundary or right boundary.

        Args:
            xmin(int): left boundary of original aligned bboxes.
            xmax(int): right boundary of original aligned bboxes.
            ymin(int): top boundary of original aligned bboxes.
            ymax(int): lower boundary of original aligned bboxes.

        Returns:
            int: the refined boundary.
        """

        a_sum = get_matrix(xmin, xmax, ymin, ymax)
        z_sum = get_vector(xmin, xmax, ymin, ymax, soft_mask[1])
        try:
            (a, b, c) = np.dot(a_sum.I, z_sum)
        except:
            return -1
        x_mean = (xmax + xmin) / 2
        y_refine = int((-1 * c / b - x_mean * a / b) + 0.5)

        return y_refine

    cls_bboxes = [[] for _ in range(len(bboxes))]
    for cls in range(len(bboxes)):
        for text_mask, bbox, soft_mask in zip(texts_masks[cls], bboxes[cls], soft_masks[cls]):
            if text_mask.sum() <= 5:
                continue

            #  Determine the boundaries of pyramid masks and the midpoint of text regions
            bbox = [round(b, 4) for b in bbox]
            X1, Y1, X2, Y2 = ceil(bbox[0]), ceil(bbox[1]), ceil(bbox[2]) - 1, ceil(bbox[3] - 1)
            xm, ym = np.where(text_mask == 1)[1].mean(), np.where(text_mask == 1)[0].mean()

            # Refine the four boundaries respectively. Only the local pyramid masks are used in current version.
            x1_refine = refine_x(X1, int(xm), Y1, Y2)
            x2_refine = refine_x(ceil(xm), X2, Y1, Y2)
            y1_refine = refine_y(X1, X2, Y1, int(ym))
            y2_refine = refine_y(X1, X2, ceil(ym), Y2)

            # Prevent the refined bboxes from going out of image bounds in extreme cases
            height, width = text_mask.shape
            x1_refine = x1_refine if 0 <= x1_refine <= width else bbox[0]
            x2_refine = x2_refine if 0 <= x2_refine <= width else bbox[2]
            y1_refine = y1_refine if 0 <= y1_refine <= height else bbox[1]
            y2_refine = y2_refine if 0 <= y2_refine <= height else bbox[3]

            cls_bboxes[cls].append([x1_refine, y1_refine, x2_refine, y2_refine, bbox[-1]])

        cls_bboxes[cls] = np.array(cls_bboxes[cls]) if len(cls_bboxes[cls]) else np.zeros((0, 5))

    return cls_bboxes


@POSTPROCESS.register_module()
class PostLGPMA(BasePostDetector):
    """Get the format html of table
    """

    def __init__(self,
                 refine_bboxes=False,
                 nms_inter=True,
                 nms_threshold=0.3
                 ):
        """
        Args:
            refine_bboxes(bool): whether refine bboxes of aligned cells according to pyramid masks.
            nms_inter(bool): whether using nms inter classes.
            nms_threshold(float): nsm threshold
        """

        super().__init__()
        self.refine_bboxes = refine_bboxes
        self.nms_inter = nms_inter
        self.nms_threshold = nms_threshold

    def post_processing(self, batch_result, **kwargs):
        """
        Args:
            batch_result(list(Tensor)): prediction results,
                like [(box_result, seg_result, local_pyramid_masks, global_pyramid_masks), ...]
            **kwargs: other parameters

        Returns:
            list(str): Format results, like [html of table1 (str), html of table2 (str), ...]
        """

        table_results = []
        for result in batch_result:
            table_result = dict()
            # Processing bboxes of aligned cells, such as nms between all classes and bboxes refined according to lgpma
            if self.refine_bboxes:
                bboxes_results = softmasks_refine_bboxes(result[0], result[1], result[2])
            else:
                bboxes_results = result[0]
            if self.nms_inter:
                bboxes, labels = nms_inter_classes(bboxes_results, self.nms_threshold)
                labels = [[lab[0]] for lab in labels]
            else:
                bboxes, labels = bboxes_results[0], [[0]] * len(bboxes_results[0])
                for cls in range(1, len(bboxes_results)):
                    bboxes = np.concatenate((bboxes, bboxes_results[cls]), axis=0)
                    labels += [[cls]] * len(bboxes_results[cls])

            # Return empty result, if processed bboxes of aligned cells is empty.
            if not len(labels):
                table_results.append({'html': '', 'bboxes': [], 'labels': []})
                continue

            bboxes = [list(map(round, b[0:4])) for b in bboxes]
            bboxes_np = np.array(bboxes)

            # Calculating cell adjacency matrix according to bboxes of non-empty aligned cells
            adjr, adjc = bbox2adj(bboxes_np)

            # Predicting start and end row / column of each cell according to the cell adjacency matrix
            colspan = adj_to_cell(adjc, bboxes_np, 'col')
            rowspan = adj_to_cell(adjr, bboxes_np, 'row')
            cells_non = [[row.min(), col.min(), row.max(), col.max()] for col, row in zip(colspan, rowspan)]
            cells_non = np.array([list(map(int, cell)) for cell in cells_non])

            # Searching empty cells and recording them through arearec
            arearec = np.zeros([cells_non[:, 2].max() + 1, cells_non[:, 3].max() + 1])
            for cellid, rec in enumerate(cells_non):
                srow, scol, erow, ecol = rec[0], rec[1], rec[2], rec[3]
                arearec[srow:erow + 1, scol:ecol + 1] = cellid + 1
            empty_index = -1  # deal with empty cell
            for row in range(arearec.shape[0]):
                for col in range(arearec.shape[1]):
                    if arearec[row, col] == 0:
                        arearec[row, col] = empty_index
                        empty_index -= 1

            # Generate html of each table.
            texts_tokens = [[""]] * len(labels)  # The final html is available if text recognition results are used.
            html_str_rec, html_text_rec = area_to_html(arearec, labels, texts_tokens)
            table_result['html'] = format_html(html_str_rec, html_text_rec)
            table_result['bboxes'] = bboxes
            table_result['labels'] = labels
            table_results.append(table_result)

        return table_results
