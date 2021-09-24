"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    post_lgpma.py
# Abstract       :    Post processing of lgpma detector. Get the format html output.

# Current Version:    1.0.1
# Date           :    2021-09-23
##################################################################################################
"""

import numpy as np
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


@POSTPROCESS.register_module()
class PostLGPMA(BasePostDetector):
    """Get the format html of table
    """

    def __init__(self,
                 nms_inter=True,
                 nms_threshold=0.3
                 ):
        """
        Args:
            nms_inter(bool): whether using nms inter classes.
            nms_threshold(float): nsm threshold
        """

        super().__init__()
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
            bboxes_results = result[0]
            if self.nms_inter:
                bboxes, labels = nms_inter_classes(bboxes_results, self.nms_threshold)
                labels = [[lab[0]] for lab in labels]
            else:
                bboxes, labels = bboxes_results[0], [[0]] * len(bboxes_results[0])
                for cls in range(1, len(bboxes_results)):
                    bboxes = np.concatenate((bboxes, bboxes_results[cls]), axis=0)
                    labels += [[cls]] * len(bboxes_results[cls])
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
