"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    visualization.py
# Abstract       :    visualize cell relation according to adjacency matrix

# Current Version:    1.0.0
# Date           :    2022-11-22
######################################################################################################
"""
import json
import collections
import os.path as osp

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def draw_relation_edges(img_prefix, ann_path, out_dir):
    """ main function of plotting cell relation edges

        Args:
            img_prefix(string): prefix folder of images to visualize
            ann_path(string): corresponding annotation file path
            out_dir(string): directory to store visualization results
        Returns:
            None
    """
    with open(ann_path, 'r', encoding='utf-8') as fp:
        ann_file = json.load(fp, object_pairs_hook=collections.OrderedDict)

    # node labels to color map
    color_map = {0: 'lightcoral',
                 1: 'yellowgreen',
                 2: 'deepskyblue',
                 3: 'black',
                 4: 'paleturquoise'}

    for idx, img_name in enumerate(ann_file):
        img_path = osp.join(img_prefix, img_name)
        img = mpimg.imread(img_path)
        img_h, img_w = img.shape[0], img.shape[1]

        plt.figure(figsize=(img_w / 100, img_h / 100), dpi=200)
        plt.axis('off')

        bboxes_whole = ann_file[img_name]['content_ann']['bboxes']  # cell bboxes
        indexs_whole = ann_file[img_name]['content_ann'].get('indexs', None)
        labels_whole = ann_file[img_name]['content_ann']['labels']  # cell labels
        adj_matrixes = ann_file[img_name]['content_ann'].get('relations', None)  # cell relation matrix
        if not adj_matrixes:
            continue

        if 'content_ann2' in ann_file[img_name].keys():
            # for multi-table images circumstance
            table_bboxes = ann_file[img_name]['content_ann2']['bboxes']
        else:
            # for cropped table image, there should be only one table bbox
            b = np.array(bboxes_whole)
            table_bboxes = [[int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 4].max()), int(b[:, 5].max())]]
        # table_num = len(table_bboxes)
        for table_id, table_bbox in enumerate(table_bboxes):
            if indexs_whole:
                bboxes = [per_bbox for x, per_bbox in enumerate(bboxes_whole) if indexs_whole[x] == [table_id]]
                bboxes = [[b[0], b[1], b[4], b[5]] for b in bboxes]
                labels = [per_label for x, per_label in enumerate(labels_whole) if indexs_whole[x] == [table_id]]
            else:
                bboxes = [[b[0], b[1], b[4], b[5]] for b in bboxes_whole]
                labels = labels_whole
            num_nodes = len(bboxes)
            if num_nodes == 0:
                continue

            pos = {}
            node_colors = []
            cell_heights = []
            for i in range(num_nodes):
                # coordinate of cell nodes
                node_x = (bboxes[i][0] + bboxes[i][2]) // 2
                node_y = (bboxes[i][1] + bboxes[i][3]) // 2
                node_colors.append(color_map[labels[i][0]])  # label-corresponding node colors
                pos[i] = (node_x, node_y)
                if labels[i][0] != 4:
                    cell_heights.append(bboxes[i][3] - bboxes[i][1])
            node_size = max(min(cell_heights) // 6, 2)

            # construct networkx graph
            relation_graph = nx.DiGraph()
            # add nodes to graph
            relation_graph.add_nodes_from(list(pos.keys()))
            adj_matrix = adj_matrixes[table_id] if indexs_whole else adj_matrixes
            if not adj_matrix:
                continue
            assert len(adj_matrix) == num_nodes

            # add edges to graph
            edge_lists = []
            edge_colors = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj_matrix[i][j] == 1 or adj_matrix[i][j] == 2:
                        color = 'r' if adj_matrix[i][j] == 1 else 'g'
                        edge_lists.append((i, j))
                        edge_colors.append(color)
            relation_graph.add_edges_from(edge_lists)

            plt.imshow(img)
            nx.draw(relation_graph,
                    pos=pos,
                    node_size=node_size,
                    width=0.5,
                    arrowstyle='->',
                    arrowsize=max(int(node_size * 0.5), 3),
                    connectionstyle='arc3,rad=0.4',
                    node_color=node_colors,
                    edge_color=edge_colors
                    )
        plt.savefig(osp.join(out_dir, 'vis_' + '_'.join(img_name.split('/'))),
                    bbox_inches='tight', pad_inches=0.01)
        plt.close()

    return


if __name__ == '__main__':
    # prefix folder of images to visualize
    img_prefix = 'path/to/img/prefix'
    # corresponding annotation file path
    ann_file_path = 'path/to/ann/file'
    # directory to store visualization results
    vis_dir = 'path/to/vis/dir'

    draw_relation_edges(img_prefix, ann_file_path, vis_dir)
