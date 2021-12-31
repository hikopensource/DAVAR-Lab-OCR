import os
import json
import cv2
import copy
import glob
import multiprocessing
from tqdm import tqdm
import numpy as np

from docbank_loader import DocBankLoader, DocBankConverter

# txt_dir = 'demo/txt'
# img_dir = 'demo/img'

txt_dir = 'DocBank_500K_txt'
img_dir = 'DocBank_500K_ori_img'

json_out_dir = 'DocBank_500K_json'
loader = DocBankLoader(txt_dir=txt_dir, img_dir=img_dir)
converter = DocBankConverter(loader)

examples = glob.glob(os.path.join(txt_dir, '*.txt'))
examples = [os.path.basename(per) for per in examples]

def worker(example):
	example = loader.get_by_filename(example)

	# filter not processed file.
	save_name = os.path.join(json_out_dir, os.path.basename(example.filepath).replace('.jpg', '.json'))
	if not os.path.exists(save_name):
		print(save_name)

		formatted_json = {}
		formatted_json['height'] = example.pagesize[1]
		formatted_json['width'] = example.pagesize[0]

		content_ann = {}
		content_ann2 = {}

		## token level
		bboxes = example.denormalized_bboxes()
		filepath = example.filepath  # The image filepath
		pagesize = example.pagesize  # The image size
		words = example.words  # The tokens
		# bboxes = example.bboxes  # The normalized bboxes
		rgbs = example.rgbs  # The RGB values
		fontnames = example.fontnames  # The fontnames
		structures = example.structures  # The structure labels

		labels_list = [[per] for per in structures]
		attributes_list = [[font, rgb[0], rgb[1], rgb[2]] for font, rgb in zip(fontnames, rgbs)]

		content_ann['bboxes'] = bboxes
		content_ann['texts'] = words
		content_ann['labels'] = labels_list
		content_ann['attributes'] = attributes_list
		content_ann['cares'] = [1]*len(attributes_list)

		# layout level
		new_filepath = os.path.basename(filepath.replace(txt_dir, img_dir))
		layout_examples = converter.get_by_filename(new_filepath)
		layout_bboxes = layout_examples.print_bbox().split('\n')
		layout_bboxes = [per.split('\t') for per in layout_bboxes]
		layout_bboxes_list = []
		layout_labels_list = []

		for per_bbox in layout_bboxes:
			layout_bboxes_list.append([int(per_bbox[0]), int(per_bbox[1]), int(per_bbox[2]), int(per_bbox[3])])
			layout_labels_list.append([per_bbox[4]])

		content_ann2['bboxes'] = layout_bboxes_list
		content_ann2['labels'] = layout_labels_list
		content_ann2['cares'] = [1]*len(layout_bboxes_list)

		formatted_json['content_ann'] = content_ann
		formatted_json['content_ann2'] = content_ann2

		# json output
		save_name = os.path.join(json_out_dir, os.path.basename(filepath).replace('.jpg', '.json'))
		if not os.path.exists(os.path.dirname(save_name)):
			os.makedirs(os.path.dirname(save_name))
		with open(save_name, 'w', encoding='utf8') as wf:
			json.dump(formatted_json, wf)

		# visualize
		if 0:
			color_map = {
				'paragraph': (255, 0, 0),
				'section': (0, 255, 0),
				'list': (0, 0, 255),
				'abstract': (0, 255, 255),
				'author': (255, 0, 255),
				'equation': (255, 255, 0),
				'figure': (128, 0, 0),
				'table': (0, 128, 0),
				'title': (0, 0, 128),
			}
			img = cv2.imread(filepath)
			layout_img = copy.deepcopy(img)
			bboxes = content_ann['bboxes']
			labels = content_ann['labels']

			for idx, per_bbox in enumerate(bboxes):
				color = color_map[labels[idx][0]] if labels[idx][0] in color_map else (0, 0, 0)
				cv2.rectangle(img, (per_bbox[0], per_bbox[1]), (per_bbox[2], per_bbox[3]), color)

			layout_bboxes = content_ann2['bboxes']
			layout_labels = content_ann2['labels']
			for idx, per_bbox in enumerate(layout_bboxes):
				color = color_map[layout_labels[idx][0]] if layout_labels[idx][0] in color_map else (0, 0, 0)
				cv2.rectangle(layout_img, (int(per_bbox[0]), int(per_bbox[1])), (int(per_bbox[2]), int(per_bbox[3])),
				              color)

			cv2.imwrite(os.path.basename(filepath), np.concatenate((img, layout_img), 1))

# ## single process
# for example in tqdm(examples):
# 	worker(example)

## multiple processes
pool = multiprocessing.Pool(processes=50)
for example in tqdm(examples):
	pool.apply_async(worker, (example,))
pool.close()
pool.join()