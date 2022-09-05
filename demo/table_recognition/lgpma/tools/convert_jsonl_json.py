import os
import cv2
import json
import jsonlines
from davarocr.davar_table.utils.format import format_html

jsonl_path = r'path/to/PubTabNet_2.0.0_val.jsonl'
json_path = r'path/to/save/PubTabNet_2.0.0_val.json'
image_root = r'path/to/PubTabNet'

with jsonlines.open(jsonl_path, 'r') as fp:
    test_file = list(fp)

gt_json = dict()
for data in test_file:
    str_true = data['html']['structure']['tokens']
    imgname = 'Images/val/' + data['filename']
    image = cv2.imread(os.path.join(image_root, imgname))
    h, w = image.shape[0], image.shape[1]
    gt_json[imgname] = {'content_ann': {'html': format_html(data)}, 'height': h, 'width': w}

with open(json_path, "w", encoding='utf-8') as writer:
    json.dump(gt_json, writer, ensure_ascii=False)
