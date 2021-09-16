"""
#################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_ic13.py
# Abstract       :    Script for inference

# Current Version:    1.0.0
# Date           :    2021-06-24
#################################################################################################
"""
import mmcv
import json
import sys
import codecs
from davarocr.davar_common.apis import inference_model, init_model
import cv2
import time
import json
import editdistance
import os
import re
from tqdm import tqdm
sys.path.append("../../evaluation/")
from script import default_evaluation_params, validate_data, evaluate_method
import rrc_evaluation_funcs

def find_nearest_word(lexicon, rec_str):
    match_word = ' '
    dist_min = 100
    for word in lexicon:
        ed = editdistance.eval(rec_str.upper(), word.upper())
        if ed < dist_min:
            dist_min = ed
            match_word = word
    return match_word, dist_min

def packing(out_dir, pack_dir):
    os.system('zip -r -q -j '+ pack_dir + ' ' + out_dir + '/*')


lexicon_type = 'NONE'

config_file = '../configs/mask_rcnn_r50_conv6_e2e_finetune.py'
checkpoint_file = '../log/checkpoint/mask_rcnn_r50_conv6_e2e_finetune.pth'  # Model weights

model = init_model(config_file, checkpoint_file, device='cuda:0')
cfg = model.cfg

test_dataset = '../../datalist/icdar2013_test_datalist.json'
img_prefix = '/path/to/ICDAR2013-Focused-Scene-Text/'

assert  lexicon_type in ['STRONG', 'WEAK', 'GENERAL', 'NONE']
lexicon_root = "../../evaluation/lexicons/icdar2013/"
strong_lexicon_path = lexicon_root + "strong_lexicon/"
weak_lexicon = []
general_lexicon = []
if lexicon_type == "WEAK":
    weak_lexicon_file = codecs.open(lexicon_root + "ch2_test_vocabulary.txt", encoding="utf-8")
    for line in weak_lexicon_file:
        if line.strip() != "":
            weak_lexicon.append(line.strip())

elif lexicon_type == "GENERAL":
    general_lexicon_file = codecs.open(lexicon_root + "GenericVocabulary.txt", encoding="utf-8")
    for line in general_lexicon_file:
        if line.strip() != "":
            general_lexicon.append(line.strip())

with open(test_dataset) as load_f:
    test_file = json.load(load_f, encoding="utf-8" )


work_dir = "../eval_result/ICDAR2013/"
out_dir = work_dir + lexicon_type + "/"
pack_dir = work_dir + '/ICDAR2013_' + lexicon_type + '.zip'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
"""
    If the lexicon type is None, the evaluation result will be generated, 
    otherwise it will be corrected according to the vocabulary
"""
for filename, _ in tqdm(test_file.items()):
    # Load images
    img_path = img_prefix + filename
    img_name = img_path.split("/")[-1]

    # Inference
    if lexicon_type == "NONE":
        result = inference_model(model, img_path)[0]
        final_text_results = result['texts']
        final_box_results = result['points']
    else:
        txt = open(work_dir + "/NONE/" + filename.split("/")[-1].replace(".jpg",".txt"), "r")
        final_text_results = []
        final_box_results = []
        for line in txt.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            points = list(map(int, line.split(',')[:-1]))
            transcription = line.split(',')[-1]
            final_text_results.append(transcription)
            final_box_results.append(points)
        txt.close()

    # ========== Rectify text using lexicons ==========
    lexicon = []
    if lexicon_type == "STRONG":
        lexicon_name = "voc_" + filename.split("/")[-1].replace(".jpg",".txt")
        lexicon_file = codecs.open(strong_lexicon_path + lexicon_name, encoding="utf-8")
        for line in lexicon_file:
            if line.strip() != "":
                lexicon.append(line.strip())
    elif lexicon_type == "WEAK":
        lexicon = weak_lexicon
    elif lexicon_type == "GENERAL":
        lexicon = general_lexicon

    if lexicon != []:
        for i, text in enumerate(final_text_results):
            # Filter text strings that are less than 3 in length or contain numbers
            partten = re.compile('[0-9+]')
            if len(text) < 3 or partten.findall(text):
                continue
            new_text, dist = find_nearest_word(lexicon, text)
            final_text_results[i] = new_text

    # ============== Write Results ================
    txt = open(out_dir + filename.split("/")[-1].replace(".jpg",".txt"), "w", encoding="utf-8")
    for j, box in enumerate(final_box_results):
        for point in box:
            txt.write(str(int(point))+",")
        txt.write(final_text_results[j].replace(",", "")+"\n")
    txt.close()
packing(out_dir, pack_dir)
sys.argv.append('-s=' + pack_dir)
rrc_evaluation_funcs.main_evaluation(None, default_evaluation_params, validate_data, evaluate_method)
