"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_utils.py
# Abstract       :    Implementations of the recognition test utils

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import os
import json
import codecs
from decimal import Decimal

import numpy as np

from PIL import Image, ImageDraw, ImageFont
from prettytable import PrettyTable

import mmcv


# Visualization Color Match Chart
colors = {
    'gt_lost': (255, 255, 0),   # GT  | lost          | Yellow
    'res_lost': (255, 102, 0),  # RES | lost          | Orange
    'gt_ok': (0, 255, 0),       # GT  | match success | Green
    'res_ok': (0, 255, 255),    # RES | match success | Cyan
    'gt_fail': (255, 0, 0),     # GT  | match fail    | Red
    'res_fail': (255, 0, 255)   # RES | match fail    | Megenta
}

font_path = 'Deng.ttf'

# =============================== file operator ================================


def make_paths(*args):
    """
    make a new directory
    Args:
        *args (str): back parameter

    Returns:

    """
    for para in args:
        if not os.path.exists(para):
            # make the new directory
            os.makedirs(para)


def check_path(file_out):
    """

     Args:
         file_out (str): check the file exists

     Returns:

    """
    (filepath, _) = os.path.split(file_out)

    # check the filepath
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def write_row(worksheet, row_list,
              row_index, col_index=0):
    """
    Args:
        worksheet (worksheet): record the xlwt worksheet
        row_list (list): max row number
        row_index (int): row interval
        col_index (int): record col number

    Returns:

    """
    for interval, _ in enumerate(row_list):
        worksheet.write(row_index, interval + col_index, row_list[interval])


# ====================================== test tools ===================================


def edit_distance(word1, word2):
    """
    Args:
        word1 (str): string1
        word2 (str): string2

    Returns:
        int: edit distance between string1 and string2

    """
    # calculate the edit distance
    sentence1 = list(word1)
    sentence2 = list(word2)
    row = len(sentence1) + 1
    col = len(sentence2) + 1
    matrix = np.zeros((row, col), dtype=int)
    for i in range(row):  # initialization
        matrix[i][0] = i
    for i in range(col):
        matrix[0][i] = i

    for i in range(1, row):  # dynamic programming
        for j in range(1, col):
            if sentence1[i-1] == sentence2[j-1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = matrix[i - 1][j - 1] + 1
                matrix[i][j] = min(matrix[i][j],
                                   min(matrix[i - 1][j] + 1,
                                       matrix[i][j - 1] + 1))

    return int(matrix[row - 1][col - 1])


def pad_resize(img, img_size=(512, 32),
               bg_color=(255, 255, 255)):
    """
    Args:
        img (Torch.Tensor): image
        img_size (tuple): resize size
        bg_color (tuple): background color

    Returns:
        np.array: resized image
    Returns:
        list: source image coordinate

    """
    x_size, y_size = img.size
    width, height = x_size, y_size
    pad_ratio = img_size[0]/img_size[1]

    # Create the new image
    re_img = Image.new('RGB', (int(max(x_size, y_size * pad_ratio)),
                               int(max(x_size / pad_ratio, y_size))), bg_color)
    if x_size <= y_size * pad_ratio:
        x_size = int((y_size * pad_ratio - x_size) / 2)
        y_size = 0
        resize_ratio = 32.0 / height
    else:
        y_size = int((x_size / pad_ratio - y_size) / 2)
        x_size = 0
        resize_ratio = 512.0 / width
    tp_box = [x_size, y_size, x_size + width, y_size+height]
    tp_box = [int(x * resize_ratio) for x in tp_box]

    re_img.paste(img, (x_size, y_size))
    re_img = re_img.resize(img_size, Image.BICUBIC)
    return re_img, tp_box


def compare_json(file_pre, compare_path, compare_list, res_json_list,
                 file_type='Tight', img_size=(512, 32)):
    """

    Args:
        file_pre (str): json file path
        compare_path (str): compare json file path
        compare_list (list): compare parameter
        res_json_list (list): compare the prediction file list
        file_type (str): compare file type
        img_size (tuple): image size

    Returns:

    """
    width, height = img_size
    print(compare_list)
    print(res_json_list)

    name1, epoch1, nick1 = compare_list[0]
    name2, epoch2, nick2 = compare_list[1]
    name1 = nick1 if nick1 != '' else name1
    name2 = nick2 if nick2 != '' else name2
    fosx_path = compare_path + f'{name1}_e{epoch1}_O_{name2}_e{epoch2}_X/'
    fxso_path = compare_path + f'{name1}_e{epoch1}_X_{name2}_e{epoch2}_O/'
    fxsx_path = compare_path + f'{name1}_e{epoch1}_X_{name2}_e{epoch2}_X/'
    make_paths(fosx_path, fxso_path, fxsx_path)

    json_list = list()
    for res_json in res_json_list:
        with codecs.open(res_json, 'r', 'utf-8') as res_in:
            input_dict = json.load(res_in)
        json_list.append(input_dict)

    dict1 = json_list[0]
    dict2 = json_list[1]

    font_ = ImageFont.truetype(font_path, 20)

    index_count = 0
    if file_type in ["Tight", "File"]:
        for k, value1 in dict1.items():
            print(f'{k}_compare_start')  # image_name
            pil_img = Image.open(os.path.join(file_pre, k))
            img_name = k.split('/')[-1]
            value2 = dict2[k]
            for index, res1 in enumerate(value1):
                res2 = value2[index]
                assert res2['bbox'] == res1['bbox'] and res2['gt_text'] == res1['gt_text']
                vlist = res1['bbox']
                label = res1['gt_text']
                result1 = res1['res_text']  # .replace('SPACE',' ')
                result2 = res2['res_text']  # .replace('SPACE',' ')
                editdis1 = edit_distance(result1.replace(' ', ''),
                                         label.replace(' ', ''))
                editdis2 = edit_distance(result2.replace(' ', ''),
                                         label.replace(' ', ''))
                before1 = res1['pred_before'] if 'pred_before' in res1 else ''
                before2 = res2['pred_before'] if 'pred_before' in res2 else ''

                text1 = f'{name1}_e{epoch1}: ' + result1.replace(' ', u'◇') + f' ({editdis1})'
                text2 = f'{name2}_e{epoch2}: ' + result2.replace(' ', u'◇') + f' ({editdis2})'
                max_w = max(width, max(font_.getsize(text1)[0], font_.getsize(text2)[0]))

                min_x, max_x = min(vlist[::2]), max(vlist[::2])
                min_y, max_y = min(vlist[1::2]), max(vlist[1::2])
                crop_img = pil_img.crop([min_x, min_y, max_x, max_y])
                pad_img, _ = pad_resize(crop_img, img_size=img_size)

                if before1 == '' and before2 == '':
                    vis_img = Image.new('RGB', (max_w, height+32*3), (255, 255, 255))
                else:
                    vis_img = Image.new('RGB', (max_w, height+32*5), (255, 255, 255))

                draw = ImageDraw.Draw(vis_img)
                vis_img.paste(pad_img, (0, 0))
                cur_h = height
                for before in [before1, before2]:
                    if before != '':
                        size = 512 // len(before)
                        sizef = 512.0 / len(before)
                        # tw,th = font_.getsize(before) #512 ÷8-> 64
                        smallfont_ = ImageFont.truetype(font_path, size)
                        timg = Image.new('RGB', (width, height), (255, 255, 255))

                        tdraw = ImageDraw.Draw(timg)
                        cur_w = 0
                        for char in before:
                            tdraw.text((int(sizef * cur_w), 0), char, fill=(0, 0, 0), font=smallfont_)
                            cur_w += 1

                        vis_img.paste(timg, (0, cur_h))
                        cur_h += height

                if editdis1 == 0 and editdis2 > 0:
                    res_color1 = colors['res_ok']
                    res_color2 = colors['res_fail']
                    save_path = fosx_path
                elif editdis1 > 0 and editdis2 == 0:
                    res_color1 = colors['res_fail']
                    res_color2 = colors['res_ok']
                    save_path = fxso_path
                elif editdis1 > 0 and editdis2 > 0:
                    res_color1 = colors['res_fail']
                    res_color2 = colors['res_fail']
                    save_path = fxsx_path
                else:
                    continue
                draw.text((0, cur_h), 'GT: ' + label.replace(' ', u'◇'), fill=(0, 0, 0), font=font_)
                draw.text((0, cur_h+32), text1, fill=res_color1, font=font_)
                draw.text((0, cur_h+64), text2, fill=res_color2, font=font_)
                save_name = save_path + f'{index_count}_{img_name}'
                vis_img.save(save_name)
                index_count += 1
    else:
        raise Exception("Not supported data type !!!")


def vis_json(file_pre, jin,
             vis_path, settings):
    """
    Args:
        file_pre (str): prefix of the image path
        jin (str): save path of the prediction json
        vis_path (str): save path of the visualization result
        settings (list): visualization setting

    Returns:

    """

    count_list = []
    unclear_path = vis_path + '#' + '/'
    save_path = ""
    make_paths(unclear_path)

    for setting in settings:
        out_path = vis_path + '_'.join(list(map(str, setting))) + '/'
        make_paths(out_path)
        if len(setting) == 2:
            count_list.append([0, 1])
        else:
            count_list.append([0, setting[2]])

    with codecs.open(jin, 'r', 'utf8') as res_in:
        input_dict = json.load(res_in)

    print('Vis...')
    for k, value in input_dict.items():
        max_editdis = 0
        img_name = k.split('/')[-1]
        need_save = False
        for index, res in enumerate(value):
            label = res['gt_text']
            result = res['res_text']
            editdis = edit_distance(result.replace(' ', ''), label.replace(' ', ''))
            if '#' in label or len(label) > 30:
                pass
            elif editdis == 0:
                pass
            else:
                max_editdis = max(max_editdis, editdis)

        for i, v_t in enumerate(settings):
            if v_t[0] <= max_editdis <= v_t[1]:
                need_save = True
                if count_list[i][1] == 1 or (count_list[i][0] % count_list[i][1] == 0):
                    save_path = vis_path + '_'.join(
                        list(map(str, v_t))) + '/' + str(
                        count_list[i][0]) + '_' + img_name
                count_list[i][0] += 1
        if not need_save:
            continue

        pil_img = Image.open(os.path.join(file_pre, k)).convert('RGB')
        width, height = pil_img.size
        if width // height > 2:
            vis_img = Image.new('RGB', (width, height * 2), (0, 0, 0))
        else:
            vis_img = Image.new('RGB', (max(width, 100), height), (0, 0, 0))

        _, v_h = vis_img.size
        draw1 = ImageDraw.Draw(pil_img)
        draw2 = ImageDraw.Draw(vis_img)
        box_num = len(value)
        size_uplimit = min(40, v_h // (box_num * 2))

        cur_y = 0
        for index, res in enumerate(value):
            label = res['gt_text']
            result = res['res_text']
            editdis = edit_distance(result.replace(' ', ''),
                                    label.replace(' ', ''))
            vlist = res['bbox']
            if '#' in label or len(label) > 30:
                gt_color = colors['gt_lost']
                res_color = colors['res_lost']
            elif editdis == 0:
                gt_color = colors['gt_ok']
                res_color = colors['res_ok']
            else:
                gt_color = colors['gt_fail']
                res_color = colors['res_fail']
            draw1.line(vlist, fill=gt_color, width=2)
            font_ = ImageFont.truetype(font_path, 30)
            draw1.text(vlist[:2], str(index), fill=gt_color, font=font_)

            res, ground_truth = result, label

            gt_text = str(index) + ': ' + ground_truth.replace(' ', u'◇')
            res_text = str(index) + ': ' + res.replace(' ', u'◇') + ' (' + str(editdis) + ')'
            imgfont = ImageFont.truetype(font_path, size_uplimit)

            g_w, _ = imgfont.getsize(gt_text)
            r_w, _ = imgfont.getsize(res_text)
            f_w = max(g_w, r_w)
            if f_w > width:
                new_size = int(width * 1.0 / f_w * size_uplimit)
                imgfont = ImageFont.truetype(font_path, new_size)
            draw2.text((0, cur_y), gt_text, fill=gt_color, font=imgfont)
            cur_y += imgfont.getsize(gt_text)[1]
            draw2.text((0, cur_y), res_text, fill=res_color, font=imgfont)
            cur_y += imgfont.getsize(res_text)[1]

        if width // height > 2:
            save_img = Image.fromarray(
                np.concatenate([np.array(pil_img), np.array(vis_img)], axis=0))
        else:
            save_img = Image.fromarray(
                np.concatenate([np.array(pil_img), np.array(vis_img)], axis=1))

        save_img.save(save_path)


def eval_json(jin, jout, batch_max_length=30):
    """

    Args:
        jin (str): save path of the prediction json
        jout (str): save path of the evaluate result
        batch_max_length (int): the max length of the recognition

    Returns:
        dict: recognition accuracy or character counting accuracy
    """

    with codecs.open(jin, 'r', 'utf-8') as res_in:
        input_dict = json.load(res_in)

    # initialization parameter
    all_ed = dict()
    all_ed['TOTAL_SAMPLE_NUM'] = 0
    all_ed['NOT_CARE_NUM'] = 0
    all_ed['VALID_NUM'] = 0
    all_ed['LABEL_LEN'] = 0
    all_ed['LONGER_LEN'] = 0
    all_ed['TOTAL_EDIT_DIS'] = 0
    all_ed['TOTAL_HIT'] = 0
    all_ed['COUNTING_HIT'] = 0

    return_flag = False

    print('Evaluating...')
    prog_bar = mmcv.ProgressBar(len(input_dict.keys()))
    for _, value in input_dict.items():
        for res in value:
            edit = dict()
            label = res['gt_text']
            result = res['res_text']

            if isinstance(result, str):
                if '#' not in label and len(label) <= batch_max_length:

                    # filter the model prediction
                    result = filter_punctuation(result, r':(\'-,%>.[?)"=_*];&+$@/|!<#`{~\}^')
                    label = filter_punctuation(label, r':(\'-,%>.[?)"=_*];&+$@/|!<#`{~\}^')

                    # calculate the metric
                    editdis = edit_distance(result.replace(' ', ''),
                                            label.replace(' ', ''))

                    edit['edit_dis'] = editdis
                    edit['hit'] = 1 if editdis == 0 else 0
                    all_ed['VALID_NUM'] += 1
                    all_ed['LABEL_LEN'] += len(label)
                    all_ed['LONGER_LEN'] += max(len(label), len(result))
                    all_ed['TOTAL_EDIT_DIS'] += edit['edit_dis']
                    all_ed['TOTAL_HIT'] += edit['hit']
                else:
                    print(res, '#')
                    all_ed['NOT_CARE_NUM'] += 1

            if isinstance(result, int):
                return_flag = True

                # calculate the text counting accuracy for rf-learning
                if result == len(label):
                    all_ed['COUNTING_HIT'] += 1

                if '#' not in label and len(label) <= batch_max_length:
                    all_ed['VALID_NUM'] += 1

            all_ed['TOTAL_SAMPLE_NUM'] += 1
        prog_bar.update()

    # calculate the value of different metric
    all_ed['EDIT_ACC'] = 1.-float(all_ed['TOTAL_EDIT_DIS']) / all_ed['LABEL_LEN'] \
        if all_ed['LABEL_LEN'] > 0 else 0.0

    all_ed['EDIT_ACC_2'] = 1.-float(all_ed['TOTAL_EDIT_DIS']) / all_ed['LONGER_LEN'] \
        if all_ed['LONGER_LEN'] > 0 else 0.0

    all_ed['WORD_ACC'] = float(all_ed['TOTAL_HIT']) / all_ed['VALID_NUM'] \
        if all_ed['VALID_NUM'] > 0 else 0.0

    all_ed['COUNTING_ACC'] = float(all_ed['COUNTING_HIT']) / all_ed['VALID_NUM'] \
        if all_ed['COUNTING_HIT'] > 0 else 0.0

    # save the metric result with the json format
    with open(jout, "w", encoding="utf-8") as file:
        json.dump(all_ed, file, ensure_ascii=False, sort_keys=True, indent=2)

    print('\nEval Finished!')
    if not return_flag:
        return all_ed['WORD_ACC'] * 100.0

    return all_ed['COUNTING_ACC'] * 100.0


def results2json(dataset, results,
                 out_file, translate_table=None,
                 sensitive=False):
    """

    Args:
        dataset (Dataset): dataset
        results (list): model prediction results
        out_file (str): output json file path
        translate_table (dict): translate table, made by maketrans
        sensitive (bool): upper or lower, default False(lower)

    Returns:

    """

    # load the images information
    check_path(out_file)
    file_prefix = dataset.img_prefix
    img_infos = dataset.img_infos
    if isinstance(results, list):
        assert len(img_infos) == len(results), 'img_infos num {} != results num {}'.format(len(img_infos),
                                                                                           len(results))
    result_files = dict()
    print('Dumping results to json...')

    if isinstance(results[0], tuple):
        pred_results, pred_befores = results
    else:
        pred_results, pred_befores = results, None

    assert len(img_infos) == len(pred_results), 'img_infos num {} != results num {}'.format(len(img_infos),
                                                                                            len(pred_results))

    for idx, _ in enumerate(img_infos):

        img_info = img_infos[idx]
        result = pred_results[idx]
        pred_before = '' if pred_befores is None or not len(pred_befores[idx]) else pred_befores[idx]

        # load the image file name information
        if "filename" in img_info:
            key_path = img_info['filename'].replace(file_prefix, '')
        else:
            key_path = idx

        # load the image bounding box information
        if "bbox" not in img_info['ann']:
            img_info['ann']['bbox'] = None

        # load the text information
        if 'text' in img_info['ann']:
            gt_text = img_info['ann']['text']
            if not sensitive:
                gt_text = gt_text.lower()
            if translate_table is not None:
                gt_text = gt_text.translate(translate_table)
        else:
            gt_text = '<N>'
        bbox = img_info['ann']['bbox']
        if key_path not in result_files:
            result_files[key_path] = [
                {
                    'bbox': bbox, 'gt_text': gt_text,
                    'res_text': result, 'pred_before': pred_before}
            ]
        else:
            result_files[key_path].append(
                {
                    'bbox': bbox, 'gt_text': gt_text,
                    'res_text': result, 'pred_before': pred_before}
            )

    # save the model prediction with json file format
    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(result_files, file, ensure_ascii=False)


def show_result_table(dataset_name, rec_result):
    """

    Args:
        dataset_name (list): dataset name
        rec_result (list): dataset corresponding recognition accuracy

    Returns:

    """
    total_dataset = ["Iter/Epoch"] + dataset_name
    table = PrettyTable(total_dataset)
    if isinstance(rec_result[0], list):
        # multiple dataset metric
        for i, _ in enumerate(rec_result[0]):
            result_show = []
            for j, _ in enumerate(rec_result):
                if j == 0:
                    result_show.append(rec_result[j][i])
                else:
                    result_show.append(str(Decimal(rec_result[j][i]).quantize(Decimal("0.00"))))
            table.add_row(result_show)
    else:
        # single dataset metric
        table.add_row([str(Decimal(item).quantize(Decimal("0.00"))) if isinstance(item, float)
                       else item for item in rec_result])
    print(table)


def filter_punctuation(sentence, punctuation):
    """
    Args:
        sentence (str): string which needs to filter the punctuation
        punctuation (str): the punctuation which is unnecessary

    Returns:
        str: string without the unnecessary punctuation

    """
    temp_result = []
    # filter the punctuation in the model prediction
    for item in sentence:
        if item not in punctuation:
            temp_result.append(item)

    result = "".join(temp_result)
    return result
