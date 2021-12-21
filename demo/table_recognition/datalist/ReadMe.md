# PubTabNet

### Dataset list：
PubTabNet_2.0.0_val.jsonl: The original annotation of validation dataset in PubTabNet 2.0.0.

PubTabNet_train_datalist.json: The formated annotation of training dataset in PubTabNet 2.0.0, which is used in LGPMA training. Samples containing noises are filtered out.

PubTabNet_train_datalist_examples.json: Exapmles of PubTabNet_train_datalist.json.

Datalists can be downloaded from [link](https://one.hikvision.com/#/link/eqaugdX6YQgFKhUxF8U4) (Access Code：YF3O)

#### annotation example：
``` markdown
{
	"Images/train/PMC3348833_020_01.png": {
        "height": 90,
        "width": 395,
        "content_ann": {
            "bboxes": [
                [40,  4, 75,  20],  # bbox of text region in cell. Empty cell are noded as []. 
                [144, 4, 163, 20],
                ...
            ],
            "cells": [
                [0,0,0,0], # start row, start column， end row and end column of cell
                [0,1,0,1],
                ...
            ],
            "labels": [
                [0],    # label of cell. [0] means cell in head and [1] means cell in body
                [0],
                [0],
                [0],
                [1],
                [1],
                [1],
                [1],
                .....
            ]
        }
    },
}
```