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
                [0],    # label of cell. [0] means cell in head and [1] means cell in body.
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

### Convert html annotation to Davar format：
Current framework requires the datalist in the 'Davar Format' for training. Here is an example that how is the annotation in PubTabNet transferred into Davar Format.

The original annotation of PubTabNet looks like below, where the structure is described by html code (in "structure": "tokens") while bboxes and texts of each cell is recorded in "cells". The order of cells recorded in "cells" is consistent with the order of cells in html.

``` markdown
{
    "html": {
        "cells": [
            {"tokens": ["<b>", "T", "r", "a", "i", "t", "</b>"],
             "bbox": [11, 5, 33, 14]},
            {"tokens": ["<b>", "M", "e", "a", "n", "</b>"],
             "bbox": [202, 5, 225, 14]},
            {"tokens": ["S", "C", "S"],
             "bbox": [14, 27, 30, 35]},
            {"tokens": ["-", " ", "0", ".", "1", "0", "2", "4"],
             "bbox": [199, 27, 229, 35]}
        ],
        "structure": {"tokens":
            ["<thead>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</thead>",
             "<tbody>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</tbody>"]
        }
    }
}
```

We provide a script to convert it to Davar format, which can be found in [lgpma/tools/convert_html_ann.py](../lgpma/tools/convert_html_ann.py). 
The input of this script is a dictionary recorded in "html" and the output is a dictionary recorded in "content_ann". 
You may follow this script and modify it to prepare your own datalist if the original annotation is in other formats. 


Thanks to @tucachmo2202 in issue #37 for providing some examples, we have updated the convert script to filter out all illegal samples, which can be downloaded from [link](https://one.hikvision.com/#/link/oHvkYbH6fwXoKlSecPwj) (Access Code：Ft8N). 

It is worth noting that the number of element in "cells" should be the same as the number of '</td>' in "structure":"tokens". If the two are not equal, the annotation must be incorrectly and the script will filter out such sample.
Furthermore, the script will convert the illegal bboxes (with size of 0) into empty cells ("[]") to prevent the problem described in #51.

