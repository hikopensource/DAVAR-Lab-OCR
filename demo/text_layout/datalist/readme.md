## Datalist preparation for text_layout

This is an explanation about how datalists are formed in davarocr for text_layout tasks.

- ### DocBank Dataset

Take the training datalist `Datalist/500K_train_datalist.json` as example:

    {
        "287.tar_1712.03005.gz_constrained_descent_paper_arXiv_15_ori.jpg": {  
            "height": 1000,
            "width": 707,
            "url": "287.tar_1712.03005.gz_constrained_descent_paper_arXiv_15_ori.json" 
        },
        "114.tar_1507.06562.gz_main_merged_3_ori.jpg": {
            "height": 1000,
            "width": 773,
            "url": "114.tar_1507.06562.gz_main_merged_3_ori.json"
        }
        ...
    }

For each image, such as `287.tar_1712.03005.gz_constrained_descent_paper_arXiv_15_ori.jpg`, its detailed annotation is in `Annos/287.tar_1712.03005.gz_constrained_descent_paper_arXiv_15_ori.json`, which is:

    { 
        "height": 1000, 
        "width": 707, 
        "content_ann": {           # token-level annotations
            "bboxes": [[326, 107, 333, 155], [75, 118, 109, 131], ...],     # bounding boxes
            "texts": ["(cid:90)", "Proof.", "The", "proof", ...],           # text content extract by PDF extractor
            "labels": [["paragraph"], ["paragraph"], ["paragraph"], ...],   # categories
            "cares": [1, 1, 1, ...],          # whether to be considered during training
         },
         "content_ann2": {         # layout-level annotations
            "bboxes": [[75, 335, 631, 608], [75, 107, 631, 288],...], 
            "labels": [["paragraph"], ["paragraph"], ["paragraph"]...], 
            "cares": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}} 
         }
     }

Please download original dataset from [DocBank](https://github.com/doc-analysis/DocBank) and use the script provided in [`DocBank/Scripts/tools.py`](./DocBank/Scripts/tools.py) to convert to Davar format.

- ### PubLayNet

Take the training datalist `Datalist/datalist_train.json` as example:

    {
        "Images/train/PMC5892139_00005.jpg": {
            "height": 698,
            "width": 536,
            "url": "PMC5892139_00005.json"
        },
        "Images/train/PMC5345541_00001.jpg": {
            "height": 842,
            "width": 596,
            "url": "PMC5345541_00001.json"
        }
        ...
    }


For each image, such as `PMC5345541_00001.jpg`, the detailed annotation is in `Annos/train/PMC5345541_00001.json`, which is:

    { 
        "height": 842, 
        "width": 596, 
        "content_ann": {           # token-level annotations
            "bboxes": [[72, 46, 507, 54], [218, 345, 235, 355], ...],     # bounding boxes
            "texts": ["Smith and Malkowicz...", "for", ...]        # text content extract by PDF extractor
            "labels": [[0], [1], [1], ...],   # categories
            "cares": [1, 1, 1, ...],          # whether to be considered during training
            "cbboxes": [[[72, 47, 77, 55], [77, 46, 84, 55]..]]      # character bounding boxes
            "ctexts": [["S","m","i"...]]                             # character content extract by PDF extractor
         },
         "content_ann2": {         # layout-level annotations
            "bboxes": [[72, 87, 289, 214], [72, 228, 289, 425],...], 
            "labels": [[1], [1], [1]...], 
            "cares": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}} ,
            "segboxes": [[[72, 87, 289, 87,...]],[[81.62, 228.29, ...]],...]           # segboxes for each layout component
         }
     }

For the PubLayNet, you can get the Davar format datalist through the following steps:

1. Dataset preparation: download the [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) dataset together with its PDF files
2. Annotation parsing: use [pdfplumber](https://github.com/jsvine/pdfplumber) to parse the *bboxes, texts, cbboxes, ctexts* in *content_ann* from PDF files (if you want to generate labels for text lines, map layout-level annotations to text lines through IOU matching). You can refer to the script provided in [DocBank](https://github.com/doc-analysis/DocBank).
3. Format convertion: convert the above annotations to Davar format.
4. The coco_val.json file can be downloaded from [Link](https://one.hikvision.com/#/link/h7WpuP9kvop6JpYFBPgF) (Access Code: 6kdQ).