#### Description about SROIE datalist：
The original [SROIE](https://rrc.cvc.uab.es/?ch=13&com=introduction) dataset contains many noise. Here, we release a clean version of SROIE datalist for fair comparison with our TRIE model.
The SROIE datalist includes OCR and IE annotations. 

- bboxes: four points annotation of text boxes(top-left, top-right, bottom-right, bottom-left)

- texts: text content annotation

- cares: whether each box is noise or not(0 means noise and will be ignored during training or testing)

- labels:  labels of each box (0 for 'other', 1 for 'company', 2 for 'total', 3 for 'total', 4 for 'date', 5 for 'address')

- bbox_bieo_labels: labels of each character in a box (with same meanings as labels)

  

#### Datalists：

 davar_train_datalist_w_bieo.json：  626 training samples

 davar_test_datalist_w_bieo.json：   347 testing samples

 classes_config is as follows:
```
{
	"classes":
	[
    	"others",
   		"company",
    	"total",
    	"date",
    	"address"
	]
}
```



#### Example：

	{
	 "Images/train/X51005757349.jpg": 
	 {
	  "height": 1373,
	  "width": 703,
	  "content_ann": 
	  {
	    "bboxes": [[227, 159, 457, 159, 457, 185, 227, 185], 
	               [242, 193, 426, 193, 426, 223, 242, 223], 
	               [185, 229, 482, 229, 482, 262, 185, 262], 
	               [198, 267, 481, 267, 481, 299, 198, 299], 
	               [170, 305, 499, 305, 499, 337, 170, 337], 
	               [226, 343, 454, 343, 454, 372, 226, 372], 
	               ...
	              ], 
		"cares": [1, 1, 1, 1, 1, 1, ...], 
		"labels": [[1], 
	               [0], 
	               [4], 
	               [4], 
	               [4], 
	               [0], 
	               ...
	              ],
	    "bbox_bieo_labels":[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
	                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 
	                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 
	                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 
	                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	                        ...
	                       ]
		"texts": [
	              "GOLDEN KEY MAKER",
	              "(000760274-K)",
	              "NO 5, JALAN KENARI 2,",
	              "BANDAR PUCHONG JAYA,",
	              "47100 PUCHONG, SELANGOR",
	              "TEL: 03-58919941",
	              ...
	             ]
	   }
	 },
	 ...
	}
	
	
##### Download Link： 

You can download the datalists via this [link](https://one.hikvision.com/#/link/O0DXYBPhlqpGQI7nmRFA). (Access Code: XYAZ)

