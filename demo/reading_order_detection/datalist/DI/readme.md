# DI Dataset

DI dataset is the e-commerce images that proposed in "An End-to-End OCR Text Re-organization Sequence Learning for Rich-text Detail Image Comprehension", the original version contains 10k images. For some business reason, they only release ~8k images for reaserch. The dataset can be downloaded from [TianChi](https://tianchi.aliyun.com/dataset/dataDetail?dataId=72926).

### Annoation 
We transfer the original annoation into the uniform Davar format as follows,

``` markdown
{
	"O1CN01020zzO1pMyIRdTHd2_!!4117975347.jpg": 
	{
		"height": 1176, 
		"width": 790, 
		"content_ann": {
			"bboxes": [[316, 68, 468, 68, 468, 97, 316, 97],             # text boxes
			          [332, 132, 458, 132, 458, 148, 332, 148], 
					  [286, 230, 343, 230, 343, 247, 286, 247], 
					  [369, 230, 589, 230, 589, 248, 369, 248], 
					  ...], 
			"texts": ["产品信息",                                        # text contents                        
			          "ABOUTPRODUCTS", 
					  "【产品】", 
					  "：康绮墨丽盈润清爽洗发乳",
					  ...], 
			"labels": [[1], [2], [3], [4],...]                           # Reader order, 1,2,3...
		}
	},
	...
},

```

The formatted datalist can be downloaded from [here](https://one.hikvision.com/#/link/KcIlJnuCJxv36pvhGqkc) (Access Code：JhF6)