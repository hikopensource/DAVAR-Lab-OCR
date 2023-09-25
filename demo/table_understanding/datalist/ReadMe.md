# ComFinTab

### Dataset list：
ComFinTab_chn_train.json: The formated annotation of Chinese training dataset in ComFinTab which is used in CTUNet training.

ComFinTab_chn_test.json: The formated annotation of Chinese training dataset in ComFinTab, which is used in CTUNet testing.

ComFinTab_eng_trainjson: The formated annotation of English training dataset in ComFinTab, which is used in CTUNet training.

ComFinTab_eng_test.json: The formated annotation of English training dataset in ComFinTab, which is used in CTUNet testing.

ComFinTab_examples.json: The formated annotation of English training dataset in ComFinTab, which is used in CTUNet testing.

Full dataset and datalists can be downloaded from [ComFinTab](https://davar-lab.github.io/dataset/comfintab.html).


#### annotation example:
``` markdown
{
	"0020.jpeg": {
		"height": 260, 
		"width": 1335, 
		"content_ann": {
			"bboxes": [
				[27, 27, 1356, 27, 1356, 82, 27, 82], 		# The bboxes of real cells.
				[27, 82, 1356, 82, 1356, 181, 27, 181], 	# ditto
				[27, 181, 397, 181, 397, 281, 27, 281], 	# ditto
				[397, 181, 1356, 181, 1356, 281, 397, 281], # ditto
				[11, 27, 27, 27, 27, 82, 11, 82], 			# The coordinates of the virtual row node. The length in the "X" direction is fixed to 16 pixels
				[11, 82, 27, 82, 27, 181, 11, 181], 		# ditto
				[11, 181, 27, 181, 27, 281, 11, 281], 		# ditto
				[27, 11, 397, 11, 397, 27, 27, 27], 		# The coordinates of the virtual column node. The length in the "Y" direction is fixed to 16 pixels
				[397, 11, 1356, 11, 1356, 27, 397, 27]], 	# ditto
			"texts": [
				"Deliberation section of auditing report of IC", 				# The texts of real cells.
				"In our opinion, the Company (Zhonglu), in line with Basic No",	# ditto
				"Disclosure details of audit report\nof internal control", 		# ditto
				"Disclosed", 													# ditto.
				"[CLS]", # The texts of the virtual node is uniformly set to "[CLS]"
				"[CLS]", # ditto
				"[CLS]", # ditto
				"[CLS]", # ditto
				"[CLS]"],# ditto
			"cells": [
				[0, 0, 0, 1], 	# start row, start column, end row and end column of real cells.
				[1, 0, 1, 1],	# ditto
				[2, 0, 2, 0],	# ditto
				[2, 1, 2, 1],	# ditto
				[0, -1, 0, -1], # Virtual row node of the first row. Note that the column number of virtual row node is - 1. 
				[1, -1, 1, -1], # Virtual row node of the seconde row. Note that the column number of virtual row node is - 1. 
				[2, -1, 2, -1], # Virtual row node of the third row. Note that the column number of virtual row node is - 1. 
				[-1, 0, -1, 0], # Virtual column node of the first column. Note that the row number of virtual column node is - 1. 	
				[-1, 1, -1, 1]],# Virtual column node of the second column. Note that the row number of virtual column node is - 1.
			"labels": [
				[0], 	# top header
				[2], 	# data
				[1], 	# left header
				[2], 	# data
				[4], 	# virtual node
				[4], 	# ditto
				[4], 	# ditto
				[4], 	# ditto
				[4]], 	# ditto
			"relations": [
				[0, 0, 0, 0, 0, 0, 0, 0, 0], # the cells is not a root cell.
				[1, 0, 0, 0, 0, 2, 0, 0, 0], # the right child node and left child node of 1-th cell ([1, 0, 1, 1]) is 0-th cell([0, 0, 0, 1]) and 5-th cell([1, -1, 1, -1]) respectively.
				[0, 0, 0, 0, 0, 0, 0, 0, 0], # the cells is not a root cell.
				[0, 0, 2, 0, 0, 0, 0, 0, 1], # the right child node and left child node of 3-th cell ([2, 1, 2, 1]) is 2-th cell([2, 0, 2, 0]) and 8-th cell([-1, 1, -1, 1]]) respectively.
				[0, 0, 0, 0, 0, 0, 0, 0, 0], # the cells is not a root cell.
				[0, 0, 0, 0, 0, 0, 0, 0, 0], # ditto
				[0, 0, 0, 0, 0, 0, 0, 0, 0], # ditto
				[0, 0, 0, 0, 0, 0, 0, 0, 0], # ditto
				[0, 0, 0, 0, 0, 0, 0, 0, 0]] # ditto
		}
	}
}
```

