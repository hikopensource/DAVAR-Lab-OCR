

# MMCV Plugins Implementations of DavarOCR

## 1. Introduction

This version of MMCV is the supplement component of official [MMCV](https://github.com/open-mmlab/mmcv), covering the related functions required by DavarOCR.

It provides the supplementary functionalities.

- [x] DavarCheckpointHook 

## 2. Functional Descriptions

- ##### DavarCheckpointHook

​       Save the best checkpoint during the training process. According to the evaluation result in the logger, save the best and latest checkpoint in order to make use of  the hardware resources. All functions implementation are upward compatible with the functions in official MMCV.

######   Demo

​       More specific usage methods, please refer to [Res_bilstm-attn](../../../demo/text_recognition/__base__/res32_bilstm_attn.py)

```
-------------------------------------------------------------------------------------
1. General Type
-------------------------------------------------------------------------------------
checkpoint_config = dict(type="DavarCheckpointHook",    # Checkpoint Hook Name
                         interval=1,              # Checkpoint save interval By Epoch
                         by_epoch=True,           # by_epoch: True  -- By Epoch
                                                  #           False -- By Iteration
                                                  #  Note: (Could not work together)
                         filename_tmpl='ckpt/ace_e{}.pth', 
                         # Checkpoint Save Name format
                         
                         metric="accuracy",               
                         # Save the best metric Name "Accuracy"
                         
                         rule="greater", 
                         # the Metric rule, including "greater" or "lower"
                         
                         save_mode="general", )
                         # General equals MMDetection Official Checkpoint Hook

-------------------------------------------------------------------------------------
2. Lightweight Type
-------------------------------------------------------------------------------------
checkpoint_config = dict(type="DavarCheckpointHook",       # Checkpoint Hook Name
                         interval=1,         # Checkpoint save interval By Epoch
                         iter_interval=1,    # Checkpoint save interval By Iteration
                         by_epoch=True,      # by_epoch: True -- By Epoch
                         by_iter=True,       # by_iter:  True -- By Iteration 
                         # (independent with By_epoch, could work together)
                         
                         filename_tmpl='ckpt/ace_e{}.pth', 
                         # Checkpoint Save Name format
                         
                         metric="accuracy",   # Save the best metric Name "Accuracy"
                         rule="greater",  
                         # the Metric rule, including "greater" or "lower"
                         
                         save_mode="lightweight",          
                         # Lightweight type, only save the best metric model and
                         # latest iteration and latest epoch model
                         
                         init_metric=-1,      # initial metric of the model 
                         
                         model_milestone=0.5) 
                         # the percentage of the training process to save checkpoint
-------------------------------------------------------------------------------------
```



## License
This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright
If there is any suggestion and problem, please feel free to contact the author with jianghui11@hikvision.com or chengzhanzhan@hikvision.com.
