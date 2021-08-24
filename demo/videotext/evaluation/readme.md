## Evaluation tool

This evaluation tools implements the End-to-End metrics for text tracking and recognition, contains: MOTA, MOTP, ATA in [Survey](https://dblp.org/rec/journals/tip/YinZTL16.html).
We have added an additional constraint on whether the recognition result is correct 

Also we implements the redefined end-to-end metrics F-score-R which is described in [YORO](https://arxiv.org/pdf/1903.03299.pdf).


The default evaluation metric sets IoU constraint as 0.5.

#### evaluation formats:

You can open `demo/videotext/yoro/evaluation/gt/` to check out the format of gt, and open `demo/videotext/yoro/evaluation/pred/` to check out predict file format.

Notice that,
> In MOT-R and ATA-R evaluation, only 'tracks', 'text' are required. 
>>'scores' is only used for training in YORO model and not used in evaluation.

> In F-socre-R evaluation, another key named 'selected_frame' is also required, which tells which frame in the track seq is chosen to stand for such seq. 
>> In YORO, we just
select the frame has highest quality score to be selected frame. 


#### Do evaluation

1. F-score evaluation

    For F-score-R evaluation, directly run
    
        python evaluate_hmeans.py ./pred/IC15_pred_recommder_result.json ./gt/IC15/IC15_e2e_gt.json --voca_file ./gt/IC15/IC15_voca.json
    	   
    will produce
    
        ****************************final Fscore result*********************************
        total recall = 468/675 = 0.6933333333333334
        total precision = 468/676 = 0.6923076923076923
        total h-means = 0.692820133234641
        ********************************************************************************
        
2. ATA evaluation
    
    For pure ATA evaluation, directly run
    
        python evaluate_ata.py ./pred/IC15_pred_track_result.json ./gt/IC15/IC15_e2e_gt.json --care_rcg 0

    will produce
        
        ****************************final ATA result***************************
        final ata : 0.6488747295392859
        ***********************************************************************
    
    If you want to add recognition constraint, directly run
    
        python evaluate_ata.py ./pred/IC15_pred_recommder_result.json ./gt/IC15/IC15_e2e_gt.json --voca_file ./gt/IC15/IC15_voca.json
    
    will produce
    
        ****************************final ATA result***************************
        final ata : 0.6258118319072516
        ***********************************************************************
    
3. MOT evaluation
    
    For pure MOT evaluation, Directly run
    
        python evaluate_mot.py ./pred/IC15_pred_track_result.json ./gt/IC15/IC15_e2e_gt.json --care_rcg 0
    
    will produce
        
        ****************************final MOT result******************************
        avg all frames MOTP: 0.7374772589563159
        avg all frames MOTA: 0.7170697433621773
        **************************************************************************
    
    If you want to add recognition constraint, directly run:
    
        python evaluate_mot.py ./pred/IC15_pred_recommder_result.json ./gt/IC15/IC15_e2e_gt.json --voca_file ./gt/IC15/IC15_voca.json

    will produce

        ****************************final MOT result******************************
        avg all frames MOTP: 0.739436918946066
        avg all frames MOTA: 0.678685008505288
        **************************************************************************   
    
Go into the directory of each algorithm for detailed evaluation results.
