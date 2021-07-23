#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import rrc_evaluation_funcs
import importlib
import sys

import math


def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """
    return {
        'Polygon': 'plg',
        'numpy': 'np'
    }


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    p = dict([s[1:].split('=') for s in sys.argv[1:]])

    if p['g'].split("/")[-1] in ['gt_ctw1500_det.zip', 'gt_ctw1500_det_with_ignore.zip']:
        return {
            'IOU_CONSTRAINT': 0.5,
            'AREA_PRECISION_CONSTRAINT': 0.5,
            'GT_SAMPLE_NAME_2_ID': '([0-9]+).txt',
            'DET_SAMPLE_NAME_2_ID': '([0-9]+).txt',
            'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
            'CRLF': False,  # Lines are delimited by Windows CRLF format
            'CONFIDENCES': False,  # Detections must include confidence value. AP will be calculated
            'PER_SAMPLE_RESULTS': True  # Generate per sample results and produce data for visualization
        }
    elif p['g'].split("/")[-1] in ['total-text-gt.zip']:
        return {
            'IOU_CONSTRAINT': 0.5,
            'AREA_PRECISION_CONSTRAINT': 0.5,
            'GT_SAMPLE_NAME_2_ID': 'poly_gt_img([0-9]+).txt',
            'DET_SAMPLE_NAME_2_ID': 'img([0-9]+).txt',
            'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
            'CRLF': False,  # Lines are delimited by Windows CRLF format
            'CONFIDENCES': False,  # Detections must include confidence value. AP will be calculated
            'PER_SAMPLE_RESULTS': True  # Generate per sample results and produce data for visualization
        }
    elif p['g'].split("/")[-1] in ['gt-icdar2015.zip']:
        return {
            'IOU_CONSTRAINT': 0.5,
            'AREA_PRECISION_CONSTRAINT': 0.5,
            'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
            'DET_SAMPLE_NAME_2_ID': 'img_([0-9]+).txt',
            'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
            'CRLF': False,  # Lines are delimited by Windows CRLF format
            'CONFIDENCES': False,  # Detections must include confidence value. AP will be calculated
            'PER_SAMPLE_RESULTS': True  # Generate per sample results and produce data for visualization
        }
    else:
        raise NotImplementedError


def validate_data(gtFilePath, submFilePath, evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])

    subm = rrc_evaluation_funcs.load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)

    # Validate format of GroundTruth
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file(k, gt[k], evaluationParams['CRLF'], evaluationParams['LTRB'], True)

    # Validate format of results
    for k in subm:
        if (k in gt) == False:
            raise Exception("The sample %s not present in GT" % k)
        rrc_evaluation_funcs.validate_lines_in_file(k, subm[k], evaluationParams['CRLF'], evaluationParams['LTRB'],
                                                    False, evaluationParams['CONFIDENCES'])


def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    for module, alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)

    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
        num_points = len(points)
        # resBoxes=np.empty([1,num_points],dtype='int32')
        resBoxes = np.empty([1, num_points], dtype='float32')
        for inp in range(0, num_points, 2):
            # print(inp, points)
            # print(resBoxes[0, inp/2])
            resBoxes[0, int(inp / 2)] = float(points[inp])
            resBoxes[0, int(inp / 2 + num_points / 2)] = float(points[inp + 1])
        pointMat = resBoxes[0].reshape([2, int(num_points / 2)]).T
        return plg.Polygon(pointMat)

    def rectangle_to_polygon(rect):
        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(rect.xmin)
        resBoxes[0, 4] = int(rect.ymax)
        resBoxes[0, 1] = int(rect.xmin)
        resBoxes[0, 5] = int(rect.ymin)
        resBoxes[0, 2] = int(rect.xmax)
        resBoxes[0, 6] = int(rect.ymin)
        resBoxes[0, 3] = int(rect.xmax)
        resBoxes[0, 7] = int(rect.ymax)

        pointMat = resBoxes[0].reshape([2, 4]).T

        return plg.Polygon(pointMat)

    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin),
                  int(rect.xmin), int(rect.ymin)]
        return points

    def get_union(pD, pG):
        areaA = pD.area();
        areaB = pG.area();
        return areaA + areaB - get_intersection(pD, pG);

    def get_intersection_over_union(pD, pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG);
        except:
            return 0

    def funcCt(x):
        if x <= 0.01:
            return 1
        else:
            return 1 - x

    def get_text_intersection_over_union_recall(pD, pG):
        '''
        Ct (cut): Area of ground truth that is not covered by detection bounding box.
        '''
        try:
            Ct = pG.area() - get_intersection(pD, pG)
            assert (Ct >= 0 and Ct <= pG.area()), 'Invalid Ct value'
            assert (pG.area() > 0), 'Invalid Gt'
            return (get_intersection(pD, pG) * funcCt(Ct * 1.0 / pG.area())) / get_union(pD, pG);
        except Exception as e:
            return 0

    def funcOt(x):
        if x <= 0.01:
            return 1
        else:
            return 1 - x

    def get_text_intersection_over_union_precision(pD, pG, gtNum, gtPolys, gtDontCarePolsNum):
        '''
        Ot: Outlier gt area
        '''
        Ot = 0
        try:
            inside_pG = pD & pG
            gt_union_inside_pD = None
            gt_union_inside_pD_and_pG = None
            count_initial = 0
            for i in xrange(len(gtPolys)):
                if i != gtNum and gtNum not in gtDontCarePolsNum:  # ignore don't care regions
                    if not get_intersection(pD, gtPolys[i]) == 0:
                        if count_initial == 0:
                            # initial
                            gt_union_inside_pD = gtPolys[i]
                            gt_union_inside_pD_and_pG = inside_pG & gtPolys[i]
                            count_initial = 1
                            continue
                        gt_union_inside_pD = gt_union_inside_pD | gtPolys[i]
                        inside_pG_i = inside_pG & gtPolys[i]
                        gt_union_inside_pD_and_pG = gt_union_inside_pD_and_pG | inside_pG_i

            if not gt_union_inside_pD == None:
                pD_union_with_other_gt = pD & gt_union_inside_pD
                Ot = pD_union_with_other_gt.area() - gt_union_inside_pD_and_pG.area()
                if Ot <= 1.0e-10:
                    Ot = 0
            else:
                Ot = 0
            # allow invalid polygon
            assert (Ot >= 0 and Ot <= pD.area())
            assert (pD.area() > 0)
            return (get_intersection(pD, pG) * funcOt(Ot * 1.0 / pD.area())) / get_union(pD, pG);
        except Exception as e:
            # print(e)
            return 0

    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def get_intersection_three(pD, pG, pGi):
        pInt = pD & pG
        pInt_3 = pInt & pGi
        if len(pInt_3) == 0:
            return 0
        return pInt_3.area()

    def compute_ap(confList, matchList, numGtCare):
        correct = 0
        AP = 0
        if len(confList) > 0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct) / (n + 1)

            if numGtCare > 0:
                AP /= numGtCare

        return AP

    perSampleMetrics = {}

    matchedSum = 0
    matchedSum_iou = 0
    matchedSum_tiouGt = 0
    matchedSum_tiouDt = 0
    matchedSum_cutGt = 0
    matchedSum_coverOtherGt = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)

    numGlobalCareGt = 0;
    numGlobalCareDet = 0;

    arrGlobalConfidences = [];
    arrGlobalMatches = [];

    totalNumGtPols = 0
    totalNumDetPols = 0

    # fper_ = open('per_samle_result.txt', 'w')

    for resFile in gt:
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0
        detMatched_iou = 0
        detMatched_tiouGt = 0
        detMatched_tiouDt = 0
        detMatched_cutGt = 0
        detMatched_coverOtherGt = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = [];
        arrSampleMatch = [];
        sampleAP = 0;

        evaluationLog = ""

        pointsList, _, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,
                                                                                                       evaluationParams[
                                                                                                           'CRLF'],
                                                                                                       evaluationParams[
                                                                                                           'LTRB'],
                                                                                                       True, False)
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams['LTRB']:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

        if resFile in subm:

            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile])

            pointsList, confidencesList, _ = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,
                                                                                                        evaluationParams[
                                                                                                            'CRLF'],
                                                                                                        evaluationParams[
                                                                                                            'LTRB'],
                                                                                                        False,
                                                                                                        evaluationParams[
                                                                                                            'CONFIDENCES'])
            for n in range(len(pointsList)):
                points = pointsList[n]

                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                if len(gtDontCarePolsNum) > 0:
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol, detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT']):
                            detDontCarePolsNum.append(len(detPols) - 1)
                            break

            evaluationLog += "DET polygons: " + str(len(detPols)) + (
                " (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")

            if len(gtPols) > 0 and len(detPols) > 0:
                # Calculate IoU and precision matrixs
                outputShape = [len(gtPols), len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)
                tiouRecallMat = np.empty(outputShape)
                tiouPrecisionMat = np.empty(outputShape)
                tiouGtRectMat = np.zeros(len(gtPols), np.int8)
                tiouDetRectMat = np.zeros(len(detPols), np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
                        tiouRecallMat[gtNum, detNum] = get_text_intersection_over_union_recall(pD, pG)
                        tiouPrecisionMat[gtNum, detNum] = get_text_intersection_over_union_precision(pD, pG, gtNum,
                                                                                                     gtPols,
                                                                                                     gtDontCarePolsNum)

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                            if iouMat[gtNum, detNum] > evaluationParams['IOU_CONSTRAINT']:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                detMatched += 1
                                detMatched_iou += iouMat[gtNum, detNum]
                                detMatched_tiouGt += tiouRecallMat[gtNum, detNum]
                                detMatched_tiouDt += tiouPrecisionMat[gtNum, detNum]
                                if iouMat[gtNum, detNum] != tiouRecallMat[gtNum, detNum]:
                                    detMatched_cutGt += 1
                                if iouMat[gtNum, detNum] != tiouPrecisionMat[gtNum, detNum]:
                                    detMatched_coverOtherGt += 1
                                pairs.append({'gt': gtNum, 'det': detNum})
                                detMatchedNums.append(detNum)
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"

            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum:
                        # we exclude the don't care detections
                        match = detNum in detMatchedNums

                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)

                        arrGlobalConfidences.append(confidencesList[detNum]);
                        arrGlobalMatches.append(match);

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        # if numGtCare == 0:
        #     recall = float(1)
        #     precision = float(0) if numDetCare > 0 else float(1)
        #     sampleAP = precision
        #     tiouRecall = float(1)
        #     tiouPrecision = float(0) if numDetCare > 0 else float(1)
        # else:
        #     recall = float(detMatched) / numGtCare
        #     precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare
        #     iouRecall = float(detMatched_iou) / numGtCare
        #     iouPrecision = 0 if numDetCare == 0 else float(detMatched_iou) / numDetCare
        #     tiouRecall = float(detMatched_tiouGt) / numGtCare
        #     tiouPrecision = 0 if numDetCare == 0 else float(detMatched_tiouDt) / numDetCare
        #
        #     if evaluationParams['CONFIDENCES'] and evaluationParams['PER_SAMPLE_RESULTS']:
        #         sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare)
        #
        # hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        # tiouHmean = 0 if (tiouPrecision + tiouRecall) == 0 else 2.0 * tiouPrecision * tiouRecall / (
        #             tiouPrecision + tiouRecall)
        # iouHmean = 0 if (iouPrecision + iouRecall) == 0 else 2.0 * iouPrecision * iouRecall / (iouPrecision + iouRecall)

        matchedSum += detMatched
        matchedSum_iou += detMatched_iou
        matchedSum_tiouGt += detMatched_tiouGt
        matchedSum_tiouDt += detMatched_tiouDt
        matchedSum_cutGt += detMatched_cutGt
        matchedSum_coverOtherGt += detMatched_coverOtherGt
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        if evaluationParams['PER_SAMPLE_RESULTS']:
            perSampleMetrics[resFile] = {
                'precision': precision,
                'recall': recall,
                'hmean': hmean,
                # 'iouPrecision': iouPrecision,
                # 'iouRecall': iouRecall,
                # 'iouHmean': iouHmean,
                # 'tiouPrecision': tiouPrecision,
                # 'tiouRecall': tiouRecall,
                # 'tiouHmean': tiouHmean,
                'pairs': pairs,
                'AP': sampleAP,
                'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
                'gtPolPoints': gtPolPoints,
                'detPolPoints': detPolPoints,
                'gtDontCare': gtDontCarePolsNum,
                'detDontCare': detDontCarePolsNum,
                'evaluationParams': evaluationParams,
                'evaluationLog': evaluationLog
            }
        # fper_.writelines(resFile+'\t"IoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})",\t"TIoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})".\n'.format(precision, recall, hmean, tiouPrecision, tiouRecall, tiouHmean))
        # try:
        totalNumGtPols += len(gtPols)
        totalNumDetPols += len(detPols)
        # except Exception as e:
        #    print
        #    raise e
    # fper_.close()

    # Compute MAP and MAR
    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    print('num_gt, num_det: ', numGlobalCareGt, totalNumDetPols)
    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
                methodRecall + methodPrecision)

    methodRecall_iou = 0 if numGlobalCareGt == 0 else float(matchedSum_iou) / numGlobalCareGt
    methodPrecision_iou = 0 if numGlobalCareDet == 0 else float(matchedSum_iou) / numGlobalCareDet
    iouMethodHmean = 0 if methodRecall_iou + methodPrecision_iou == 0 else 2 * methodRecall_iou * methodPrecision_iou / (
                methodRecall_iou + methodPrecision_iou)

    methodRecall_tiouGt = 0 if numGlobalCareGt == 0 else float(matchedSum_tiouGt) / numGlobalCareGt
    methodPrecision_tiouDt = 0 if numGlobalCareDet == 0 else float(matchedSum_tiouDt) / numGlobalCareDet
    tiouMethodHmean = 0 if methodRecall_tiouGt + methodPrecision_tiouDt == 0 else 2 * methodRecall_tiouGt * methodPrecision_tiouDt / (
                methodRecall_tiouGt + methodPrecision_tiouDt)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}
    iouMethodMetrics = {'iouPrecision': methodPrecision_iou, 'iouRecall': methodRecall_iou, 'iouHmean': iouMethodHmean}
    tiouMethodMetrics = {'tiouPrecision': methodPrecision_tiouDt, 'tiouRecall': methodRecall_tiouGt,
                         'tiouHmean': tiouMethodHmean}
    # print('matchedSum: ', matchedSum, 'matchedSum_cutGt: ', matchedSum_cutGt, 'cut_Rate: ', round(matchedSum_cutGt*1.0/matchedSum, 3), 'matchedSum_coverOtherGt: ', matchedSum_coverOtherGt, 'cover_Outlier_Rate: ', round(matchedSum_coverOtherGt*1.0/matchedSum, 3))
    print('Origin:')
    print("recall: ", round(methodRecall, 4), "precision: ", round(methodPrecision, 4), "hmean: ",
          round(methodHmean, 4))
    # print('SIoU-metric:')
    # print("iouRecall:", round(methodRecall_iou,3), "iouPrecision:", round(methodPrecision_iou,3), "iouHmean:", round(iouMethodHmean,3))
    # print('TIoU-metric:')
    # print("tiouRecall:", round(methodRecall_tiouGt,3), "tiouPrecision:", round(methodPrecision_tiouDt,3), "tiouHmean:", round(tiouMethodHmean,3))

    resDict = {'calculated': True, 'Message': '', 'method': methodMetrics, 'per_sample': perSampleMetrics,
               'iouMethod': iouMethodMetrics, 'tiouMethod': tiouMethodMetrics}

    return resDict


if __name__ == '__main__':
    rrc_evaluation_funcs.main_evaluation(None, default_evaluation_params, validate_data, evaluate_method)
