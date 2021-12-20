"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    TL_iou.py
# Abstract       :    Text detection evaluation metrics. Refactored from official implemantion of
                      IC15 and SCUT-CTW1500 (support curve text)

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
import numpy as np
import Polygon as plg

def evaluate_method(det_results, gt_results, evaluationParams):
    """ Text detection evaluation metrics computation, including Hmean and tiou-Hmean.

    Args:
        det_results(list(dict)): the detection results predicted by model, in form of:
                            [
                                {
                                    "points": [[x1, y1, ..., xn, yn], [], ...[]],
                                    "confidence": [1.0, 0.4, 0.5, ..., 1.0 ],
                                    "texts": ["apple", "mango", "###', ..., "aaa"].
                                },
                                {
                                },
                                ...
                            ]
        gt_results(list(dict)): ground truth for dataset, in form of:
                          [
                             "{
                                "gt_bboxes": [[x1, y1, ..., xn, yn], [], ...[]],
                                "gt_texts": ["apple", "mango", "###', ..., "aaa"]
                             }
                          ]
        evaluationParams(dict): evaluation parameters, including:
                               - 'IOU_CONSTRAINT' : default 0.5,
                               - 'AREA_PRECISION_CONSTRAINT' : default 0.5
                               - 'CONFIDENCES':deafult False, #  if True, AP will be calculated


    Returns:
        dict: evaluation results, including:
             {
              'precision', 'recall','hmean', 'ave_precision', 'tiouPrecision', 'tiouRecall', 'tiouHmean',
               "IOU_CONSTRAINT", "AREA_PRECISION_CONSTRAINT","CONFIDENCES": evaluationParams['CONFIDENCES']
              }
    """

    def polygon_from_points_any_shape(points):
        """Returns a Polygon object to use with the Polygon2 class from a list of arbitrary number of points.

        Args:
            points(list(float)): points in list format [ x1,y1,x2,y2,...,xn,yn]

        Returns:
            plg.Polygon: polygon obj of the box
        """
        n = len(points)
        resBoxes = np.empty([1, n], dtype='int32')
        for i in range(int(n / 2)):
            resBoxes[0, i] = int(points[2 * i])
            resBoxes[0, i + int(n / 2)] = int(points[2 * i + 1])
        temp = resBoxes[0][0:int(n/2)*2]
        pointMat = temp.reshape([2, int(n / 2)]).T
        return plg.Polygon(pointMat)

    def get_union(pD, pG):
        """Get the union area of two areas.

        Args:
            pD(plg.Polygon): polygon object
            pG(plg.Polygon): polygon object
        Returns:
            float: union area of pD and pG
        """
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(pD, pG):
        """Get the iou two areas.

        Args:
            pD(plg.Polygon): polygon object
            pG(plg.Polygon): polygon object
        Returns:
            float: iou of pD and pG
        """
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except Exception as e:
            return 0

    def funcCt(x):
        if x <= 0.01:
            return 1
        else:
            return 1 - x

    def get_text_intersection_over_union_recall(pD, pG):
        """Ct (cut): Area of ground truth that is not covered by detection bounding box. Used for tiou-computation.

        Args:
            pD(plg.Polygon): polygon object
            pG(plg.Polygon): polygon object
        Returns:
            float: tiou-recall
        """
        try:
            Ct = pG.area() - get_intersection(pD, pG)
            assert 0 <= Ct <= pG.area(), 'Invalid Ct value'
            assert pG.area() > 0, 'Invalid Gt'
            return (get_intersection(pD, pG) * funcCt(Ct * 1.0 / pG.area())) / get_union(pD, pG)
        except Exception as e:
            return 0

    def funcOt(x):
        if x <= 0.01:
            return 1
        else:
            return 1 - x

    def get_text_intersection_over_union_precision(pD, pG, gtNum, gtPolys, gtDontCarePolsNum):
        """ Ot: Outlier gt area, Used for tiou-computation.

        Args:
            pD(plg.Polygon): polygon object
            pG(plg.Polygon): polygon object
        Returns:
            float: tiou-precision
        """
        try:
            inside_pG = pD & pG
            gt_union_inside_pD = None
            gt_union_inside_pD_and_pG = None
            count_initial = 0
            for i in range(len(gtPolys)):
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

            # Allow invalid polygon
            assert 0 <= Ot <= pD.area()
            assert pD.area() > 0
            return (get_intersection(pD, pG) * funcOt(Ot * 1.0 / pD.area())) / get_union(pD, pG)
        except Exception as e:
            return 0

    def get_intersection(pD, pG):
        """Calculate the intersect area of two areas

        Args:
            pD(plg.Polygon): polygon object
            pG(plg.Polygon): polygon object
        Returns:
            float: intersection of two areas
        """
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def compute_ap(confList, matchList, numGtCare):
        """Compute AP metric of detection results."""
        correct = 0
        AP = 0
        if len(confList)>0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)

            if numGtCare>0:
                AP /= numGtCare

        return AP

    matchedSum = 0
    matchedSum_tiouGt = 0
    matchedSum_tiouDt = 0
    matchedTag = {}

    numGlobalCareGt = 0
    numGlobalCareDet = 0
    numGlobalCareTag = {}
    
    arrGlobalConfidences = []
    arrGlobalMatches = []

    for gt_result, det_result in zip(gt_results, det_results):

        detMatched = 0
        detMatched_tiouGt = 0
        detMatched_tiouDt = 0

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

        arrSampleConfidences = []
        arrSampleMatch = []

        # Process grount truth
        pointsList = gt_result['gt_bboxes']
        transcriptionsList = gt_result['gt_texts']
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            gtPol = polygon_from_points_any_shape(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols)-1 )

        # Process detection results
        pointsList = det_result['points']
        confidencesList = det_result['confidence']
        for n in range(len(pointsList)):
            points = pointsList[n]
            detPol = polygon_from_points_any_shape(points)
            detPols.append(detPol)
            detPolPoints.append(points)
            # If the det are is intersected with some NOTCARE area, then append it into DontCare list
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = detPol.area()
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if precision > evaluationParams['AREA_PRECISION_CONSTRAINT']:
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        # import ipdb;ipdb.set_trace()
        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrix
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            tiouRecallMat = np.empty(outputShape)
            tiouPrecisionMat = np.empty(outputShape)
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
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum \
                            and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > evaluationParams['IOU_CONSTRAINT']:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            detMatched_tiouGt += tiouRecallMat[gtNum, detNum]
                            detMatched_tiouDt += tiouPrecisionMat[gtNum, detNum]
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)

        if evaluationParams['CONFIDENCES']:
            for detNum in range(len(detPols)):
                if detNum not in detDontCarePolsNum:
                    # we exclude the don't care detections
                    match = detNum in detMatchedNums

                    arrSampleConfidences.append(confidencesList[detNum])
                    arrSampleMatch.append(match)

                    arrGlobalConfidences.append(confidencesList[detNum])
                    arrGlobalMatches.append(match)

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))


        matchedSum += detMatched
        matchedSum_tiouGt += detMatched_tiouGt
        matchedSum_tiouDt += detMatched_tiouDt

        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
                methodRecall + methodPrecision)
    methodTagPrecision = {}

    methodRecall_tiouGt = 0 if numGlobalCareGt == 0 else float(matchedSum_tiouGt) / numGlobalCareGt
    methodPrecision_tiouDt = 0 if numGlobalCareDet == 0 else float(matchedSum_tiouDt) / numGlobalCareDet
    tiouMethodHmean = 0 if methodRecall_tiouGt + methodPrecision_tiouDt == 0 else \
        2 * methodRecall_tiouGt * methodPrecision_tiouDt / (methodRecall_tiouGt + methodPrecision_tiouDt)

    for tag in numGlobalCareTag:
        methodTagPrecision[tag] = None if numGlobalCareTag[tag] == 0 else float(matchedTag[tag])/numGlobalCareTag[tag]

    methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean, 'ave_precision': AP,
                     'tiouPrecision': methodPrecision_tiouDt, 'tiouRecall': methodRecall_tiouGt,
                     'tiouHmean': tiouMethodHmean,
                     "IOU_CONSTRAINT": evaluationParams['IOU_CONSTRAINT'],
                     "AREA_PRECISION_CONSTRAINT": evaluationParams['AREA_PRECISION_CONSTRAINT'],
                     "CONFIDENCES": evaluationParams['CONFIDENCES'],
                     }
    resDict = {'summary': methodMetrics}   
    return resDict
