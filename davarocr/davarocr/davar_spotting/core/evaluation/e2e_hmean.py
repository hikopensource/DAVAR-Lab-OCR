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
        det_results(dict): the detection results predicted by model, in form of:
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
        gt_results(dict): ground truth for dataset, in form of:
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
            print("error occurs when calculate IOU")
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

    def transcription_match(transGt,
                            transDet,
                            specialCharacters='!?.:,*"()·[]/\'',
                            onlyRemoveFirstLastCharacterGT=True):
        """ Judge whether two transcriptions matches

        Args:
            transGt(str): text 1
            transDet(str): text 2
            specialCharacters(str): characters that not considered
            onlyRemoveFirstLastCharacterGT(boolean): Whether to remove first or Last character when comparing

        Returns:
            boolean: whether these two text matches
        """
        if onlyRemoveFirstLastCharacterGT:
            # special characters in GT are allowed only at initial or final position
            if (transGt == transDet):
                return True

            if len(transGt) >0 and specialCharacters.find(transGt[0]) > -1:
                if transGt[1:] == transDet:
                    return True

            if len(transGt) >0 and specialCharacters.find(transGt[-1]) > -1:
                if transGt[0:len(transGt) - 1] == transDet:
                    return True

            if len(transGt) >0 and specialCharacters.find(transGt[0]) > -1 and specialCharacters.find(transGt[-1]) > -1:
                if transGt[1:len(transGt) - 1] == transDet:
                    return True
            return False
        else:
            # Special characters are removed from the begining and the end of both Detection and GroundTruth
            while len(transGt) > 0 and specialCharacters.find(transGt[0]) > -1:
                transGt = transGt[1:]

            while len(transDet) > 0 and specialCharacters.find(transDet[0]) > -1:
                transDet = transDet[1:]

            while len(transGt) > 0 and specialCharacters.find(transGt[-1]) > -1:
                transGt = transGt[0:len(transGt) - 1]

            while len(transDet) > 0 and specialCharacters.find(transDet[-1]) > -1:
                transDet = transDet[0:len(transDet) - 1]

            return transGt == transDet

    def include_in_dictionary(transcription):
        """Function used in Word Spotting that finds if the Ground Truth transcription meets the rules to enter into
           the dictionary. If not, the transcription will be cared as don't care

        Args:
            transcription(str): predicted text
        Returns:
            boolean: whether it is in the dictionary
        """

        # special case 's at final
        if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
            transcription = transcription[0:len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        specialCharacters = evaluationParams['SPECIAL_CHARACTERS']
        for character in specialCharacters:
            transcription = transcription.replace(character, ' ')
        transcription = transcription.strip()

        if len(transcription) != len(transcription.replace(" ", "")):
            return False

        if len(transcription) < evaluationParams['MIN_LENGTH_CARE_WORD']:
            return False

        notAllowed = "×÷·"

        range1 = [ord(u'a'), ord(u'z')]
        range2 = [ord(u'A'), ord(u'Z')]
        range3 = [ord(u'À'), ord(u'ƿ')]
        range4 = [ord(u'Ǆ'), ord(u'ɿ')]
        range5 = [ord(u'Ά'), ord(u'Ͽ')]
        range6 = [ord(u'-'), ord(u'-')]

        for char in transcription:
            charCode = ord(char)
            if (notAllowed.find(char) != -1):
                return False

            valid = (charCode >= range1[0] and charCode <= range1[1]) or (
                    charCode >= range2[0] and charCode <= range2[1]) or (
                                charCode >= range3[0] and charCode <= range3[1]) or (
                            charCode >= range4[0] and charCode <= range4[1]) or (
                            charCode >= range5[0] and charCode <= range5[1]) or (
                            charCode >= range6[0] and charCode <= range6[1])
            if valid == False:
                return False

        return True

    def include_in_dictionary_transcription(transcription):
        """Function applied to the Ground Truth transcriptions used in Word Spotting. It removes special characters or
           terminations

        Args:
            transcription(str): predicted text
        Returns:
            boolean: whether it is in the dictionary
        """
        # special case 's at final
        if transcription[len(transcription) - 2:] == "'s" or transcription[len(transcription) - 2:] == "'S":
            transcription = transcription[0:len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip('-')

        specialCharacters = evaluationParams['SPECIAL_CHARACTERS']
        for character in specialCharacters:
            transcription = transcription.replace(character, ' ')

        transcription = transcription.strip()

        return transcription
    
    matchedSum_det = 0
    matchedSum_spot = 0

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    for gt_result, det_result in zip(gt_results, det_results):

        detMatched = 0
        detCorrect = 0
        gtPols = []
        detPols = []
        gtTrans = []
        detTrans = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = []  # Array of Ground Truth Polygons' keys marked as don't Care
        detDontCarePolsNum = []  # Array of Detected Polygons' matched with a don't Care GT
        detMatchedNums = []
        pairs = []

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

            # On word spotting we will filter some transcriptions with special characters
            if evaluationParams['WORD_SPOTTING']:
                if dontCare == False:
                    if include_in_dictionary(transcription) == False:
                        dontCare = True
                    else:
                        transcription = include_in_dictionary_transcription(transcription)
            gtTrans.append(transcription)

            if dontCare:
                gtDontCarePolsNum.append(len(gtPols)-1)

        # Process detection results
        pointsList = det_result['points']
        predictTexts = det_result['texts']
        for n in range(len(pointsList)):
            points = pointsList[n]
            detPol = polygon_from_points_any_shape(points)
            transcription = predictTexts[n]
            detPols.append(detPol)
            detPolPoints.append(points)
            detTrans.append(transcription)

            # If the det are is intersected with some NOTCARE area, then append it into DontCare list
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = detPol.area()
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT']):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum \
                            and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > evaluationParams['IOU_CONSTRAINT']:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            # detection matched only if transcription is equal
                            if evaluationParams['WORD_SPOTTING']:
                                correct = gtTrans[gtNum].upper() == detTrans[detNum].upper()
                            else:
                                correct = transcription_match(gtTrans[gtNum].upper(), detTrans[detNum].upper(),
                                                              evaluationParams['SPECIAL_CHARACTERS'],
                                                              evaluationParams[
                                                                  'ONLY_REMOVE_FIRST_LAST_CHARACTER']) == True
                            detCorrect += (1 if correct else 0)
                            if correct:
                                detMatchedNums.append(detNum)
                            pairs.append({'gt': gtNum, 'det': detNum, 'correct': correct})

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))

        matchedSum_det += detMatched
        matchedSum_spot += detCorrect
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

    det_recall = 0 if numGlobalCareGt == 0 else float(matchedSum_det) / numGlobalCareGt
    det_precision = 0 if numGlobalCareDet == 0 else float(matchedSum_det) / numGlobalCareDet
    det_hmean = 0 if det_recall + det_precision == 0 else 2 * det_recall * det_precision / (
    det_recall + det_precision)

    spot_recall = 0 if numGlobalCareGt == 0 else float(matchedSum_spot) / numGlobalCareGt
    spot_precision = 0 if numGlobalCareDet == 0 else float(matchedSum_spot) / numGlobalCareDet
    spot_hmean = 0 if spot_recall + spot_precision == 0 else 2 * spot_recall * spot_precision / (
    spot_recall + spot_precision)

    methodMetrics = {'det_precision':det_precision, 'det_recall':det_recall,'det_hmean': det_hmean,
                     'spot_precision':spot_precision, 'spot_recall':spot_recall,'spot_hmean': spot_hmean,
                     "IOU_CONSTRAINT": evaluationParams['IOU_CONSTRAINT'],
                     "AREA_PRECISION_CONSTRAINT": evaluationParams['AREA_PRECISION_CONSTRAINT']
                     }
    resDict = {'summary': methodMetrics}
    return resDict

