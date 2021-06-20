import numpy as np
import os
import datetime
import cv2
from . import SupportMethods as SM
from . import PreProcessing as PP
from . import options as O

import json #NTG

PrintPrefix = "BD"

def getCounterFromImage(SourceImage):
    ResTemp, _ = cv2.findContours(SourceImage, cv2.RETR_LIST, O.countersDetectionParam)   
    print(PrintPrefix, "gCFI", "Contours Count:", len(ResTemp))    
    return ResTemp

def getBlobsFromCounters(CountersList):
    ResTemp = []

    for i in range(0, len(CountersList)-1):
        t_area = cv2.contourArea(CountersList[i])
        
        if (t_area > O.MinCounterArea) and (t_area < O.MaxCounterArea):
            
            tt_min = 2
            for j in range(i+1, len(CountersList)):
                tt = cv2.matchShapes(CountersList[i], CountersList[j], O.matchShapeParam, 0.0)
                if tt < tt_min:
                    tt_min = tt
                    
            if tt_min > O.ThreshContourCompareSelf:
                if O.NeedRectangles:
                    rect = cv2.minAreaRect(CountersList[i])
                    approx = cv2.boxPoints(rect)
                    ResTemp.append(approx)
                    
                if O.NeedConvexHulls:
                    approx = cv2.convexHull(CountersList[i])
                    ResTemp.append(approx)
                        
                if O.NeedPolygons:
                    epsilon = O.epsilonApproxParam * cv2.arcLength(CountersList[i], True)
                    approx = cv2.approxPolyDP(CountersList[i], epsilon, True)
                    ResTemp.append(approx)
    
    print(PrintPrefix, "gBFI", "Blobs Count:", len(ResTemp))
    return ResTemp

def getSameBlobs(AllBlobs_MD):
    allSize = len(AllBlobs_MD)
    ResTemp = [[] for _ in range(allSize)]
    blobsCounter = 0 #    
    for i in range(allSize):
        ResTemp[i] = [[] for _ in range(min(O.FutureSameImagesCount, allSize - i - 1))]
        for ii in range(len(AllBlobs_MD[i])):
            tempBlob = AllBlobs_MD[i][ii]
            tempBlobArc = cv2.arcLength(tempBlob, True)
            for k in range(i + 1, min(i + 1 + O.FutureSameImagesCount, allSize)):  
                for j in range(len(AllBlobs_MD[k])):
                    tempBlobKJ = AllBlobs_MD[k][j]
                    tempSelf = cv2.matchShapes(tempBlob, tempBlobKJ, O.matchShapeParam, 0.0)
                    if tempSelf < O.ThreshContourCompareTwo:
                        tempArc = abs((2 * tempBlobArc) / (tempBlobArc + cv2.arcLength(tempBlobKJ, True)) - 1)
                        if tempArc < O.ArcDelta:
                            tempSelf = cv2.matchShapes(tempBlobKJ, tempBlob, O.matchShapeParam, 0.0)                            
                            if tempSelf < O.ThreshContourCompareTwo:
                                x1, y1 = SM.getContourCenter(tempBlob)
                                x2, y2 = SM.getContourCenter(tempBlobKJ)
                                ResTemp[i][k-i-1].append(((x1, y1), (x2, y2)))
                                blobsCounter += 1 #

    print(PrintPrefix, "gSB", "Same Blobs Count:", blobsCounter)    
    return ResTemp, blobsCounter

def getPairsRules_List_X(AllSourceImgs, rulesListPP):
    #print(PP, rulesListPP)
    SameBlobsMap = []

    #REGION OF JSON START
    if O.NeedPrintJSON:
        PP_LIST_RESULT = [] #NTG
        O.SHARED_LIST = [] #NTG
    #REGION OF JSON END

    time_PP = 0
    tempBlurVal = -1
    tempBlurImgs = []
    
    for PPTemp in rulesListPP:
        AllPPImgs = []        
        
        if PPTemp == "Auto":
            Key, AllPPImgs = PP.getAllImagesPreProcessedAuto(AllSourceImgs, int(np.mean(AllSourceImgs)))
            if not Key:
                print(PrintPrefix, "getAllImagesPreProcessedAuto", PPTemp, "FAIL")
                continue
        else:
            startTime = datetime.datetime.now() #Time
            if (tempBlurVal != PPTemp[0]): #x2 Memory Here
                tempBlurImgs = PP.getAllImagesBlurred(AllSourceImgs, PPTemp[0])
                tempBlurVal = PPTemp[0]
            AllPPImgs = PP.getAllImagesThresh(tempBlurImgs, PPTemp[1]) #x3 Memory Here
            time_PP = datetime.datetime.now() - startTime #Time
            
        tempBlobs = []
        
        for tempImg in AllPPImgs:
            tempBlobs.append(getBlobsFromCounters(getCounterFromImage(tempImg)))
            print(len(tempBlobs[-1]))

        tempResBlobs, tempBlobsCount = getSameBlobs(tempBlobs)
        SameBlobsMap.append(tempResBlobs.copy())

        #REGION OF JSON START
        if O.NeedPrintJSON:
            O.SHARED_LIST.append({
                "PP_Blur": PPTemp[0],
                "PP_Threshold": PPTemp[1],
                "BlobsCount": 0
                })
                    
            PP_LIST_RESULT.append({
                "PP_Blur": PPTemp[0],
                "PP_Threshold": PPTemp[1],
                "BlobsCount": tempBlobsCount
                })
        #REGION OF JSON END
    
    AllPPImgs = []
    tempBlurImgs = []
    tempResBlobs = None

    #REGION OF JSON START
    if O.NeedPrintJSON:
        with open(os.path.join(O.SAVEPATH, O.SHARED_NAME + '_ALL_Counters.json'), 'w') as outfile:
            json.dump(PP_LIST_RESULT, outfile)

        SM.SaveJsonAsCSV(PP_LIST_RESULT, os.path.join(O.SAVEPATH, O.SHARED_NAME + '_ALL_Counters'))
    #REGION OF JSON END
    
    return True, SameBlobsMap, time_PP

if __name__ == "__main__":
    print(PrintPrefix, "Main Compiled")
else:
    print(__name__, "Loaded")
