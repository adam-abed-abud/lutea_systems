import datetime

from . import SupportMethods as SM
from . import PreProcessing as PPr
from . import BlobDetection as BD
from . import options as O

import cv2
import numpy as np
import os

import json
import pickle

PP = "main"

rulesListPP = []

def ppInit(tMin = 0, tMax = 255):
    tempRes = []
    
    for i in range(O.minBlurPP, O.maxBlurPP, O.stepBlurPP):
        for j in range(O.minThreshPP, O.maxThreshPP, O.stepThreshPP):
            if j >= tMin and j <= tMax:
                tempRes.append((i, j))

    return tempRes.copy()

def getPointsOnly(soruceDirName):    
    AllSourceImgs = SM.getImgsFromDir(soruceDirName)    

    if len(AllSourceImgs)<2:
        print(PP, "folder", soruceDirName, "EMPTY")
        return
    
    O.SHARED_NAME = os.path.basename(soruceDirName)

    if O.isNeedMinMaxCorrections:
        tMin, tMax = SM.getMinMaxVals(AllSourceImgs)
        rulesListPP = ppInit(tMin, tMax)
    else:
        rulesListPP = ppInit()
    
    SameBlobsMap = []
    for channelRule in O.CHANNELS: #Color channels processing
        Key, SameBlobsMap_Temp, _ = BD.getPairsRules_List_X(PPr.prepareAllImgs(AllSourceImgs, channelRule), rulesListPP)

        if not Key:
            print(PP, "getPairsRules", "FAIL")
            return

        SameBlobsMap.extend(SameBlobsMap_Temp)

    resList = []
    for tempList in SameBlobsMap:
        tempList_2  = [list(tempList[i][j] for j in [0,2,3,4,5,6]) for i in range(0, len(tempList), 2)]
        resList.append(tempList_2)
        SM.printListNewRow(tempList_2)
            
    if O.NeedSaveFiles:
        with open(O.SHARED_NAME + "_PairsList.txt", "wb") as fp:   #Pickling
            pickle.dump(resList, fp)

    return resList

if __name__ == "__main__":
    print(PrintPrefix, "Main Compiled")
    startTime = datetime.datetime.now()
    print(PrintPrefix, "Time", datetime.datetime.now() - startTime)
else:
    print(__name__, "Loaded")
