import cv2
import datetime
import numpy as np
from . import options as O
from . import SupportMethods as SM

PrintPrefix = "PP"

def imagePreProcess(img, BlurSize, ThresholdValue):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (BlurSize, BlurSize), 0)
        rev, img = cv2.threshold(img, ThresholdValue, 255, cv2.THRESH_BINARY)
        img = cv2.Canny(img, 100, 300)
    except Exception as e:
        print(PrintPrefix, e)
        print("ERROR", PrintPrefix, "iPP", "BlurSize", BlurSize, "ThresholdValue", ThresholdValue)

    return img

def getAllImagesPreProcessed(imagesList, BlurSize, ThresholdValue):
    ResTemp = []
    
    for tempImage in imagesList:
        tempImg = imagePreProcess(tempImage, BlurSize, ThresholdValue)
        ResTemp.append(tempImg)

    print(PrintPrefix, "Images preProcessed", len(ResTemp), "BlurSize", BlurSize, "ThresholdValue", ThresholdValue)
    return ResTemp

def prepareAllImgs(imagesList, channelRule = O.CHANNELS[0]):
    ResTemp = []
    
    for img in imagesList:
            
        if channelRule == "all":
            img2 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) #NTG
        elif channelRule == "red":
             img2 = img[:,:,[2]].copy() #NTG
        elif channelRule == "green":
            img2 = img[:,:,[1]].copy() #NTG
        elif channelRule == "blue":
            img2 = img[:,:,[0]].copy() #NTG
        else:
            img2 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) #NTG                
        if O.isNeedEqualizeHist:
            img2 = cv2.equalizeHist(img2) #NTG

        ResTemp.append(img2)

    print(PrintPrefix, "pAI", "Done")
    return ResTemp   

def imageBlur(img, BlurSize):
    try:
        #(channel_b, channel_g, channel_r) = cv2.split(img)
        img2 = cv2.GaussianBlur(img.copy(), (BlurSize, BlurSize), 0)        
        return img2
    except Exception as e:
        print(PrintPrefix, e)
        print("ERROR", PrintPrefix, "iB", "BlurSize", BlurSize)

    return img

def getAllImagesBlurred(imagesList, BlurSize):
    ResTemp = []
    
    for tempImage in imagesList:
        tempImg = imageBlur(tempImage, BlurSize)
        ResTemp.append(tempImg)

    print(PrintPrefix, "gAIB", len(ResTemp), "BlurSize", BlurSize)
    return ResTemp


def imageThresh(img, ThresholdValue):
    try:
        rev, img2 = cv2.threshold(img.copy(), ThresholdValue, 255, cv2.THRESH_BINARY)
        img2 = cv2.Canny(img2, 100, 300)
    except Exception as e:
        print(PrintPrefix, e)
        print("ERROR", PrintPrefix, "iT",  "ThresholdValue", ThresholdValue)

    return img2

def getAllImagesThresh(imagesList, ThresholdValue):
    ResTemp = []
    
    for tempImage in imagesList:
        tempImg = imageThresh(tempImage, ThresholdValue)
        ResTemp.append(tempImg)

    print(PrintPrefix, "gAIT", len(ResTemp), "ThresholdValue", ThresholdValue)
    return ResTemp

def imagePreProcessAutoThresh(img, BlurSize, ThresholdValue = -1):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (BlurSize, BlurSize), 0)
        if ThresholdValue>0:
            rev, img = cv2.threshold(img, ThresholdValue, 255, cv2.THRESH_BINARY)
        else:
            rev, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.Canny(img, 100, 300)
    except Exception as e:
        print(PrintPrefix, e)
        print("ERROR", PrintPrefix, "iPPAT", "BlurSize", BlurSize)

    return img

def getAllImagesPreProcessedAutoThresh(imagesList, BlurSize, ThresholdValue = -1):
    ResTemp = []
    
    for tempImage in imagesList:
        tempImg = imagePreProcessAutoThresh(tempImage, BlurSize, ThresholdValue)
        ResTemp.append(tempImg)

    print(PrintPrefix, "gAIPPAT", "BlurSize", BlurSize, "Images preProcessed", len(ResTemp))
    return ResTemp

def getAllImagesPreProcessedAuto(imagesList, ThresholdValue = -1):
    for blur in range(O.minBlur, O.maxBlur, O.stepBlur):
        BlurSize = 2 * blur + 1
        
        imgs = getAllImagesPreProcessedAutoThresh(imagesList, BlurSize, ThresholdValue)
        res = getCountersInfo(imgs)
        print(PrintPrefix, "gIPPA", res)

        if res[1]>O.areasMinCount and res[2]>O.areasMinArea:
            print(PrintPrefix, "gIPPA", "Blur Founded", BlurSize)
            return True, imgs
        
    print(PrintPrefix, "gIPPA", "Blur Founded FAIL")
    return False, imgs
    
def getCountersInfo(imagesList):    
    tempList = [[],[],[],[],[]]
    
    for img in imagesList:
        ResTemp, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

        ListTemp = []
        for i in range(len(ResTemp)):
            ListTemp.append(cv2.contourArea(ResTemp[i]))
        ListTemp = np.array(ListTemp)

        tempList[0].append(0)
        tempList[1].append(len(ResTemp))
        tempList[2].append(np.mean(ListTemp))
        tempList[3].append(np.sum(ListTemp) / img.size)

    resList = []
    for i in range(4):
        resList.append(np.mean(np.array(tempList[i])))
        
    resList.append(resList[3] / resList[1])
    return resList

def saveCountersDataByFolder(soruceDirName):
    ResList = [["BlurSize:"] , ["Contours Count:"], ["Contours AVG Area:"], ["Area Percent:"], ["AVG Percent"]]    
    filenames = SM.getFilesFromDir(soruceDirName)
    
    for i in range (O.minBlur, O.maxBlur, O.stepBlur):
        BlurSize = i*2+1
        imgList = []
        
        for temp_f0 in filenames:
            tempImg = cv2.imread(temp_f0)
            imgList.append(imagePreProcessAutoThresh(tempImg, BlurSize))

        tempRes = getCountersInfo(imgList)
        
        ResList[0].append(BlurSize)
        for i in range(1, len(tempRes)):
            ResList[i].append(tempRes[i])
            
    np.savetxt("ResList3_"+str(soruceDirName)+".csv", np.swapaxes(ResList, 0, 1), delimiter=";", fmt='%s') #

def mainTest():
    soruceDirName = "30"
    targetDirName = "30_Out"

    filenames = SM.getFilesFromDir(soruceDirName)

    sourceImagesList = SM.getImgsFromDir(soruceDirName)

    key, resImagesList = getAllImagesPreProcessedAuto(sourceImagesList, int(np.mean(sourceImagesList[0])))

    if key:
        tempDirName = targetDirName + "//"
        SM.createDir(tempDirName) 
        for i in range(len(resImagesList)):
            cv2.imwrite(tempDirName + SM.getBaseName(filenames[i])+"_Up.png", resImagesList[i])
                
    print("Done")
    
if __name__ == "__main__":
    print(PrintPrefix, "Main Compiled")
    startTime = datetime.datetime.now()
    #saveCountersDataByFolder("old")
    #mainTest()
    print(PrintPrefix, "Time", datetime.datetime.now() - startTime)
else:
    print(__name__, "Loaded")
