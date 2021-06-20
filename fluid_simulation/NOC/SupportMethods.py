import cv2
import os
import math
import numpy as np
from . import options as O

PrintPrefix = "SM"

def clearFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(PrintPrefix, e)
            
def createDir(DirName):
    DirName = os.path.dirname(DirName)
    if not os.path.exists(DirName):
        os.makedirs(DirName)

def getFilesFromDir(soruceDirName):
    resList = []
    counter = 0
    for (dirpath, dirnames, filenames) in os.walk(soruceDirName):
        for filename in filenames:
            counter += 1
            if counter%1 == 0:
                fileType = filename[filename.rfind('.')+1:].lower()
                if fileType in O.ImageFilesList:         
                    resList.append(os.path.join(dirpath, filename))

    return sorted(resList)

def getImgsFromDir(soruceDirName):
    filenames = getFilesFromDir(soruceDirName)

    resList = []
    for temp_f0 in filenames:
        img = cv2.imread(temp_f0)
        try:
            h, w, n = img.shape            
            resList.append(img)
            print(PrintPrefix, "gIFD", "Image Name:", temp_f0)
        except:
            print(PrintPrefix, "gIFD", "Image Name:", temp_f0, "ERROR")
            pass

    return resList

def getBaseName(SourceName):
    return os.path.basename(SourceName)

def printListNewRow(SourceList):
    print('\n'.join([ str(myelement) for myelement in SourceList ]))
    
def resizeImage(sourceImage):
    shapes = sourceImage.shape
    if shapes[1] >= O.newX and shapes[0] >= O.newY:
        return sourceImage
    
    sourceImage = np.insert(sourceImage, [0 for j in range(O.offX)], [0], axis=1)
    sourceImage = np.insert(sourceImage, [0 for j in range(O.offY)], [0], axis=0)
    
    shapes = sourceImage.shape
        
    sourceImage = np.insert(sourceImage, [shapes[1] for i in range(shapes[1], O.newX)], [0], axis=1)
    sourceImage = np.insert(sourceImage, [shapes[0] for i in range(shapes[0], O.newY)], [0], axis=0)
    
    return sourceImage

def resizeImage_Paste(sourceImage, flatImage):
    if sourceImage.shape == flatImage.shape:
        flatImage = None
        return sourceImage
    
    w, h, _ = sourceImage.shape 
    flatImage[O.offX:O.offX + w, O.offY:O.offY + h, :] = sourceImage
    return flatImage

def shiftImage_Paste(sourceImage):
    w, h, _ = sourceImage.shape    
    flatImage = np.zeros((O.offX + w, O.offY + h, 3), dtype=np.uint8)     
    flatImage[O.offX:O.offX + w, O.offY:O.offY + h, :] = sourceImage
    return flatImage

def removeBorders(sourceImg):
    print(PrintPrefix, "rB", "Before", sourceImg.shape)

    resMask = cv2.inRange(sourceImg, np.array((1., 1., 1.)), np.array((255., 255., 255.)))
    mask = resMask > 0
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    sourceImg = sourceImg[x0:x1, y0:y1, :]
    
    print(PrintPrefix, "rB", "After", sourceImg.shape)    
    return sourceImg

def resizeImagesList(sourceImagesList):
    resList = []
    for tempImg in sourceImagesList:
        resList.append(resizeImage(tempImg))

    return resList

def initSizes(AllSourceImgs):
    maxW = 0
    maxH = 0
    allW = 0
    allH = 0
   
    for tempImg in AllSourceImgs:
        #print(PrintPrefix, "Shape", np.shape(tempImg))
        tempH, tempW, _ = np.shape(tempImg)
        allW += tempW/2
        allH += tempH/2

        if tempW > maxW:
            maxW = tempW
            
        if tempH > maxH:
            maxH = tempH

    allW+=maxW
    allH+=maxH

    O.offX = int(maxW)
    O.offY = int(maxH)

    O.newX = int(allW)
    O.newY = int(allH)

def getContourCenter(sourceCountor):
    M = cv2.moments(sourceCountor)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except Exception as e:
        cX = cY = 0
        
    return cX, cY

#Histogram for comapare
def GetHist(image_0):
    return cv2.calcHist([image_0], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])

#Comparison of two images (cv2)
def GetDifference(image_1, image_2):
    first_image_hist = GetHist(image_1)
    second_image_hist = GetHist(image_2)
    img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
    first_image_hist = None
    second_image_hist = None
    return img_hist_diff

#Вычисление расстояния
def p1p2Dist(px1, py1, px2, py2):
    return math.sqrt((px1 - px2) * (px1 - px2) + (py1 - py2) * (py1 - py2))

#Вычисление расстояния
def p12Dist(p1, p2):
    return p1p2Dist(p1[0], p1[1], p2[0], p2[1])

def SaveJsonAsCSV(data, saveName):
    PP_Blur_List = []
    PP_Threshold_List = []
    for tempVal in data:
        if tempVal['PP_Blur'] not in PP_Blur_List:
            PP_Blur_List.append(tempVal['PP_Blur'])
        if tempVal['PP_Threshold'] not in PP_Threshold_List:
            PP_Threshold_List.append(tempVal['PP_Threshold'])

    resTable = [[[] for x in range(len(PP_Blur_List))] for x in range(len(PP_Threshold_List))] 

    for tempVal in data:
        resTable[PP_Threshold_List.index(tempVal['PP_Threshold'])][PP_Blur_List.index(tempVal['PP_Blur'])] = tempVal['BlobsCount']

    PP_Threshold_List2 = PP_Threshold_List
    PP_Threshold_List2.insert(0, "")

    resTable2 = resTable
    resTable2.insert(0, PP_Blur_List)

    resTable3 = resTable2
    for i in range(len(resTable3)):
        resTable3[i].insert(0, PP_Threshold_List2[i])
        
    np.savetxt(saveName + ".csv", resTable3, delimiter=";", fmt='%s')

def getMinMaxVals(AllSourceImgs):
    minArr = []
    maxArr = []
    for tempImg in AllSourceImgs:
        minArr.append(np.min(tempImg))
        maxArr.append(np.max(tempImg))
        
    return np.min(minArr), np.max(maxArr)
    
if __name__ == "__main__":
    print(PrintPrefix, "Main Loaded")
else:
    print(__name__, "Loaded")
