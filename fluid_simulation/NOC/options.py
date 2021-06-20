import cv2

#General
NeedOneBigImage = True

NNBreaks = 0 #Maximal amount of breaks in Graph. = 0 For One Big Image
GraphBreaks = 0 #Maximal amount of bad stiteches. = 0 For One Big Image
MinImgsCount = 0 #Minimal amount of images in serie

IsNeedPoolGraph = False
PROCNUMBER = 4
PROCQOEF = 16  #Minimal amount of stitches to be Pooled. PROCNUMBER*4 Recomended 

KeyPointsDetector = "BIM"

#BIM
CHANNELS = ["all"] #all, red, green, blue, 

isNeedEqualizeHist = False
isNeedMinMaxCorrections = True #Uses for BF params correction

minBlurPP = 9 #1
maxBlurPP = 45 #45
stepBlurPP = 2 #4

minThreshPP = 20 #10 110 70
maxThreshPP = 135 #250 235 170
stepThreshPP = 5 #25

MinCounterArea = 1000 #Minimal Square for areas Filtering
MaxCounterArea = 1000000 #Maximal Square for areas Filtering

#Contour approximation style Only Polygons RECOMENDED
NeedPolygons = True
NeedRectangles = False
NeedConvexHulls = False

ThreshContourCompareSelf = 0.04 #0.04
ThreshContourCompareTwo = 0.15 #0.15
ArcDelta = 0.2 #0.1 Percent for Counters perimeter

matchShapeParam = 3 #3
epsilonApproxParam = 0.005 #0.005
# https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
countersDetectionParam = cv2.CHAIN_APPROX_TC89_L1 #cv2.CHAIN_APPROX_TC89_L1

#Params
FutureSameImagesCount = 100 #Amount of frames of view (less - faster)
minNeighboursCount = 4 #Minimal amount of Blobs to Images Union

BF_Count = 1 #Amount of Shuffling. 1 - MINIMAL
WARP = "warpA" #warpA - Affine; warpP - Homography

#Stitching
offX = 5000 #4000
offY = 5000 #4000

newX = 15000 #10000
newY = 15000 #10000

#Elements of H matrix Analyzis
MatrixH00 = 0.15
MatrixH11 = 0.15
MatrixH_XX = 0.01 # 0.01
MatrixH_X = 0.02 #0.02 # 0.015 #T7 CURRENTLY USED

MinPixelDistance = 10 #10 #Distance between pixels Filters before RANSAC

#Orher params
ImageFilesList = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']

#DEBUG PRINT REGION True False
NeedPrintJSON = True
NeedSaveFiles = True
UseSavedFiles = False

SHARED_LIST = []
SHARED_NAME = 'SHARED_NAME'

SWAP = 0
SAVEPATH = ""

if __name__ == "__main__":
    print("Just Options File")
else:
    print(__name__, "Loaded")
