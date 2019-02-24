import numpy as np
import re
import copy
import csv
from io import StringIO
import argparse
import os
import glob
import sys

# call with python3 pamonoDataToYolo.py --csvFileName=NanoSynthMLPolygonFormFactors.csv --outputName=OUT --imageWidth=1080 --imageHeight=145

def main():
    parser = argparse.ArgumentParser(description='CSV file name')
    parser.add_argument('--csvFileName', type=str, default=None, help='No help.')
    parser.add_argument('--outputName', type=str, default=None, help='No help.')
    parser.add_argument('--imageWidth', type=int, default=None, help='No help.')
    parser.add_argument('--imageHeight', type=int, default=None, help='No help.')
    args = parser.parse_args()
    csvName = args.csvFileName
    outName = args.outputName
    img_w = args.imageWidth
    img_h = args.imageHeight

    print("\nArguments:")
    print(args)

    data = open(csvName,"r").readlines()
    YoloDataS = np.array(getDataForYolo(data))
    YoloDataN = YoloDataS.astype(np.float)
    YoloDataRounded = getRoundedValues(YoloDataN)


    YoloDataV2Format = np.array(createDarknetYOLOv2Format(YoloDataRounded, img_w, img_h))
    saveDataToDisk(YoloDataV2Format, outName)

def getDataForYolo(examples):
    allExamples = []
    # iterate every entry
    for index, line in enumerate(examples):

        # there is no data in the first line, skip
        if (index == 0):
            continue

        f = StringIO(line)
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            #print('\t'.join(row))
            # only take x,y,w,h
            allExamples.append(row[4:8])

    return allExamples

def getRoundedValues(YoloDataN):
    firstDim = YoloDataN.shape[0]
    secondDim = YoloDataN.shape[1]

    # convert all floats to ints by rounding
    for i in range(0, firstDim):
        for j in range(0, secondDim):
            YoloDataN[i][j] = int(round(YoloDataN[i][j]))

    YoloDataRounded = YoloDataN.astype(np.int32)

    return YoloDataRounded

def createDarknetYOLOv2Format(YoloDataRounded, img_w, img_h):
    # our current format  [object top left X] [object top left Y] [object width in X] [object height in Y]
    # YOLOv2 format [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]
    # <object-class> <x> <y> <width> <height> in each line for every annotation

    # center in x = round(lX + 0.5*W)
    # center in y = round(lY + round0.5*H)

    # yoloW = W / imageW
    # yoloH = H / imageH

    firstDim = YoloDataRounded.shape[0]

    # convert all floats to ints by rounding
    yolov2Set = []
    for i in range(0, firstDim):
        buffer = []

        # left top X absolute value
        ltX = YoloDataRounded[i][0]
        # left top Y absolute value
        ltY = YoloDataRounded[i][1]
        # w of bounding box annotation absolute
        absW = YoloDataRounded[i][2]
        # h of bounding box annotation absolute
        absH = YoloDataRounded[i][3]

        # yolov2 wants relative values
        dw = 1./img_w
        dh = 1./img_h

        # calculate center coordinates of bounding box relative to image dimensions
        centerXrel = (ltX + 0.5 * absW) * dw
        centerYrel = (ltY + 0.5 * absH) * dh
        buffer.append(centerXrel)
        buffer.append(centerYrel)

        # calculate width and height of bounding box relative to image dimensions
        relW = absW * dw
        relH = absH * dh
        buffer.append(relW)
        buffer.append(relH)

        yolov2Set.append(buffer)
    return yolov2Set

def saveDataToDisk(set, outName):

    DataString = ""
    firstDim = set.shape[0]
    for i in range(0, firstDim):
        DataString = DataString + "0 " + str(set[i][0]) + " " + str(set[i][1]) + " " + str(set[i][2]) + " " + str(set[i][3]) + "\n"

    # save to disk
    outName = outName + ".txt"
    text_file = open(outName, "w")
    text_file.write(DataString)
    text_file.close()

if __name__=="__main__":
    main()
