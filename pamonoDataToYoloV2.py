import numpy as np
import re
import copy
import csv
from io import StringIO
import argparse
import os
import glob
import sys
from shutil import copyfile

# copy this file in the folder with the images and the annotation file and run it from there

# call with python3 pamonoDataToYolo.py --csvFileName=NanoSynthMLPolygonFormFactors.csv --imageWidth=1080 --imageHeight=145 --prefix=imagesAndAnnotations


def main():
    parser = argparse.ArgumentParser(description='CSV file name')
    parser.add_argument('--csvFileName', type=str, default=None, help='No help.')
    parser.add_argument('--imageWidth', type=int, default=None, help='No help.')
    parser.add_argument('--imageHeight', type=int, default=None, help='No help.')
    parser.add_argument('--prefixDataPath', type=str, default=None, help='No help.')
    args = parser.parse_args()
    csvName = args.csvFileName
    img_w = args.imageWidth
    img_h = args.imageHeight
    prefixDataPath = args.prefixDataPath

    print("\nArguments:")
    print(args)

    # create folder where Images and annotations should be stored
    os.makedirs(prefixDataPath)

    # open csv
    data = open(csvName,"r").readlines()

    # create everything in the folder specified in prefixDataPath
    allImageNames = createDataForYolo(data, img_w, img_h, prefixDataPath)

    # copy all images from this folder to the one in prefixDataPath
    copyAllImages(allImageNames, prefixDataPath)

def createDataForYolo(examples, img_w, img_h, prefixDataPath):
    # example stack
    allExamples = []
    # image file stack
    allImageFiles = []
    # iterate every entry
    for index, line in enumerate(examples):

        # there is no data in the first line, skip
        if (index == 0):
            continue

        f = StringIO(line)
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            #print('\t'.join(row))
            # strip path from image filenames, only keep names
            currentName = os.path.basename(row[2])
            # load this image and copy it to prefixDataPath
            # TODO:
            allImageFiles.append(currentName)

            # only take x,y,w,h
            allExamples.append(row[4:8])
            # calculate Darknet format for currently loaded lines
            # for every annotation, create a new annotation file
            # the name of the annotation file has to be the name of the image
            saveAs = os.path.join(prefixDataPath, currentName)
            createDarknetYOLOv2Format(allExamples, img_w, img_h, saveAs)
    return allImageFiles

def getRoundedValues(YoloDataN):
    firstDim = YoloDataN.shape[0]
    secondDim = YoloDataN.shape[1]

    # convert all floats to ints by rounding
    for i in range(0, firstDim):
        for j in range(0, secondDim):
            YoloDataN[i][j] = int(round(YoloDataN[i][j]))

    YoloDataRounded = YoloDataN.astype(np.int32)

    return YoloDataRounded

def createDarknetYOLOv2Format(YoloDataRounded, img_w, img_h, saveAs):
    # our current format  [object top left X] [object top left Y] [object width in X] [object height in Y]
    # YOLOv2 format [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]
    # <object-class> <x> <y> <width> <height> in each line for every annotation

    # center in x = round(lX + 0.5*W)
    # center in y = round(lY + round0.5*H)

    # yoloW = W / imageW
    # yoloH = H / imageH

    # need values now, not strings
    YoloDataRounded = np.array(YoloDataRounded)
    YoloDataRounded = YoloDataRounded.astype(np.float)
    firstDim = YoloDataRounded.shape[0]

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
    yolov2Set = np.array(yolov2Set)
    saveDataToDisk(yolov2Set, saveAs)

def saveDataToDisk(set, outName):
    DataString = ""
    firstDim = set.shape[0]
    for i in range(0, firstDim):
        DataString = DataString + "0 " + str(set[i][0]) + " " + str(set[i][1]) + " " + str(set[i][2]) + " " + str(set[i][3]) + "\n"

    # save to disk
    # remove ".png", other methods remove the path, find better one
    outName = outName[:-4]
    outName = outName + ".txt"

    try:
        annotationFile = open(outName, 'w')
        # it already exists, so append only last line - just overwrite
        #toAppend = "0 " + str(set[-1][0]) + " " + str(set[-1][1]) + " " + str(set[-1][2]) + " " + str(set[-1][3]) + "\n"
        annotationFile.write(DataString)
        annotationFile.close()
    except IOError:
        # If not exists, create the file, and write everything
        annotationFile = open(outName, 'w+')
        annotationFile.write(DataString)
        annotationFile.close()

def copyAllImages(allImageNames, prefixDataPath):
    for imageName in allImageNames:
        destpathPos = os.path.join(prefixDataPath, imageName)
        copyfile(imageName, destpathPos)

if __name__=="__main__":
    main()
