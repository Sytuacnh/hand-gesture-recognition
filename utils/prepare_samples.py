# v0
# 14 people
# each 10 classes(gestures)
# each class has 10 samples
# each selected sample: r_rgb.png: Kinect color map (1280 x 960).

# target: 560 samples
# 4 classes: 
#   fist, peace, love, five
#   ['G1', 'G5', 'G8', 'G9']
# each class 140 samples
# 10 images from each class of each person

import cv2
import os, random, shutil
import glob as gb
import numpy as np

targetLabels = set(['G1', 'G5', 'G8', 'G9'])
datasetPath = '/Users/sytu/Desktop/data_project/kinect_leap_dataset/acquisitions'
tarDir = '/Users/sytu/Desktop/data_project/output_dataset'
star = '/*'
pickByPicType = True
pickNumber = 1
picType = 'rgb'

bitFlag = False
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

def pick_img(imgDir, tarDir, pickNumber, labelName, pIndex):
    allFiles = os.listdir(imgDir)
    if pickByPicType:
        # print(allFiles)
        img_samples = [p for p in allFiles if picType in p]
        # print(len(img_samples))
    else:
        img_samples = random.sample(allFiles, pickNumber)
        assert len(img_samples) is pickNumber, 'number does not match'

    for img_name in img_samples:
        imgPath = imgDir + '/' + img_name
        destPath = tarDir + '/' + labelName + '-' + pIndex + '-' + img_name
        if bitFlag:
            bit_n_save_image(imgPath, destPath) # convert binary image and save
        else:
            shutil.copyfile(imgPath, destPath) # directly copy to destination


def remove_background(img):
    fgmask = bgModel.apply(img, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=fgmask)
    return res


# contour extraction and Binarization 
def bit_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    ret, res = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res


def bit_n_save_image(imgPath, savePath):
    img = cv2.imread(imgPath)
    img = cv2.bilateralFilter(img, 5, 50, 100)  # smoothing
    img = remove_background(img)
    img = bit_contour(img)
    cv2.imwrite(savePath,img,[int(cv2.IMWRITE_JPEG_QUALITY),100])


def main():
    samples = gb.glob(datasetPath + star) # plus os.path.sep skip files and keeps only folders if in the root directory instead of the acquisitions folder

    for sample in samples:
        pIndex = os.path.basename(sample) # person index
        for labelDir in gb.glob(sample+star):
            labelName = os.path.basename(labelDir) # folder(label) name
            if labelName in targetLabels:
                pick_img(labelDir, tarDir, pickNumber, labelName, pIndex)

    print("\nFinished images selection and conversion.")
    print("Image size: " + str(len(gb.glob(tarDir+star)))) # should print 560


if __name__ == '__main__':
    main()