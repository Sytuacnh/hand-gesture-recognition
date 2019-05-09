# 10 people
# each 10 class
# target: 1000 samples
# 10 people
# each class 10 images

import cv2
import os, random, shutil
import glob as gb
import numpy as np

star = '/*'
tarDir = '/Users/sytu/Desktop/data_project/output_dataset'
pickNumber = 10

threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

def pick_by_number(fileDir, tarDir, pickNumber, labelName):
    pathDir = os.listdir(fileDir) # all files in fileDir
    # percentage
    # filenumber = len(pathDir) # number of files
    # rate = 0.1    # set percentage
    # picknumber = int(filenumber*rate) # select number based on percentage
    img_samples = random.sample(pathDir, pickNumber)  # randomly pick pickNumber files
    assert len(img_samples) is pickNumber, 'length does not match'
    for img_name in img_samples:
        imgPath = fileDir + '/' + img_name
        destPath = tarDir + '/' + labelName + '-' + img_name
        bit_n_save_image(imgPath, destPath)
        # shutil.move(fileDir+'/'+name, tarDir+'/'+labelName+'-'+name) # directly copy to destination


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
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res


def bit_n_save_image(imgPath, savePath):
    img = cv2.imread(imgPath)
    img = cv2.bilateralFilter(img, 5, 50, 100)  # smoothing
    img = remove_background(img)
    img = bit_contour(img)
    cv2.imwrite(savePath,img,[int(cv2.IMWRITE_JPEG_QUALITY),100])


def main():
    samples = gb.glob("/Users/sytu/Desktop/data_project/leapGestRecog/*")
    # select and move image
    for sample in samples:
        for labelDir in gb.glob(sample+star):
            labelName = os.path.basename(labelDir) # folder(label) name
            pick_by_number(labelDir, tarDir, pickNumber, labelName)

    print("Finished images selection and conversion.")
    print("Image size: " + str(len(gb.glob(tarDir+star))))

if __name__ == '__main__':
    main()