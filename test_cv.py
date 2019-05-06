import cv2
import numpy as np

# remove background and Binarization
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# contour extraction and Binarization 
def bit_contour(frame):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    ret, res = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res

img = cv2.imread("/Users/sytu/Desktop/data_project/leapGestRecog/04/05_thumb/frame_04_05_0005.png")
img = cv2.bilateralFilter(img, 5, 50, 100)  # smoothing
img = remove_background(img)
img = bit_contour(img)

while True:
    cv2.imshow('img', img)

    if cv2.waitKey(1) is ord('q'):
        break
# testing kaggle dataset
# img = cv2.imread("/Users/sytu/Desktop/data_project/leapGestRecog/06/10_down/frame_06_10_0012.png")

# contour extraction
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 2)
# th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# while True:
#     cv2.imshow('img', res)

#     if cv2.waitKey(1) is ord('q'):
#         break
# face, eyes detection demo
# face_cascade = cv2.CascadeClassifier('/Applications/anaconda3/envs/hand/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/Applications/anaconda3/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')

# capture = cv2.VideoCapture(0)

# while True:
#     ret, img = capture.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#     cv2.imshow('img',img)

#     if cv2.waitKey(1) is ord('q'):
#         break

cv2.destroyAllWindows()
