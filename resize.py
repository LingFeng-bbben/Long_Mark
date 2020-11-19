import cv2 as cv
import os

fileList = os.listdir("./imgcroped")

imgcount = 0

for file in fileList:
    imgPath = "imgcroped/" + file
    img = cv.imread(imgPath)
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_resized = cv.resize(img_grey,(32,32))
    cv.imwrite("./imgresized/"+ str(imgcount) +".jpg",img_resized)
    imgcount += 1