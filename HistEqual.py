import cv2 as cv
import os

fileList = os.listdir("./imgresized")

imgcount = 0

for file in fileList:
    imgPath = "imgresized/" + file
    img = cv.imread(imgPath)
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_denose = cv.fastNlMeansDenoising(img_grey)
    img_eqed = cv.equalizeHist(img_grey)
    cv.imwrite("./imgequal/"+ str(imgcount) +".jpg",img_eqed)
    imgcount += 1