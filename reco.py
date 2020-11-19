import cv2 as cv
import os

fileList = os.listdir("./img")
classfier = cv.CascadeClassifier("cv_models/haarcascade_frontalface_default.xml")

imgcount = 0

for file in fileList:

    imgPath = "img/" + file
    img = cv.imread(imgPath)
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    face_rects = classfier.detectMultiScale(img_grey, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5))

    color = (0,0,255)

    if len(face_rects) > 0: # 大于0则检测到人脸
            # 图片帧中有多个图片，框出每一个人脸
            for face_rect in face_rects:
                x, y, w, h = face_rect
                img_crop = img[y:y + h , x :x + w ]
                # 保存人脸图像
                cv.imwrite("./imgcroped/"+ str(imgcount) +".jpg",img_crop)
                #cv.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                imgcount += 1

    cv.imshow('long',img)
    #cv.waitKey(0)