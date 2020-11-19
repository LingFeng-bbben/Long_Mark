import cv2 as cv
import os
import train
import PIL.Image as Image

classfier = cv.CascadeClassifier("cv_models/haarcascade_frontalface_alt2.xml")

imgcount = 0

cap = cv.VideoCapture(0)


while cap.isOpened():

    ok, img = cap.read()
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if ok!=True:
        break

    face_rects = classfier.detectMultiScale(img_grey, scaleFactor=1.2, minNeighbors=3, minSize=(5, 5))

    color = (0,0,255)

    if len(face_rects) > 0: # 大于0则检测到人脸
            # 图片帧中有多个图片，框出每一个人脸
            for face_rect in face_rects:
                x, y, w, h = face_rect
                img_crop = img[y:y + h , x :x + w ]
                img_resized = cv.resize(img_crop,(32,32))
                PIL_image = Image.fromarray(cv.cvtColor(img_resized, cv.COLOR_BGR2RGB))
                # 使用模型进行人脸识别
                label = train.predict_model(PIL_image)
                print("Longly:" + str(label))
                cv.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                font=cv. FONT_HERSHEY_SIMPLEX
                cv.putText(img, "Longly:",(x,y+h-10), font,0.4,(0,0,255),1, cv. LINE_AA)
                cv.putText(img, str(label),(x,y+h+5), font,0.4,(0,0,255),1, cv. LINE_AA)
                imgcount += 1

    cv.imshow('long',img)
    cv.waitKey(1)