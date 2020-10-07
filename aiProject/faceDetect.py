import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time,os
from os.path import split

#model path
model_path01 = 'haarcascade_frontalface_default.xml'
model_path02 = '_vgg16_01_.34-0.77-0.6478.h5'

#model path configuration
print(os.path.exists(model_path01))
print(os.path.exists(model_path02))

#model loading
face_detection = cv2.CascadeClassifier(model_path01)
emotion_detection = load_model(model_path02,compile=False)
EMOTIONS = ["happy","angry","sad","neutral","surprise"]


#test webcam
def videoCamera():

    camera =cv2.VideoCapture(0)
    while camera.isOpened():
        success, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

        #emotion_detection
        if len(faces)>0:
            face = sorted(faces, reverse=True, key=lambda x:(x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face

            img = gray[fY:fY + fH, fX:fX + fW]
            img = cv2.resize(img, (48,48))
            img = img.astype("float") /255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)

            preds = emotion_detection.predict(img)[0]

            emotion_probability = np.max(preds)

            #emotions output
            label = EMOTIONS[preds.argmax()]



            cv2.putText(frame,
                        label,
                        (fX, fY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 0),
                        2)
            cv2.rectangle(frame,
                          (fX, fY),  # 사각형의 시작점
                          (fX + fW, fY + fH),  # 시작점과 대각선에 있는 사각형의 끝점
                          (255, 255, 0),  # 사각형 색
                          3  # 선굵기(default =1), -1이면 사각형 내부가 채워짐
                          )







            if success:
                cv2.imshow('CameraWindow', frame)

                key = cv2.waitKey(1) & 0xFF
                if (key == 27):
                    break



    camera.release()
    cv2.destroyAllWindows()


'''
emotion_data = {1:[0.09,"abc"],2:[0.09,"cde"],3:[0.08,"fgd"],4:[0.07,"abc"]}
prob_list = []
for keys, values in emotion_data.items():
    prob_list.append(values)

max = max(prob_list)
print(prob_list)
print(max[0])
print(max[1])
'''

data = b'\xd8\x0fI@ff\xe6\x01\x00\x00\x00@'
print(data)
data = np.frombuffer(data,np.uint8)
print(data)
print(type(data))
img_decode = cv2.imdecode(data,cv2.IMREAD_COLOR)
print(img_decode)

import operator
best_prob = {1:[0.9,"a"],2:[0.91,"a"],3:[0.89,"b"],4:[0.78,"c"]}

s_best_prob = sorted(best_prob.items(), key=operator.itemgetter(1),reverse=True)
print(s_best_prob)
print(s_best_prob[0][1][1])
print(s_best_prob[1][1])
print(s_best_prob[2][1])



data = {0: None,  # level_1
                      1: None,  # level_2
                      2: None,  # level_3

                      }

print(data[0]==None)
data[0]=[1,"abc"]
print(type(data[0]))


exist = False
num = 0
while exist != True:
    while num < 10:
        num+=1
        print(exist)
    exist = True




