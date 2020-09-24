import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time,os

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

videoCamera()

