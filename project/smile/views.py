import cv2,os
from django.shortcuts import render,redirect,get_object_or_404
from django.http import HttpResponse, StreamingHttpResponse, HttpResponseServerError
from django.contrib.auth import get_user_model, authenticate
import base64, re, json, time
#import matplotlib.pyplot as plt
from io import BytesIO
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from statistics import mode
#from skimage.transform import resize
from scipy.spatial import distance
from tensorflow.keras.models import load_model

def home(request):
    render(request, 'smile/index.html')

# model path
detection_model_path = 'detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'C:/dev/finalProject2/project/smile/emotion_models/_vgg16_01_.34-0.77-0.6478.h5'
print(os.path.exists(emotion_model_path))
#loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
print("load_model")
emotion_labels =["happy","angry","sad","neutral","surprise"]

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]   #emotion_target_size = (48,48)
image_size =''
embeddings =[]
emotion_count = 0
emotion_data ={}

# initialization
#graph = tf.get_default_graph() --- 오류미해결 생략
camera = 0
cameraChange=False
video_capture = cv2.VideoCapture(camera)
#User = get_user_model()  ---오류미해결 _user관련한 DB??!


# stream video actively
def video(request):
    return StreamingHttpResponse(gen(), content_type="multipart/x-mixed-replace; boundary=frame"), render(request,'smile/index.html') #내용물형식

# face recognition and expression recognition
def processImg(object):
    frame_window = 10
    emotion_offsets = (20, 40)
    global face_detection
    global emotion_classifier
    global emotion_labels

    emotion_window =[]
    gray = cv2.cvtColor(object,cv2.COLOR_RGB2GRAY)
    rgb = cv2.cvtColor(object, cv2.COLOR_BGR2RGB)

    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    face = sorted(faces, reverse=True, key=lambda x:(x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = face
    gray_face = gray[fY:fY + fH, fX:fX + fW]
    color_face = rgb[fY:fY + fH, fX:fX + fW]


    gray_face = cv2.resize(gray_face,(48,48))   #(48,48)로 조정
    gray_face = gray_face.astype("float") /255.0    #정규화진행
    gray_face = gray_face = img_to_array(gray_face)
    gray_face = np.expand_dims(gray_face, axis=0)

    '''emotion output'''
    emotion_prediction = emotion_classifier.predict(gray_face)[0]
    emotion_probablity = np.max(emotion_prediction)
    emotion_label = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label]       #happy, sad, surprise...

    emotion_window.append(emotion_text)

    if len(emotion_window) > frame_window:   #감정이 10개이상 측정되면 frame_window =10
        emotion_window.pop(0)


    emotion_mode = mode(emotion_window) #감정이 가장 많이 나온 최빈값을 도출

    if emotion_text =='happy':
        color = emotion_probablity * np.asarray((255, 0, 0))
    else:
        color = emotion_probablity * np.asarray((0, 255, 0))
    color = color.astype(int)
    color = color.tolist()

    cv2.rectangle(rgb,color,(fX, fY),(fX + fW, fY + fH),color)
    cv2.putText(rgb,emotion_mode,(fX, fY),cv2.FONT_HERSHEY_SIMPLEX,2,color,2)

    bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr_image



def gen():  #generateVideo
    global video_capture
    global cameraChange
    while True:
        #time.sleep(0.1)
        success, frame =video_capture.read()

        try:
            frame = bytes(cv2.imencode('.jpg',processImg(frame))) # byte단위로 이미지를 읽어와서 jpg로 변환해주는 코드
        except:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


