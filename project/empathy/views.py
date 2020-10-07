from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponseServerError
import cv2, time, operator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from os.path import split
import os

from statistics import mode
#
# emotion_image_data = {0: None,  # level_1
#                       1: None,  # level_2
#                       2: None,  # level_3
#
#                       }
#
# model path
#대윤
# detection_model_path = 'C:/dev/finalProject2/project/smile/detection_models/haarcascade_frontalface_default.xml'
# emotion_model_path = 'C:/dev/finalProject2/project/smile/emotion_models/_vgg16_01_.34-0.77-0.6478.h5'
#찬욱
# detection_model_path = 'C:/Users/acorn-519/PycharmProjects/finalProject/project/smile/detection_models/haarcascade_frontalface_default.xml'
# emotion_model_path = 'C:/Users/acorn-519/PycharmProjects/finalProject/project/smile/emotion_models/_vgg16_01_.34-0.77-0.6478.h5'
#아영
detection_model_path = 'C:/Users/acorn-508/PycharmProjects/finalProject/project/smile/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'C:/Users/acorn-508/PycharmProjects/finalProject/project/smile/emotion_models/_vgg16_01_.34-0.77-0.6478.h5'


emotion_labels = ["happy", "angry", "sad", "neutral", "surprise"]

# initialization
frame_window = 30
emotion_window = []
best_prob_level = [None]


def training1(request):
    return render(request, 'empathy/training.html')

def training2(request):
    return render(request, 'empathy/training_2.html')

def training3(request):
    return render(request, 'empathy/training_3.html')

def training4(request):
    return render(request, 'empathy/training_4.html')

def training5(request):
    return render(request, 'empathy/training_5.html')

def training_result(request):
    return render(request, 'empathy/training_result.html')

def mainpage(request):
    return render(request, 'service/mainpage1.html')


# _______________________________________________________________________

def happy_training(request):
    try:
        time.sleep(5)
        return StreamingHttpResponse(gen_level(VideoCamera_smile(), frame_count=10, emotion='happy'),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)


def angry_training(request):
    try:
        time.sleep(5)
        return StreamingHttpResponse(gen_level(VideoCamera_smile(), frame_count=10, emotion='angry'),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)


def sad_training(request):
    try:
        time.sleep(5)
        return StreamingHttpResponse(gen_level(VideoCamera_smile(), frame_count=10, emotion='sad'),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)

def surprise_training(request):
    try:
        time.sleep(5)
        return StreamingHttpResponse(gen_level(VideoCamera_smile(), frame_count=10, emotion='surprise'),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)

# _______________________________________________________________________
class VideoCamera_smile:
    global detection_model_path
    global emotion_model_path
    global emotion_labels
    global frame_window
    global emotion_window
    global best_prob_level
    global emotion_image_data

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.cascade = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]  # emotion_target_size = (48,48)
        self.smile_count = 0
        self.save_file_count = 0
        self.smile_data = {}  # {count:percent}
        self.trainIsDone = False

    def __del__(self):
        self.video.release()
        # self.save_file_count

    def training_frame(self, img_count, emotion):
        global emotion_image_data

        # while emotion_image_data[level_index] == None:
        while self.trainIsDone != True:
            success, frame = self.video.read()

            self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 추가코드0924
            self.faces = self.cascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5)

            for face_coordinates in self.faces:

                gray_face = self.gray[face_coordinates[1]:face_coordinates[1] + face_coordinates[3],
                            face_coordinates[0]:face_coordinates[0] + face_coordinates[2]]
                gray_face = cv2.resize(gray_face, (48, 48))
                gray_face = gray_face.astype("float") / 255.0
                gray_face = img_to_array(gray_face)
                gray_face = np.expand_dims(gray_face, axis=0)  # (48,48,1)

                emotion_prediction = self.emotion_classifier.predict(gray_face)[0]

                emotion_probability = np.max(emotion_prediction)
                emotion_probability = round(emotion_probability*100,2)  #소수둘째자리까지 반올림 ex)90.12
                emotion_label = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label]  # happy, sad, surprise
                emotion_window.append(emotion_text)

                if emotion_text == emotion:
                    while self.smile_count < img_count:  # 30
                        self.smile_count += 1

                        draw_rectangle(face_coordinates, frame, (0, 255, 100))
                        put_text(face_coordinates, frame, (str(self.smile_count)),
                                 (0, 255, 100))
                        success, jpeg = cv2.imencode('.jpg', frame)
                        jpeg_tobytes = jpeg.tobytes()
                        # success_saved, jpeg_saved = cv2.imencode('.jpg', frame_saved)

                        # smile percent
                        # emotion_probability
                        # self.smile_data[self.smile_count] = [emotion_probability, jpeg_saved]  # dictionary

                        return jpeg_tobytes


                    self.trainIsDone = True
                    # put_text_info(face_coordinates, frame, "Very well!! Please click the next button",
                    #               (0, 255, 100))
                    # success, jpeg = cv2.imencode('.jpg', frame)
                    # return jpeg.tobytes()
                    #
                else:
                    self.smile_count = 0

            success, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        success_next, frame_next = self.video.read()
        # success_saved, frame_saved = self.video.read()
        self.faces = self.cascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5)
        for face_coordinates in self.faces:
            put_text_info(face_coordinates, frame_next, "Very well!! Please click the next button", (0, 255, 100))
            success, jpeg = cv2.imencode('.jpg', frame_next)
            return jpeg.tobytes()

#-------------------------------------------------------------------------------------------------------


def gen_level(camera, frame_count,emotion):
    while True:
        frame = camera.training_frame(frame_count,emotion)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# --------------------------------------------------------------------------

def imgwrite(best_prob_level, emotion_image_data, level_index):
    data_prob = best_prob_level[0][0]
    data_img = best_prob_level[0][1]
    # 대윤
    # path = 'C:/dev/finalProject2/aiProject/images/'
    # 찬욱
    # path = 'C:/Users/acorn-519/PycharmProjects/finalProject/aiProject/images/'
    # 아영
    path = 'C:/Users/acorn-508/PycharmProjects/finalProject/aiProject/images/'
    img = cv2.imdecode(data_img, cv2.IMREAD_COLOR)
    cv2.imwrite( path + 'best_level' + str(level_index + 1) + '.png', img)

    dir, file = os.path.split(path + 'best_level' + str(level_index + 1) + '.png')
    imgPath = dir+file




    emotion_image_data[level_index] = [data_prob,imgPath]    #emotion_image_data에 저장

    print(emotion_image_data)



def draw_rectangle(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def put_text(coordinates, image_array, text, color, font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def put_text_info(coordinates, image_array, text, color, font_scale=0.9, thickness=2, x_pixel=75, y_pixel=100):
    x_root, y_root = coordinates[:2]
    x = x_root - x_pixel
    y = y_root - y_pixel
    cv2.putText(image_array, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def reset():
    emotion_image_data[0] = "None"
    emotion_image_data[1] = "None"
    emotion_image_data[2] = "None"
    print(emotion_image_data)


