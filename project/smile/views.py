from django.shortcuts import render,redirect, get_object_or_404, get_list_or_404
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponseServerError
import cv2, time, operator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from os.path import split
import os
from statistics import mode
from smile.models import PHRASE

msg = "Please click the next button"


emotion_image_data = {0: None,  # 무표정
                      1: None,  # level_1
                      2: None,  # level_2
                      3: None,  # level_3

                      }
phraseList =[]

# model path
#대윤
detection_model_path = 'C:/dev/finalProject2/project/smile/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'C:/dev/finalProject2/project/smile/emotion_models/_vgg16_01_.34-0.77-0.6478.h5'
#찬욱
# detection_model_path = 'C:/Users/acorn-519/PycharmProjects/finalProject/project/smile/detection_models/haarcascade_frontalface_default.xml'
# emotion_model_path = 'C:/Users/acorn-519/PycharmProjects/finalProject/project/smile/emotion_models/_vgg16_01_.34-0.77-0.6478.h5'
#아영
# detection_model_path = 'C:/dev/finalProject2/project/smile/detection_models/haarcascade_frontalface_default.xml'
# emotion_model_path = 'C:/dev/finalProject2/project/smile/emotion_models/_vgg16_01_.34-0.77-0.6478.h5'




emotion_labels = ["happy", "angry", "sad", "neutral", "surprise"]

# initialization
frame_window = 30
emotion_window = []
best_prob_level = [None]

def index(request):
    if emotion_image_data[0] == None and emotion_image_data[1] == None and emotion_image_data[2] == None and emotion_image_data[3] == None:
        return render(request, 'smile/neutral.html')
    elif emotion_image_data[0] != None and emotion_image_data[1] == None and emotion_image_data[2] == None and emotion_image_data[3] == None:
        return render(request, 'smile/smile_1.html')
    elif emotion_image_data[0] != None and emotion_image_data[1] != None and emotion_image_data[2] == None and emotion_image_data[3] == None:
        return render(request, 'smile/smile_2.html')
    elif emotion_image_data[0] != None and emotion_image_data[1] != None and emotion_image_data[2] != None and emotion_image_data[3] == None:
        return render(request, 'smile/smile_3.html')

    else:
        return render(request, 'service/mainpage1.html')


def ListPhrase(request):
    #context = {'phraseList':phraseList}
    context ={'phraseList':phraseList}
    print(context)
    return render(request, 'smile/emotion_detection_2.html',context)



class VideoCamera_smile:
    global detection_model_path
    global emotion_model_path
    global emotion_labels
    global frame_window
    global emotion_window
    global best_prob_level
    global emotion_image_data
    global phraseList


    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.cascade = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]  # emotion_target_size = (48,48)
        self.smile_count = 0
        self.save_file_count = 0

        self.smile_data = {}  # {count:percent}
        self.emotion_label_list =[]
        self.emo_label_exist = False
        self.emo_image_exist = False
        self.frame_count = 0
        self.today_emotion_label = []

    def __del__(self):
        self.video.release()
        # self.save_file_count


    # 오늘의 한마디
    def today_phrase(self, img_count):
        while self.emo_label_exist != True:
            success, frame = self.video.read()
            self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                emotion_probability = round(emotion_probability * 100, 2)  # 소수둘째자리까지 반올림 ex)90.12
                emotion_label = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label]  # happy, sad, surprise
                emotion_window.append(emotion_text)

                while self.frame_count < img_count:
                    self.frame_count += 1
                    draw_rectangle(face_coordinates, frame, (0, 255, 100))
                    put_text(face_coordinates, frame, "checking", (0, 255, 100)),
                    success, jpeg = cv2.imencode('.jpg', frame)
                    jpeg_tobytes = jpeg.tobytes()
                    self.emotion_label_list.append(emotion_label)
                    return jpeg_tobytes

                while self.frame_count >= img_count:

                    self.today_emotion_label.append(mode(self.emotion_label_list))

                    print(self.today_emotion_label)

                    #오늘의 한마디 가져오기
                    Phrase_list = get_list_or_404(PHRASE, EMOTION_KIND=self.today_emotion_label[0])[0]

                    print(type(Phrase_list))



                    phraseList.append(Phrase_list)
                    print(phraseList)
                    #print(session["loginuser"])
                    #session["aa"] = Phrase_list
                    #print(session["aa"])
                    #self.msg = Phrase_list


                    success, jpeg = cv2.imencode('.jpg', frame)
                    jpeg_tobytes = jpeg.tobytes()

                    self.frame_count = 0
                    self.emo_label_exist = True
                    self.emotion_label_list.clear()

                    return jpeg_tobytes

        # print(phraseList)

        success_next, frame_next = self.video.read()
        self.faces = self.cascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5)
        for face_coordinates in self.faces:
            put_text_info(face_coordinates, frame_next, "neutral_face is detected", (0, 255, 100))
            success, jpeg = cv2.imencode('.jpg', frame_next)
            return jpeg.tobytes()

    def get_frame(self, img_count, level_index, emotion='happy'):
        global emotion_image_data

        # 학습후 저장된 데이터가 없으면!
        # while len(emotion_image_data) == 0:
        # while emotion_image_data[0] == None:
        #while emotion_image_data[level_index] == None:
        while self.emo_image_exist != True:

            success, frame = self.video.read()
            success_saved, frame_saved = self.video.read()
            success_next, frame_next = self.video.read()

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

                if emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 0, 0))

                elif emotion_text == 'angry':
                    color = emotion_probability * np.asarray((0, 0, 255))

                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((255, 255, 0))

                elif emotion_text == 'neutral':
                    color = emotion_probability * np.asarray((0, 255, 255))

                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                color = color.astype(int)
                color = color.tolist()

                if emotion_text == emotion:
                    while self.smile_count < img_count:  # 30
                        self.smile_count += 1

                        draw_rectangle(face_coordinates, frame, (0, 255, 100))
                        put_text(face_coordinates, frame, (str(self.smile_count) + ": " + str(emotion_probability)),
                                 (0, 255, 100))
                        success, jpeg = cv2.imencode('.jpg', frame)
                        jpeg_tobytes = jpeg.tobytes()

                        success_saved, jpeg_saved = cv2.imencode('.jpg', frame_saved)

                        # smile percent
                        emotion_probability
                        self.smile_data[self.smile_count] = [emotion_probability, jpeg_saved]  # dictionary

                        return jpeg_tobytes

                    while self.smile_count >= img_count and self.smile_count % img_count == 0:

                        prob_list = []
                        for keys, values in self.smile_data.items():
                            prob_list.append(values)  # values = [prob, img]

                        # if best_prob_level[0] == None:

                        best_prob_level[0] = max(prob_list)  # [[prob, img]]
                        draw_rectangle(face_coordinates, frame, (0, 255, 100))
                        put_text(face_coordinates, frame, (str(self.smile_count) + ": " + str(emotion_probability)),
                                 (0, 255, 100))
                        success, jpeg = cv2.imencode('.jpg', frame)

                        imgwrite(best_prob_level, emotion_image_data, level_index)

                        self.smile_count = 0
                        self.emo_image_exist = True

                        # else:
                        #     best_prob_level[0] = None



                else:
                    self.smile_count = 0

            success, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        # 데이터가 저장되어 있으면

        success_next, frame_next = self.video.read()
        self.faces = self.cascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5)
        for face_coordinates in self.faces:
            put_text_info(face_coordinates, frame_next, "Please click the next button", (0, 255, 100))
            success, jpeg = cv2.imencode('.jpg', frame_next)
            return jpeg.tobytes()





#-------------------------------------------------------------------------------------------------------
def video_today_phrase(request):
    try:

        return StreamingHttpResponse(gen_today_phrase(VideoCamera_smile(), frame_count=25),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)

def video_neutral(request):
    try:
        return StreamingHttpResponse(gen_non_smile(VideoCamera_smile(), frame_count=15, level_index=0),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)



def video_smile_level1(request):
    try:
        return StreamingHttpResponse(gen_level(VideoCamera_smile(), frame_count=10, level_index=1),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)


def video_smile_level2(request):
    try:
        return StreamingHttpResponse(gen_level(VideoCamera_smile(), frame_count=20, level_index=2),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)


def video_smile_level3(request):
    try:
        return StreamingHttpResponse(gen_level(VideoCamera_smile(), frame_count=30, level_index=3),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)


# _______________________________________________________________________

def gen_today_phrase(camera, frame_count):
    while True:
        frame = camera.today_phrase(frame_count)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_non_smile(camera,frame_count, level_index=0):
    while True:
        frame = camera.get_frame(frame_count, level_index, 'neutral')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



def gen_level(camera, frame_count, level_index=1):
    while True:
        frame = camera.get_frame(frame_count, level_index)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# --------------------------------------------------------------------------

'''
def img_sort(best_prob, rank=3):
    best_prob_sort = sorted(best_prob.items(), key=operator.itemgetter(1), reverse=True)
    best_prob_sort = best_prob_sort[0:rank]  # 랭크몇위입력

    return best_prob_sort


def imgwrite_new(rank=3):
    best_prob_sort = img_sort(best_prob_level)
    for i in range(rank):
        data_rank = best_prob_sort[i][1][1]
        img = cv2.imdecode(data_rank, cv2.IMREAD_COLOR)
        cv2.imwrite('C:/dev/finalProject/aiProject/' + str(i) + '.png', img)


def imgwrite(best_prob_level, emotion_image_data):
    data_img = best_prob_level[0][1]
    img = cv2.imdecode(data_img, cv2.IMREAD_COLOR)
    cv2.imwrite('C:/dev/finalProject/aiProject/' + 'best_level' + str(0 + 1) + '.png', img)

    emotion_image_data[0] = best_prob_level
    print(emotion_image_data)
    print(len(emotion_image_data))
'''

def imgwrite(best_prob_level, emotion_image_data, level_index):
    data_prob = best_prob_level[0][0]
    data_img = best_prob_level[0][1]
    # 대윤
    #path = 'C:/dev/finalProject2/aiProject/images/'
    # 찬욱
    path = 'C:/Users/acorn-519/PycharmProjects/finalProject/aiProject/images/'
    # 아영
    #path = 'C:/dev/finalProject2/aiProject/images/'


    img = cv2.imdecode(data_img, cv2.IMREAD_COLOR)
    cv2.imwrite( path + 'best_level' + str(level_index) + '.png', img)

    dir, file = os.path.split(path + 'best_level' + str(level_index) + '.png')
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


# def get_today_phrase():
#     return today_emotion_label

# def reset_today_phrase():
#     today_emotion_label.clear()



def reset():
    emotion_image_data[0] = "None"
    emotion_image_data[1] = "None"
    emotion_image_data[2] = "None"



