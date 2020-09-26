from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponseServerError
import cv2, time, operator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

from statistics import mode

emotion_image_data = {

}

# model path
detection_model_path = 'C:/dev/finalProject/project/smile/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'C:/dev/finalProject\project\smile\emotion_models\_vgg16_01_.34-0.77-0.6478.h5'
emotion_labels = ["happy", "angry", "sad", "neutral", "surprise"]

# initialization
frame_window = 30
emotion_window = []
best_prob = {}



class VideoCamera_smile:

    global detection_model_path
    global emotion_model_path
    global emotion_labels
    global frame_window
    global emotion_window
    global best_prob


    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.cascade = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]  # emotion_target_size = (48,48)
        self.smile_count = 0
        self.save_file_count = 0
        self.smile_data = {}  # {count:percent}

    def __del__(self):
        self.video.release()


    def get_frame(self, img_count):

        success, frame = self.video.read()
        success_saved, frame_saved = self.video.read()

        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 추가코드0924
        self.faces = self.cascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5)

        for face_coordinates in self.faces:


            gray_face = self.gray[face_coordinates[1]:face_coordinates[1] + face_coordinates[3], face_coordinates[0]:face_coordinates[0] + face_coordinates[2]]
            gray_face = cv2.resize(gray_face, (48, 48))
            gray_face = gray_face.astype("float") / 255.0
            gray_face = img_to_array(gray_face)
            gray_face = np.expand_dims(gray_face, axis=0)  # (48,48,1)

            emotion_prediction = self.emotion_classifier.predict(gray_face)[0]

            emotion_probability = np.max(emotion_prediction)
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

            if emotion_text == 'happy':
                while self.smile_count < img_count:  # 30
                    self.smile_count += 1

                    draw_rectangle(face_coordinates, frame,(0, 255, 100))
                    put_text(face_coordinates, frame, (str(self.smile_count) + ": " + str(emotion_probability)),(0, 255, 100))
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
                        prob_list.append(values)

                    if len(best_prob) < 5:
                        best_prob[self.save_file_count]= (max(prob_list))
                        draw_rectangle(face_coordinates, frame, (0, 255, 100))
                        put_text(face_coordinates, frame, (str(self.smile_count) + ": " + str(emotion_probability)),(0, 255, 100))
                        success, jpeg = cv2.imencode('.jpg', frame)
                        self.save_file_count += 1



                    # img write

                    elif len(best_prob) == 5:

                        imgwrite_new()
                        #imgwrite(self.save_file_count)


                    elif len(best_prob) > 5:
                        best_prob.clear()
                        self.save_file_count = 0



                    self.smile_count = 0

                    return jpeg.tobytes()
            else:
                self.smile_count =0





        success, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()


def video_smile(request):
    try:
        return StreamingHttpResponse(gen_level1(VideoCamera_smile()), content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("asborted", e)


# _______________________________________________________________________

def gen_level1(camera):
    while True:
        frame = camera.get_frame(20)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_level2(camera):
    while True:
        frame = camera.get_frame(30)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_level3(camera):
    while True:
        frame = camera.get_frame(40)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#--------------------------------------------------------------------------


def img_sort(best_prob, rank=3):
    best_prob_sort = sorted(best_prob.items(), key=operator.itemgetter(1),reverse=True)
    best_prob_sort = best_prob_sort[0:rank]    #랭크몇위입력

    return best_prob_sort


def imgwrite_new(rank=3):
    best_prob_sort = img_sort(best_prob)
    for i in range(rank):
        data_rank = best_prob_sort[i][1][1]
        img = cv2.imdecode(data_rank, cv2.IMREAD_COLOR)
        cv2.imwrite('C:/dev/finalProject/aiProject/'+str(i)+'.png', img)











def imgwrite(save_file_count):
    data = best_prob[0][1]
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    cv2.imwrite(('C:/dev/finalProject/aiProject/{0}.png'.format(save_file_count)), img)
    best_prob.clear()




def draw_rectangle(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x,y),(x+w, y+h), color, 2)


def put_text(coordinates, image_array, text, color, font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


