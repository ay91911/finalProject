import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time,os

path = "#"
img=cv2.imread(path,cv2.IMREAD_COLOR)
print(img)
success, frame = cv2.imencode('.jpg',img)
print(frame)
img=frame.tobytes()
print(img)