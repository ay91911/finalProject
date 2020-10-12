import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time,os



dic ={0:[98.02,'abc'],
      1:[59.03,'avds'],
      2:[87.54,'dsdf'],
      3:[99.99,'sedsdf00']}

best_smile = max([dic[1],dic[2],dic[3]])[0]
for keys, values in dic.items():
    if values[0] == best_smile:
        print(keys)