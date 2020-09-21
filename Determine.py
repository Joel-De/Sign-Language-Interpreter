import cv2
import pathlib
import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)
data_dir = pathlib.Path('dataset/')

imported = keras.models.load_model("savedsign.h5")
print(imported.summary())

while True:
    ret, image = cap.read()

    fin = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    imag = cv2.resize(image, (32, 32))

    imag = np.expand_dims(imag, 0)

    predict = imported.predict(imag)
    predict = np.argmax(predict, axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    ####### CHAR FOR SIGN LANGUAGE
    chr(predict[0] + 65)
    cv2.putText(fin, chr(predict[0] + 65), (10, 100), font, 3, (255, 255, 255), 10)
    print(chr(predict[0] + 65))
    cv2.imshow("window3", fin)
    cv2.waitKey(1)
