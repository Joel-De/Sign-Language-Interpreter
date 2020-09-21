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
    ret,image  = cap.read()



    # ORANGE_MIN = np.array([0, 0, 114], np.uint8)
    # ORANGE_MAX = np.array([179, 255, 255], np.uint8)
    #
    # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv_img[:, :, 2] -= 60
    # frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
    #
    #
    # fin = frame_threshed
    #
    # frame_threshed = cv2.resize(frame_threshed,(64,64))
    # frame_threshed = np.expand_dims(frame_threshed, 0)

    #print(frame_threshed.shape)
    fin = image
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(32,32))
    imag = cv2.resize(image,(32,32))
    #imag = imag/255
    imag = np.expand_dims(imag,0)
    #cv2.waitKey()
    #print (image.shape)
    predict = imported.predict(imag)
    predict = np.argmax(predict, axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX


    ####### CHAR FOR SING LANGUAGE
    chr(predict[0] + 65)

    cv2.putText(fin, chr(predict[0] + 65), (10, 100), font, 3, (255, 255, 255), 10)


    print(chr(predict[0]+65))
    cv2.imshow("window3", fin)
    cv2.waitKey(1)



    #print("MODEL SAYS THIS IS")





