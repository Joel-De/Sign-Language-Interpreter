import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras


import pathlib
import time

# cap = cv2.VideoCapture(0)
# cap.set(3, 480)
# cap.set(4, 480)

#                     Data collection
###############################################################
#
# cap = cv2.VideoCapture(0)
# cap.set(3, 480)
# cap.set(4, 480)
# c = 0
# n=0
# while True:
#     ret,image  = cap.read()
#     image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#
#     #cv2.waitKey()
#     num = cv2.Laplacian(image,cv2.CV_64F).var()
#     print (num)
#     cv2.imshow("window3", image)
#     if num > 40 and c%10==0:
#         cv2.imshow("windows", image)
#         n+=1
#         cv2.imwrite("Cup/img" +  str(n) + ".jpg" , image)
#
#     c += 1
#     cv2.waitKey(1)
#     if n>300:
#         break
#

####################################################################


data_dir = pathlib.Path('NEWDATA/')

batch_size = 32
img_height = 128
img_width = 128

train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=1256,
    color_mode="grayscale",
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


num_classes = 9

data_augmentation = tf.keras.Sequential([

    layers.experimental.preprocessing.RandomRotation(0.3),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.1)

])

model = tf.keras.Sequential([
    # data_augmentation,

    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(64, 12, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 6, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(72, activation='relu'),
    layers.Dense(num_classes, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])

model.fit(train_ds, epochs=50)

print(model.summary())

model.save("savedsign.h5")
