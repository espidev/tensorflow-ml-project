import cv2
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


model = VGG19()
print(model.summary())
model.save("files/imagenetVGG19.h5")
# RUN THIS, then uncomment below and comment this section


# model = tf.keras.models.load_model('files/imagenetVGG19.h5')
# print(model.summary())


# img = cv2.imread("files/rawdata/airplane/airplane_395.jpg",
#                  cv2.IMREAD_UNCHANGED)
# img = cv2.resize(img, (224, 224))
# img = np.array(img)[np.newaxis, ]
# img = img/255
# print(img.shape)
# img = preprocess_input(img)

# prediction = model(img)
# print(prediction.shape)
# label = decode_predictions(prediction.numpy())  # does not work
# label = label[0:5][0]  # print top 5
# for x in label:
#     print(f"{x[1]} {x[2]}")
