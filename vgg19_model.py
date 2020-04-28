import input
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


# model = VGG19(include_top=False, weights="imagenet")  # remove dense layers
# print(model.summary())
# model.save("files/imagenetVGG19.h5")
# comment all of below, RUN THIS, then uncomment below and comment this section


# experimental

# load model that does not have classification layer, just convolution layers
vggmodel = tf.keras.models.load_model('files/imagenetVGG19.h5')
print(vggmodel.summary())

colours = input.load("files/VGG19ImageDataPickle")
labels = input.load("files/VGG19LabelDataPickle")

temp = []
for i in range(5):
    temp.append(colours[i])  # add 5 images to a test batch

temp = np.array(temp)
print(temp.shape)

new_input = vggmodel.predict(temp)
# new_input = np.array(new_input)
# (5, 7, 7, 512) this is the input for our own neural network
print(new_input.shape)

# training model TO DO

model = keras.Sequential([
    layers.Flatten(input_shape=(7, 7, 512)),
    layers.Dense(256, activation=tf.nn.relu),  # consider changing
    keras.layers.Dropout(0.3),
    layers.Dense(34, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_crossentropy'])
model.summary()


# img = np.array(img)[np.newaxis, ]
# img = img/255
# print(img.shape)
# img = preprocess_input(img)

# prediction = model(img)
# print(prediction.shape)
# label = decode_predictions(prediction.numpy())
# label = label[0:5][0]  # print top 5
# for x in label:
#     print(f"{x[1]} {x[2]}")
