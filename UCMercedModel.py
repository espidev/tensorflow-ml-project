import input
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import pickle
from tensorflow import keras
from tensorflow.keras import layers  # noqa

images = input.load("UCMercedImages")
grays = input.grayscale(images)
labels = np.array(input.load("UCMercedLabels"))

index = []  # mapping number label (0-27) to text label ("agriculture")
file = open("landuses.txt", "r")
landuses = file.readlines()
for landuse in landuses:
    index.append(landuse[:-1])


# baseline grayscale model

combined = list(zip(grays, labels))
np.random.shuffle(combined)
train_data, train_label = zip(*combined)
train_data = np.array(train_data)
train_label = np.array(train_label)

test_data = train_data[1900:]  # take 200 datapoints for testing
test_label = train_label[1900:]

train_data = train_data[:1900]
train_label = train_label[:1900]

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_data[i], cmap='gray')
#     plt.xlabel(index[train_label[i]])
# plt.show()

baseline_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(256, 256)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(21, activation=tf.nn.softmax)
])

baseline_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
baseline_model.summary()

history = baseline_model.fit(train_data,
                             train_label,
                             epochs=15,
                             batch_size=512,
                             validation_split=0.2,
                             verbose=2)

predictions = baseline_model.predict(test_data)

print(predictions[0])
print(test_label[0])
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_data[i], cmap='gray')
#     plt.xlabel(f"{index[test_label[i]]}, {predictions[i]}")
# plt.show()

# print(history)
