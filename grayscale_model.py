# import input
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import cv2
# from tensorflow import keras
# from tensorflow.keras import layers  # noqa
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  # noqa

# mpl.use('tkagg')

# IMG_SIZE = 256


# def get_classes():
#     for landuse in open("landuses.txt", "r").readLines():
#         yield landuse


# def get_model():
#     model = keras.Sequential([
#         keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu,
#                             input_shape=(IMG_SIZE, IMG_SIZE, 1)),  # 256 by 256, 1 for grayscale
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Flatten(),
#         keras.layers.Dense(128, activation=tf.nn.relu),
#         keras.layers.Dense(21, activation=tf.nn.softmax)
#     ])

#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy', "sparse_categorical_crossentropy"])
#     model.summary()

#     return model


# def train_model(train_data, train_labels, test_data, test_labels):
#     model = get_model()

#     return model, model.fit(train_data,
#                             train_labels,
#                             epochs=10,
#                             validation_split=0.1,
#                             verbose=2)


# def test_model(model, test_data, test_labels):
#     test_loss, test_acc, a = model.evaluate(test_data, test_labels)

#     print('Test accuracy:', test_acc)
#     # predictions = model.predict(test_data)
#     # print(predictions[0])


# def plot_history(history):
#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch

#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.plot(hist['epoch'], hist['loss'], label='Train Error')
#     plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
#     plt.ylim([0, 0.5])
#     plt.legend()

#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel('accuracy')
#     plt.plot(hist['epoch'], hist['accuracy'], label='Train Acc')
#     plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Acc')
#     plt.ylim([0, 1])
#     plt.legend()
#     plt.show()


# def process_data():
#     images = input.load("files\\UCMercedImages")
#     labels = input.load("files\\UCMercedLabels")
#     #grays = input.grayscale(images)
#     grays = []
#     for image in images:
#         grays.append(img_to_array(cv2.cvtColor(cv2.resize(
#             image, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)))

#     combined = list(zip(grays, labels))
#     np.random.seed(19)
#     np.random.shuffle(combined)

#     train_data, train_labels = zip(*combined)
#     train_data = np.array(train_data)
#     train_data = train_data / 255
#     train_data = train_data.reshape(2100, IMG_SIZE, IMG_SIZE, 1)
#     train_labels = np.array(train_labels)

#     test_data = train_data[2000:]
#     test_labels = train_labels[2000:]

#     train_data = train_data[:2000]
#     train_labels = train_labels[:2000]

#     print(train_data.shape)
#     print(train_labels.shape)
#     print(test_data.shape)
#     print(test_labels.shape)

#     # model, history = train_model(
#     #     train_data, train_labels, test_data, test_labels)

#     # test_model(model, test_data, test_labels)

#     # print("Saving model...")
#     # model.save('files\\test.h5')
#     # print("Saved.")

#     # plot_history(history)

#     new_model = tf.keras.models.load_model('files\\test.h5')
#     new_model.summary()
#     test_model(new_model, test_data, test_labels)


# process_data()


# # plt.figure(figsize=(10, 10))
# # for i in range(25):
# #     plt.subplot(5, 5, i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(train_data[i], cmap='gray')
# #     plt.xlabel(index[train_label[i]])
# # plt.show()


# # history = baseline_model.fit(train_data,
# #                              train_label,
# #                              epochs=10,
# #                              batch_size=10,
# #                              validation_split=0.1,
# #                              verbose=2)

# # predictions = baseline_model.predict(test_data)

# # print(predictions[0:10])
# # print(test_label[0:10])

# # plt.figure(figsize=(10, 10))
# # for i in range(25):
# #     plt.subplot(5, 5, i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(test_data[i], cmap='gray')
# #     plt.xlabel(f"{index[test_label[i]]}, {predictions[i]}")
# # plt.show()

# # print(history)

# # save model
# # baseline_model.save('baseline_model.h5')
# # print('Saved model.')
