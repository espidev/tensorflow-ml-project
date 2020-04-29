import input
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import keras
from keras import layers  # noqa
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  # noqa

mpl.use('tkagg')

IMG_SIZE = 150


def get_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(
            IMG_SIZE, IMG_SIZE, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation=tf.nn.relu),  # consider changing
        keras.layers.Dropout(0.3),
        layers.Dense(34, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])
    model.summary()
    return model


def load_model(name):
    model = tf.keras.models.load_model(
        f'files/{name}.h5', custom_objects={'softmax_v2': tf.nn.softmax})
    model.summary()
    return model


def save_model(model, name):
    print("Saving model...")
    model.save(f'files/{name}.h5')
    print("Saved.")


def train_model(model, train_data, train_labels, test_data, test_labels):
    return model, model.fit(train_data,
                            train_labels,
                            epochs=1,
                            # batch_size=512,
                            validation_split=0.2,
                            verbose=2)


def test_model(model, test_data, test_labels):
    test_loss, test_acc, a = model.evaluate(test_data, test_labels)
    print('Test accuracy:', test_acc)
    # predictions = model.predict(test_data)
    # print(predictions[0])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
    plt.ylim([0, 1])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Acc')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Acc')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def run(model):
    colours = input.load("files/BaseImageDataPickle")
    labels = input.load("files/BaseLabelDataPickle")

    combined = list(zip(colours, labels))
    np.random.seed(19)
    np.random.shuffle(combined)

    train_data, train_labels = zip(*combined)
    train_data = np.array(train_data, dtype="float32")
    # instead of float64, reduce memory usage
    train_data = train_data / 255
    train_labels = np.array(train_labels)

    test_data = train_data[21000:]
    test_labels = train_labels[21000:]

    train_data = train_data[:21000]
    train_labels = train_labels[:21000]

    print("Train Data:", train_data.shape)
    print("Train Label:", train_labels.shape)
    print("Test Data:", test_data.shape)
    print("Train Label:", test_labels.shape)

    trained_model, history = train_model(
        model, train_data, train_labels, test_data, test_labels)

    test_model(trained_model, test_data, test_labels)
    plot_history(history)


#model = get_model()
# model = load_model("colour_model2")
# run(model)
