import tensorflow as tf
import numpy
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib as mpl
import matplotlib.pyplot as plt
import input

mpl.use('tkagg')

def get_classes():
    for landuse in open("landuses.txt", "r").readLines():
        yield landuse


def get_model():
    # model = keras.Sequential([
    #     layers.Flatten(input_shape=(256, 256)),
    #     layers.Dense(64, activation=tf.nn.relu),
    #     layers.Dropout(0.5),
    #     layers.Dense(21, activation=tf.nn.softmax)
    # ])

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(21, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])
    model.summary()
    return model


def train_model(train_data, train_labels, test_data, test_labels):
    model = get_model()

    return model, model.fit(train_data,
                        train_labels,
                        epochs=5,
                        batch_size=512,
                        validation_split=0.2,
                        verbose=2)


def test_model(model, test_data, test_labels):
    test_loss, test_acc = model.evaluate(test_data, test_labels)

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
    plt.ylim([0, 0.5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Acc')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Acc')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def process_data():
    images = input.load("UCMercedImages")
    labels = input.load("UCMercedLabels")
    # grays = input.grayscale(images)
    colours = []
    for image in images:
        colours.append(img_to_array(image))

    combined = list(zip(colours, labels))
    numpy.random.shuffle(combined)

    train_data, train_labels = zip(*combined)
    train_data = numpy.array(train_data)
    train_labels = numpy.array(train_labels)

    test_data = train_data[2000:]
    test_labels = train_labels[2000:]

    train_data = train_data[:2000]
    train_labels = train_labels[:2000]

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    model, history = train_model(train_data, train_labels, test_data, test_labels)
    test_model(model, test_data, test_labels)

    plot_history(history)


# process_data()


