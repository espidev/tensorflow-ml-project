import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib as mpl
import matplotlib.pyplot as plt

def getModel():
    model = keras.Sequential([
        layers.Flatten(input_shape=(256, 256)),
        layers.Dense(16, activation=tf.nn.relu),
        layers.Dense(16, activation=tf.nn.relu),
        layers.Dense(21, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
    model.summary()
    return model

def trainModel(train_data, train_labels, test_data, test_labels, class_names):
    model = getModel()

    history = model.fit(train_data,
                        train_labels,
                        epochs=15,
                        batch_size=512,
                        validation_split=0.2,
                        verbose=2)
    print(history)

    testModel(model, test_data)



def testModel(model, test_data):
    predictions = model.predict(test_data)
    print(predictions[0])




