from keras import activations
from keras import optimizers
from keras import losses
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *

WIDTH = 340
HEIGHT = 425
CHANNEL = 3


def race_model_test2b(num_classes):
    model_name = 'mdl_catent_adam_test2b_dataset'
    model = Sequential()
    # model.add(Flatten(input_shape=( WIDTH, HEIGHT, CHANNEL)))
    model.add(InputLayer(input_shape=(HEIGHT, WIDTH, 3,)))
    # model.add(Reshape(target_shape=(10, HEIGHT, WIDTH, CHANNEL)))

    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_32f"))
    model.add(AveragePooling2D(pool_size=3, padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=48, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_48f"))
    model.add(AveragePooling2D(pool_size=3, padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_64f"))
    model.add(AveragePooling2D(pool_size=3, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation=activations.relu, name="Dense_Relu_256"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(256, activation=activations.relu, name="Dense_Relu_128"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activations.softmax, name=f"Dense_{num_classes}_softmax"))
    model.add(Dense(1, activation=activations.sigmoid, name="Dense_sigmoid"))
    # optimizer = optimizers.Adam(lr=1e-3)
    optimizer = optimizers.Adam(lr=1e-3)
    loss = losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model, model_name


def race_model_test2(num_classes):
    model_name = 'mdl_catent_adam_test2a_dataset'
    model = Sequential()
    # model.add(Flatten(input_shape=( WIDTH, HEIGHT, CHANNEL)))
    model.add(InputLayer(input_shape=(HEIGHT, WIDTH, 3,)))
    # model.add(Reshape(target_shape=(10, HEIGHT, WIDTH, CHANNEL)))

    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_32f"))
    model.add(MaxPool2D(pool_size=3, padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=48, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_48f"))
    model.add(MaxPool2D(pool_size=3, padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_64f"))
    model.add(MaxPool2D(pool_size=3, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation=activations.relu, name="Dense_Relu_256"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(256, activation=activations.relu, name="Dense_Relu_128"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activations.softmax, name=f"Dense_{num_classes}_softmax"))
    model.add(Dense(1, activation=activations.sigmoid, name="Dense_sigmoid"))
    # optimizer = optimizers.Adam(lr=1e-3)
    optimizer = optimizers.Adam(lr=1e-3)
    loss = losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model, model_name


def race_model_test(num_classes):
    model_name = 'mdl_binent_sgd_test_dataset'
    model = Sequential()
    # model.add(Flatten(input_shape=( WIDTH, HEIGHT, CHANNEL)))
    model.add(InputLayer(input_shape=(HEIGHT, WIDTH, 3,)))
    # model.add(Reshape(target_shape=(10, HEIGHT, WIDTH, CHANNEL)))
    model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_16f"))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_32f"))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name="Conv2D_64f"))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(0.1, name="Dropout_0.1"))
    model.add(Flatten())
    model.add(Dense(512, activation=activations.relu, name="Dense_Relu"))
    model.add(Dropout(rate=0.2, name="Dropout_0.2"))
    model.add(Dense(num_classes, activation=activations.softmax, name=f"Dense_{num_classes}_softmax"))
    model.add(Dense(1, activation=activations.sigmoid, name="Dense_sigmoid"))
    # optimizer = optimizers.Adam(lr=1e-3)
    optimizer = optimizers.SGD(lr=1e-3, decay=0.1, momentum=0.9)
    loss = losses.binary_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model, model_name