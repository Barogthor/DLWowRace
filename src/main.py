from tensorflow.python.keras.datasets import *
from keras import activations
from keras import optimizers
from keras import losses
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.callbacks import *

from keras import utils
from tensorflow.python.keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import json
import dataset

WIDTH = 340
HEIGHT = 425
CHANNEL = 3


def race_model():
    model = Sequential()
    # model.add(Flatten(input_shape=( WIDTH, HEIGHT, CHANNEL)))
    model.add(InputLayer(input_shape=(HEIGHT, WIDTH, 3,)))
    # model.add(Reshape(target_shape=(10, HEIGHT, WIDTH, CHANNEL)))
    model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=24, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation=activations.relu))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, activation=activations.softmax))
    optimizer = optimizers.Adam(lr=1e-3)
    loss = losses.binary_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def main():
    x_train, y_train, races = dataset.load_dataset()
    x_train = x_train / 255.
    print(y_train)
    print(races)
    # plt.figure(figsize=(500, 500))
    # plt.imshow(x_train[5])
    # plt.show()

    model_name = 'mdl_binent_adam_test'
    # for image in image_ds.take(1):
    #     print(image)
    model = race_model()
    model.summary()
    tb_callback = TensorBoard('../logs/' + model_name)
    model.fit(x_train, y_train, epochs=100, callbacks=[tb_callback])
    plot_model(model, '../models/' + model_name + '.png', show_shapes=True, show_layer_names=True)
    model.save('./models/' + model_name)


# tf.enable_eager_execution()
if __name__ == "__main__":
    main()
