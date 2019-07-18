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
from math import *
import dataset

WIDTH = 340
HEIGHT = 425
CHANNEL = 3
BATCH_SIZE = 40
STEPS_PER_EPOCH = 0
EPOCHS = 30


def race_model(num_classes):
    model = Sequential()
    # model.add(Flatten(input_shape=( WIDTH, HEIGHT, CHANNEL)))
    model.add(InputLayer(input_shape=(HEIGHT, WIDTH, 3,)))
    # model.add(Reshape(target_shape=(10, HEIGHT, WIDTH, CHANNEL)))
    model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(512, activation=activations.relu))
    model.add(Dropout(rate=0.2))
    model.add(Dense(num_classes, activation=activations.softmax))
    model.add(Dense(1, activation=activations.sigmoid))
    # optimizer = optimizers.Adam(lr=1e-3)
    optimizer = optimizers.SGD(lr=1e-3, decay=0.1)
    loss = losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def main():

    # plt.figure(figsize=(500, 500))
    # plt.imshow(x_train[5])
    # plt.show()
    train_ds, test_ds, races, train_amount, test_amount = dataset.load_dataset_tf()
    train_ds = train_ds.shuffle(1200).repeat().batch(BATCH_SIZE)
    test_ds = test_ds.shuffle(test_amount).repeat().batch(BATCH_SIZE)
    STEPS_PER_EPOCH = ceil(train_amount/BATCH_SIZE)
    model_name = 'mdl_catent_sgd_test_dataset'
    # for rnum in enumerate(my_ds.take(1)):
    model = race_model(len(races))
    model.summary()
    tb_callback = TensorBoard('../logs/' + model_name)
    model.fit(train_ds, epochs=5, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[tb_callback])
    plot_model(model, '../models/' + model_name + '.png', show_shapes=True, show_layer_names=True)
    model.save('../models/' + model_name)
    test_loss, test_acc = model.evaluate(test_ds, steps=ceil(test_amount/BATCH_SIZE))

    print('Test accuracy:', test_acc)
    # predictions = model.predict(test_ds)


# tf.enable_eager_execution()
if __name__ == "__main__":
    main()
