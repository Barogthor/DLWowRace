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
import model
from constant import *


def main():
    train_ds, test_ds, races, train_amount, test_amount = dataset.load_dataset_tf()
    train_ds = train_ds.shuffle(1200).repeat().batch(BATCH_SIZE)
    test_ds = test_ds.shuffle(test_amount).repeat().batch(BATCH_SIZE)
    steps_per_epoch = ceil(train_amount / BATCH_SIZE)

    mdl, model_name = model.race_model_res_net_1(len(races))
    mdl.summary()
    if 1:
        tb_callback = TensorBoard('../logs/' + model_name)
        mdl.fit(train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=test_ds,
                validation_steps=ceil(test_amount / BATCH_SIZE), callbacks=[tb_callback])
        plot_model(mdl, '../models/' + model_name + '.png', show_shapes=True, show_layer_names=True)
        mdl.save('../models/' + model_name)
        # test_loss, test_acc = mdl.evaluate(test_ds, steps=ceil(test_amount / BATCH_SIZE))
        # print('Test accuracy:', test_acc)

    # predictions = model.predict(test_ds)


# tf.enable_eager_execution()
if __name__ == "__main__":
    main()

# plt.figure(figsize=(500, 500))
# plt.imshow(x_train[5])
# plt.show()
