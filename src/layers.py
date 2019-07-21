import numpy as np
from keras import activations
from keras import optimizers
from keras import losses
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
import math
from constant import *


def head_section(x):
    top_offset = math.ceil(HEIGHT * 0.15)
    height_section = math.ceil(HEIGHT * 0.35)
    left_offset = math.floor(WIDTH * 0.25)
    return x[:, top_offset:height_section + top_offset, left_offset:-left_offset, :]


def left_arm_section(x):
    left_offset = math.floor(WIDTH * 0.15)
    width_section = math.ceil(WIDTH * 0.25)
    return x[:, :, left_offset:width_section + left_offset, :]


def right_arm_section(x):
    right_offset = math.floor(WIDTH * 0.15)
    width_section = math.ceil(WIDTH * 0.25)
    return x[:, :, -(right_offset + width_section):-right_offset, :]


# 70%, 50%
def body_section(x):
    top_offset = math.ceil(WIDTH * 0.3)
    left_offset = math.floor(WIDTH * 0.25)
    return x[:, top_offset:, left_offset:-left_offset, :]


def merge_section(tensors):
    left_arm, body, right_arm = tensors[0], tensors[1], tensors[2]
    x = np.concatenate((left_arm, body, right_arm))
    return x


def head_layers(input_tensor, num_classes):
    head_layer = Lambda(head_section, name="Cut_head_section")(input_tensor)
    head_layer = Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu',
                        name="Conv2D_64f_head")(head_layer)
    head_layer = AveragePooling2D(pool_size=3, padding='same')(head_layer)
    head_layer = Dropout(0.25)(head_layer)
    head_layer = Flatten()(head_layer)
    head_layer = Dense(384, activation=activations.linear, name="Dense_384_head")(head_layer)
    head_layer = LeakyReLU(name="LeakyReLU_head")(head_layer)
    head_layer = Dense(192, activation=activations.sigmoid, name="Dense_192_sigmoid_head")(head_layer)
    head_layer = Dense(num_classes, activation=activations.softmax, name=f"Dense_{num_classes}_classes_head")(
        head_layer)
    return head_layer


def body_layers(input_tensor, num_classes):
    body_layer = Lambda(body_section, name="Cut_body_section")(input_tensor)
    body_layer = Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu',
                        name="Conv2D_64f_body")(body_layer)
    body_layer = AveragePooling2D(pool_size=3, padding='same')(body_layer)
    body_layer = Dropout(0.25)(body_layer)
    body_layer = Flatten()(body_layer)
    body_layer = Dense(512, activation=activations.linear, name="Dense_512_body")(body_layer)
    body_layer = LeakyReLU(name="LeakyReLU_body")(body_layer)
    body_layer = Dense(256, activation=activations.sigmoid, name="Dense_256_sigmoid_body")(body_layer)
    body_layer = Dense(num_classes, activation=activations.softmax, name=f"Dense_{num_classes}_classes_body")(
        body_layer)
    return body_layer


def left_layers(input_tensor, num_classes):
    left_arm_layer = Lambda(left_arm_section, name="Cut_left_arm_section")(input_tensor)
    left_arm_layer = Conv2D(32, kernel_size=5, strides=3, padding='same', activation='relu',
                            name="Conv2D_64f_left")(left_arm_layer)
    left_arm_layer = AveragePooling2D(pool_size=2, padding='same')(left_arm_layer)
    left_arm_layer = Dropout(0.25)(left_arm_layer)
    left_arm_layer = Flatten()(left_arm_layer)
    left_arm_layer = Dense(128, activation=activations.linear, name="Dense_128_left")(left_arm_layer)
    left_arm_layer = LeakyReLU(name="LeakyReLU_left")(left_arm_layer)
    left_arm_layer = Dense(64, activation=activations.sigmoid, name="Dense_128_sigmoid_left")(left_arm_layer)
    left_arm_layer = Dense(num_classes, activation=activations.softmax, name=f"Dense_{num_classes}_classes_left")(
        left_arm_layer)
    return left_arm_layer


def right_layers(input_tensor, num_classes):
    right_arm_layer = Lambda(right_arm_section, name="Cut_right_arm_section")(input_tensor)
    right_arm_layer = Conv2D(32, kernel_size=5, strides=3, padding='same', activation='relu',
                             name="Conv2D_64f_right")(right_arm_layer)
    right_arm_layer = AveragePooling2D(pool_size=2, padding='same')(right_arm_layer)
    right_arm_layer = Dropout(0.25)(right_arm_layer)
    right_arm_layer = Flatten()(right_arm_layer)
    right_arm_layer = Dense(128, activation=activations.linear, name="Dense_128_right")(right_arm_layer)
    right_arm_layer = LeakyReLU(name="LeakyReLU_right")(right_arm_layer)
    right_arm_layer = Dense(64, activation=activations.sigmoid, name="Dense_128_sigmoid_right")(right_arm_layer)
    right_arm_layer = Dense(num_classes, activation=activations.softmax, name=f"Dense_{num_classes}_classes_right")(
        right_arm_layer)
    return right_arm_layer
