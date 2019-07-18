import json
import ntpath
import os
import tensorflow as tf
from PIL import Image
import numpy as np

WIDTH = 340
HEIGHT = 425
CHANNEL = 3
RES_PATH = os.path.join('..', 'res')
JSON_PATH = os.path.join(RES_PATH, "dataset.json")
IMAGES_PATH = os.path.join(RES_PATH, 'img_resized')
allowed_races = ['Blood Elf', 'Night Elf', 'Troll']


def file_basename(path: str):
    return os.path.splitext(ntpath.basename(path))[0]


def open_image(path):
    image = Image.open(path)
    return np.array(image)


def load_images_tf(path, label):
    # image = open_image(path)
    # image = np.reshape(image, (HEIGHT, WIDTH, CHANNEL))
    image_string = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image_decoded, [HEIGHT, WIDTH])
    image = tf.cast(image, tf.float32) / 255.
    return image, label


def load_images(characters, races):
    images_ds = np.array([])
    labels_ds = np.array([]).astype(int)
    for character in characters:
        img = open_image(character["path"])
        images_ds = np.append(images_ds, img)
        labels_ds = np.append(labels_ds, races.index(character['race']))

    images_ds = np.reshape(images_ds, (len(labels_ds), HEIGHT, WIDTH, CHANNEL))
    return images_ds, labels_ds


def process_json():
    characters = np.array([])
    labels = np.array([])
    with open(JSON_PATH) as json_file:
        data = json.load(json_file)
        for character in data["characters"]:
            if character['race'] in allowed_races:
                character["path"] = os.path.join(IMAGES_PATH, str(character["id"]) + '.jpg')
                characters = np.append(characters, character)
                labels = np.append(labels, character['race'])

        json_file.close()
        return characters, list(dict.fromkeys(labels))


def load_dataset():
    characters, races = process_json()
    images_ds, labels_ds = load_images(characters[:50], races)

    return images_ds, labels_ds, races


def extracts_labels_and_paths(characters, races):
    train_labels = np.array([]).astype(int)
    train_paths = np.array([])
    test_labels = np.array([]).astype(int)
    test_paths = np.array([])
    for character in characters[:4400]:
        train_labels = np.append(train_labels, races.index(character['race']))
        train_paths = np.append(train_paths, character['path'])
    for character in characters[-111:]:
        test_labels = np.append(test_labels, races.index(character['race']))
        test_paths = np.append(test_paths, character['path'])

    return train_labels, train_paths, test_labels, test_paths


def load_dataset_tf():
    characters, races = process_json()
    train_labels, train_paths, test_labels, test_paths = extracts_labels_and_paths(characters, races)

    train_labels = tf.constant(train_labels)
    train_paths = tf.constant(train_paths)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(load_images_tf)

    test_labels = tf.constant(test_labels)
    test_paths = tf.constant(test_paths)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_dataset = test_dataset.map(load_images_tf)

    return train_dataset, test_dataset, races, len(characters[:4400]), len(characters[-111:])
