import json
import ntpath
import os
from PIL import Image
import numpy as np

WIDTH = 340
HEIGHT = 425
CHANNEL = 3
RES_PATH = os.path.join('..', 'res')
JSON_PATH = os.path.join(RES_PATH, "dataset.json")
IMAGES_PATH = os.path.join(RES_PATH, 'img_resized')


def file_basename(path: str):
    return os.path.splitext(ntpath.basename(path))[0]


def open_image(path):
    image = Image.open(path)
    return np.array(image)


def load_images(characters, races):
    images_ds = np.array([])
    labels_ds = np.array([]).astype(int)
    for character in characters:
        img = open_image(character["path"])
        images_ds = np.append(images_ds, img)
        labels_ds = np.append(labels_ds, races.index(character['race']))
        # print(character['race'])

    images_ds = np.reshape(images_ds, (len(labels_ds), HEIGHT, WIDTH, CHANNEL))
    return images_ds, labels_ds


def process_json():
    characters = []
    labels = np.array([])
    with open(JSON_PATH) as json_file:
        data = json.load(json_file)
        for character in data["characters"]:
            character["path"] = os.path.join(IMAGES_PATH, str(character["id"]) + '.jpg')
            labels = np.append(labels, character['race'])
        characters = data["characters"]
        json_file.close()
        return characters, list(dict.fromkeys(labels))


def load_dataset():
    characters, races = process_json()
    images_ds, labels_ds = load_images(characters[:50], races)

    return images_ds, labels_ds, races
