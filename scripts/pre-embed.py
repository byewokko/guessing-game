"""
Extract image features
"""

import numpy as np
import logging
import os
from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, ResNet50
# from keras.applications.resnet_v2 import preprocess_input, ResNet50V2
# from keras.applications.xception import preprocess_input, Xception
from keras.applications.vgg19 import preprocess_input, VGG19

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.getLogger(__name__).addHandler(logging.NullHandler())
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

#IMG_PATH = "D:/Pracant/Desktop/ESPGame100k/originals"
IMG_PATH = "E:/TMP/imagenet_images"
IMG_SHAPE = (224, 224, 3)  # for VGG
# IMG_SHAPE = (299, 299, 3)  # for Xception
OUT_NAME = "../data/big/imagenet-227x80-vgg19.emb"
OUT_DIM = 50
BATCH_SIZE = 16

model = VGG19(weights='imagenet')
# model = Xception(weights='imagenet')
log.info("model loaded")

fnames = []
for root, subs, files in os.walk(IMG_PATH):
    for f in files:
        if f.lower().endswith(".jpg"):
            fnames.append(os.path.join(root, f))
log.info(f"{len(fnames)} images to process")

with open(OUT_NAME, "w") as out:
    pass

for i in range(0, len(fnames), BATCH_SIZE):
    log.info(f"processing batch {i//BATCH_SIZE+1}")
    batch = []
    for j in range(i, min(i + BATCH_SIZE, len(fnames))):
        fname = fnames[j]
        img = image.load_img(os.path.join(IMG_PATH, fname), target_size=IMG_SHAPE[:2])
        x = image.img_to_array(img)
        batch.append(x)
    batch = np.stack(batch)
    batch = preprocess_input(batch)
    output = model.predict(batch, batch_size=BATCH_SIZE)
    with open(OUT_NAME, "a") as out:
        for j in range(i, min(i + BATCH_SIZE, len(fnames))):
            print(fnames[j].replace(" ", "\\ "), " ".join([f"{n:5e}" for n in output[j%BATCH_SIZE]]), file=out)
    log.info(f"{i+BATCH_SIZE} images completed")

log.info("DONE")
