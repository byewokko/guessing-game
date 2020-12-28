import numpy as np
import logging
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
import gzip

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_PATH = "images"
IMG_SHAPE = (224, 224, 3)  # for VGG
OUT_NAME = "../../data/mcrae-wordnet-vgg16.emb"
OUT_DIM = 50
BATCH_SIZE = 8
FILES_ATLEAST = 100

print("loading model")
model = VGG16(weights='imagenet')

print("listing image files")
fnames = []
for root, subs, files in os.walk(IMG_PATH):
    for f in files:
        fnames.append(os.path.join(root, f).replace("\\", "/"))
print(f"{len(fnames)} images to process")

with open(OUT_NAME, "w") as out:
    pass

n_batches = len(fnames) // BATCH_SIZE + 1

for i in range(0, len(fnames), BATCH_SIZE):
    print(f"BATCH {i//BATCH_SIZE+1}/{n_batches}: ", end="")
    batch = []
    for j in range(i, min(i + BATCH_SIZE, len(fnames))):
        fname = fnames[j]
        img = image.load_img(fname, target_size=IMG_SHAPE[:2])
        x = image.img_to_array(img)
        batch.append(x)
    batch = np.stack(batch)
    print("preprocessing... ", end="")
    batch = preprocess_input(batch)
    print("predicting... ", end="")
    output = model.predict(batch, batch_size=BATCH_SIZE)
    print("saving... ", end="")
    with open(OUT_NAME, "a") as out:
        for j in range(i, min(i + BATCH_SIZE, len(fnames))):
            print(fnames[j], " ".join([f"{n:4e}" for n in output[j % BATCH_SIZE]]), file=out)
    print(f"done")
