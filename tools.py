import numpy as np
import sys
import logging
import os
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, ResNet50


def load_images(path, return_fnames=False):
    images = []
    fnames = os.listdir(path)
    for fname in fnames:
        img = image.load_img(os.path.join(path, fname), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, 0)
        x = preprocess_input(x)
        images.append(x)
    if return_fnames:
        return [os.path.join(path, fname) for fname in fnames], images
    return images


def embed_images(img_stack):
    model = ResNet50(weights='imagenet')
    img_out = []
    for img in img_stack:
        img_out.append(np.squeeze(model.predict(img)))
    return np.stack(img_out)


def save_emb(keys, values, filename):
    assert len(keys) == len(values)
    with open(filename, "w") as out:
        for k, v in zip(keys, values):
            print(k, " ".join([f"{n:5e}" for n in v]), file=out)


def load_emb(filename):
    word2ind = {}
    ind2word = []
    embeddings = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            word, *emb_str = line.strip().split()
            vector = np.asarray([float(s) for s in emb_str])
            word2ind[word] = len(word2ind)
            ind2word.append(word)
            embeddings.append(vector)
    return word2ind, ind2word, np.stack(embeddings)


def reduce_emb(x, n_comp):
    from sklearn.decomposition import PCA
    pca = PCA(n_comp)
    return pca.fit_transform(x)
