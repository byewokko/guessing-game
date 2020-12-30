import pickle

import numpy as np
import os
import gzip


def save_emb(keys, values, filename, number_format=".6f"):
    assert len(keys) == len(values)
    number_format = f"{{n:{number_format}}}"
    out = gzip.open(filename, "wb")
    for i, (k, v) in enumerate(zip(keys, values)):
        if i and not i % 1000:
            print(f"{i} items written ...", end="\r")
        line = f"{k} " + " ".join([number_format.format(n=n) for n in v]) + "\n"
        out.write(line.encode("ascii"))
    print(f"DONE. {i+1} items written to {filename}.")
    out.close()


def save_emb_gz(keys, values, filename, number_format=".4e"):
    assert len(keys) == len(values)
    number_format = f"{{n:{number_format}}}"
    out = gzip.open(filename, "wb")
    for i, (k, v) in enumerate(zip(keys, values)):
        if i and not i % 1000:
            print(f"{i} items written ...", end="\r")
        line = f"{k} " + " ".join([number_format.format(n=n) for n in v]) + "\n"
        out.write(line.encode("ascii"))
    print(f"DONE. {i+1} items written to {filename}.")
    out.close()


def load_emb(filename, n_items=None, encoding="utf-8"):
    word2ind = {}
    ind2word = []
    embeddings = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if n_items and len(ind2word) >= n_items:
                break
            if not len(ind2word) % 1000:
                print(f"{len(ind2word)} items loaded ...", end="\r")
            word, *emb_str = line.strip().split()
            vector = np.asarray([float(s) for s in emb_str])
            word2ind[word] = len(word2ind)
            ind2word.append(word)
            embeddings.append(vector)
        print(f"DONE. {len(ind2word)} items loaded from {filename}.")
    return word2ind, ind2word, np.stack(embeddings)


def load_emb_gz(filename, n_items=None):
    word2ind = {}
    ind2word = []
    embeddings = []
    f = gzip.open(filename, "rb")
    for line in f:
        if n_items and len(ind2word) >= n_items:
            break
        if not len(ind2word) % 1000:
            print(f"{len(ind2word)} items loaded ...", end="\r")
        word, *emb_str = line.decode("ascii").strip().split()
        vector = np.asarray([float(s) for s in emb_str])
        word2ind[word] = len(word2ind)
        ind2word.append(word)
        embeddings.append(vector)
    f.close()
    print(f"DONE. {len(ind2word)} items loaded from {filename}.")
    return word2ind, ind2word, np.stack(embeddings)


def load_emb_pickled(filename):
    print(f"Loading features from {filename}.npy.gz")
    with gzip.open(filename + ".npy.gz", "rb") as f:
        embeddings = np.load(f)
    print(f"Loading metadata from {filename}.meta.pkl")
    with open(filename + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    uniq, classes = np.unique(meta["classes"], return_inverse=True)
    print(f"{len(uniq)} categories found.")
    meta["class_idx"] = classes
    meta["class_names"] = uniq
    return meta, embeddings


def make_categories(filenames, sep=None):
    """
    Extracts categories from a list of file paths, assuming
    the files are sorted in folders corresponding to their categories, e.g.:
    ["path/to/data/category1/image1.png",
    "path/to/data/category1/image2.png",
    "path/to/data/category2/image1.png"]

    Returns a np.array of category indices.
    """
    if sep is None:
        sep = os.path.sep
    if len(filenames[0].split(sep)) < 2:
        print("No categories found.")
        return None
    filenames_split = [f.split(sep)[-2] for f in filenames]
    uniq, cat = np.unique(filenames_split, return_inverse=True)
    print(f"{len(uniq)} categories found.")
    return cat


def split_dataset(data_length, ratio):
    ind = np.arange(data_length)
    np.random.shuffle(ind)
    ratio = np.asarray(ratio)
    ratio = ratio / ratio.sum() * data_length
    splits = [0]
    for n in ratio:
        splits.append(splits[-1] + n)
    out = [ind[ratio[i]:ratio[i + 1]] for i in range(len(ratio))]
