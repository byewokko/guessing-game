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


def save_emb_gz(keys, values, filename, number_format=".6f"):
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


def make_categories(filenames):
    filenames_split = [f.split(os.path.sep)[-2] for f in filenames]
    uniq, cat = np.unique(filenames_split, return_inverse=True)
    return cat


def reduce_emb(x, n_comp):
    from sklearn.decomposition import PCA
    pca = PCA(n_comp)
    return pca.fit_transform(x)
