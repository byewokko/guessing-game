import numpy as np
import pickle
import gzip

FILE_NAME = "../../data/big/mcrae-wordnet-vgg16.emb"

wnids = []
classes = []
weights = []
fnames = []

print(f"reading source")
with open(FILE_NAME, "r") as fin:
    for line in fin:
        name, numbers = line.split(" ", 1)
        wnid, cls, _ = name.split("/")[-2].split(".")
        fname = name.split("/")[-1]
        wnids.append(wnid)
        fnames.append(fname)
        classes.append(cls)
        weights.append(np.fromstring(numbers, dtype=float, sep=" "))

weights = np.stack(weights)

print(f"saving weights")
with gzip.open(FILE_NAME+".npy.gz", "wb") as out:
    np.save(out, weights.astype("float32"))

print(f"saving metadata")
with open(FILE_NAME+".meta.pkl", "wb") as out:
    pickle.dump({"categories": wnids, "classes": classes, "fnames": fnames}, out, protocol=pickle.HIGHEST_PROTOCOL)
