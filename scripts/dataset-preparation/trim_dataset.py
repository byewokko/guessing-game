import os
import shutil

folder = "images"

for synset in os.listdir(folder):
    wnid, cls, name = synset.split(".")
    if cls not in ("artifact", "animal", "food", "plant"):
        print(f"removing {os.path.join(folder, synset)}")
        shutil.rmtree(os.path.join(folder, synset))
