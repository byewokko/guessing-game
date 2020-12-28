import os

folder = "images"

stats = {}
other = {"synsets": 0, "images": 0}
major = {"synsets": 0, "images": 0}

for synset in os.listdir(folder):
    wnid, cls, name = synset.split(".")
    if cls not in stats:
        stats[cls] = {"synsets": 0, "images": 0}
    stats[cls]["synsets"] += 1
    images =  len(os.listdir(os.path.join(folder, synset)))
    stats[cls]["images"] += images

print(f"Class name\tNr. synsets\tNr. images")

for cls, st in sorted(stats.items(), key=lambda x: x[1]["synsets"], reverse=True):
    if st['synsets'] < 10:
        other['synsets'] += st['synsets']
        other['images'] += st['images']
    else:
        print(f"{cls}\t{st['synsets']}\t{st['images']}")
        major['synsets'] += st['synsets']
        major["images"] += st['images']

print(f"major\t{major['synsets']}\t{major['images']}")
print(f"other\t{other['synsets']}\t{other['images']}")
