import os

import requests
from nltk.corpus import wordnet as wn

urldir = "urls"
geturls = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}"

if not os.path.isdir(urldir):
    os.makedirs(urldir)

with open("base_concepts.txt") as fin:
    for line in fin:
        concept = line.strip().split("_")[0]
        print("===", concept)
        syns = wn.synsets(concept, pos=wn.NOUN)
        available = []
        for synset in syns:
            category = synset.lexname().split(".")[-1]
            name = synset.name().split(".")[0]
            offset = synset.offset()
            wnid = f"n{offset:08d}"
            print(f"{wnid}.{category}.{name}")
            r = requests.get(geturls.format(wnid=wnid))
            if "\n" not in r.text:
                continue
            urls = r.text.split()
            if len(urls) < 100:
                continue
            filename = os.path.join(urldir, f"{wnid}.{category}.{name}.{len(urls)}.txt")
            available.append((filename, len(urls), urls))
        if not available:
            continue
        available.sort(key=lambda x: x[1], reverse=True)
        filename, _, urls = available[0]
        print(f"BEST: {filename}")
        with open(filename, "w", encoding="utf-8") as fout:
            for url in urls:
                try:
                    print(url, file=fout)
                except Exception as e:
                    print(type(e), url)
