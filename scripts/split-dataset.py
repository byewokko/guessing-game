"""
Splits one big categorized dataset into three smaller ones so that:
  - the ctg test set consists of ~test_ctgs~ number of whole categories,
  - the remaining categories are split between the training set and img test set:
    - the img test set gets ~test_imgs~ number of images of each category,
    - the training set gets all the remaining images.
The resulting files are gz-compressed.
"""

import gzip

data_file = "../data/big/imagenet-227x80-vgg19.emb"
train_file = "../data/imagenet-200x65-vgg19.train.emb.gz"
test_ctg_file = "../data/imagenet-27x80-vgg19.test-ctg.emb.gz"
test_img_file = "../data/imagenet-200x15-vgg19.test-img.emb.gz"
test_imgs = 15
test_ctgs = 27

with open(train_file, "w"), open(test_ctg_file, "w"), open(test_img_file, "w"):
    pass

train_out = gzip.open(train_file, "w")
test_ctg_out = gzip.open(test_ctg_file, "w")
test_img_out = gzip.open(test_img_file, "w")
train_count = 0
test_ctg_count = 0
test_img_count = 0

with open(data_file) as fin:
    img_buff = []
    ctgs = []
    for line in fin:
        line = line.strip().replace("\\ ", "_")
        ctg = line.split("\\")[-2]
        if not ctgs:
            ctgs.append(ctg)
        elif ctgs[-1] != ctg:
            if len(ctgs) <= test_ctgs:
                test_ctg_count += len(img_buff)
                test_ctg_out.writelines(img_buff)
            else:
                test_img_count += len(img_buff[:test_imgs])
                train_count += len(img_buff[test_imgs:])
                test_img_out.writelines(img_buff[:test_imgs])
                train_out.writelines(img_buff[test_imgs:])
            ctgs.append(ctg)
            img_buff.clear()
        img_buff.append(line.encode("ascii"))
    train_out.close()
    test_ctg_out.close()
    test_img_out.close()

print(f"Training images: {train_count}")
print(f"Test images (new categories): {test_ctg_count}")
print(f"Test images (training categories): {test_img_count}")
