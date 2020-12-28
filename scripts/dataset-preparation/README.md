`base_concepts.txt` contains a list of concepts introduced in McRae et al. (2005)

`get_image_urls.py` pairs the base concepts with WordNet synsets 
and retrieves their image URLs using image-net.org API, writing them in `urls/`

`download_images.py` reads URLs from `urls/` and downloads the images, saving them in `images/`

`print_stats.py` prints stats about the dataset in `images/`

`trim_dataset.py` removes small classes

`extract_features.py` embeds the images using imagenet-pretrained VGG16 model

`pickle_features.py` compresses the embedding file and prepares for faster loading
