import numpy as np
import matplotlib.pyplot as plt


def show_images(images, rows=1, titles=None):
    """Display a list of images in a single figure with matplotlib.
    SRC: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    rows: Number of rows in figure (number of cols is
                        set to np.ceil(n_images/float(rows))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, np.ceil(n_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


class Manual:
    """
    Interface for human input. Requires Jupyter Notebook environment to show images properly.
    """
    input_type = "filename"

    def __init__(self, n_symbols):
        self.n_symbols = n_symbols

    def act_send(self, images):
        show_images(images, 1)
        clue = -1
        while not (0 <= clue < self.n_symbols):
            clue = int(input(f"Clue: "))
        return clue

    def act_receive(self, images, clue):
        show_images(images, 1)
        guess = -1
        while guess < 0 or guess >= len(images):
            guess = int(input(f"Clue: {clue}. Guess: "))
        return guess

    def reward_send(self, reward):
        pass

    def reward_receive(self, reward):
        pass


class Random:
    """
    Provides random input. Available actions are sampled uniformly.
    """
    input_type = None

    def __init__(self, n_symbols):
        self.n_symbols = n_symbols

    def act_send(self, images):
        clue = np.random.randint(0, self.n_symbols)
        return clue

    def act_receive(self, images, clue):
        guess = np.random.randint(0, len(images))
        return guess

    def reward_send(self, reward):
        pass

    def reward_receive(self, reward):
        pass
