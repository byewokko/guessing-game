import logging
import game.game as game
import agent.basic_reinforce as agent
from tools.tools import load_emb
import tensorflow as tf

log = logging.getLogger("game")
log.setLevel(logging.DEBUG)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

IMG_EMB_FILE = "data/imgnet-emb1000-sample500.txt"
N_SYMBOLS = 10
N_IMAGES = 2
EMB_SIZE = 50

_, fnames, embs = load_emb(IMG_EMB_FILE)
IMG_SHAPE = embs[0].shape
IMG_N = len(embs)

sender = agent.Sender(img_shape=IMG_SHAPE,
                      n_images=N_IMAGES,
                      n_symbols=N_SYMBOLS,
                      embedding_size=EMB_SIZE)
receiver = agent.Receiver(img_shape=IMG_SHAPE,
                          n_images=N_IMAGES,
                          n_symbols=N_SYMBOLS,
                          embedding_size=EMB_SIZE)
g = game.Game(images=embs,
              images_filenames=fnames,
              sender=sender,
              receiver=receiver)

print(g.take_turn())
