import logging
import game.game as game
import agent.pg_agent as agent
from tools.tools import load_emb
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import keras.backend as K
from time import sleep

log = logging.getLogger("game")
log.setLevel(logging.DEBUG)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


IMG_EMB_FILE = "data/imgnet-emb1000-sample500.txt"
N_SYMBOLS = 10
N_IMAGES = 2
EMB_SIZE = 50

_, fnames, embs = load_emb(IMG_EMB_FILE)
IMG_SHAPE = embs[0].shape
IMG_N = len(embs)

sender = agent.Sender(input_sizes=[IMG_SHAPE, IMG_SHAPE],
                      output_size=N_SYMBOLS,
                      n_symbols=N_SYMBOLS,
                      embedding_size=50,
                      learning_rate=0.001,
                      gibbs_temp=10)
receiver = agent.Receiver(input_sizes=[IMG_SHAPE, IMG_SHAPE, (1,)],
                          output_size=N_IMAGES,
                          n_symbols=N_SYMBOLS,
                          embedding_size=50,
                          learning_rate=0.002,
                          gibbs_temp=10)
g = game.Game(images=embs,
              images_filenames=fnames,
              sender=sender,
              receiver=receiver)

avg_success = 0.5
for i in range(10000):
    avg_success += g.take_turn()
    avg_success /= 2
    if i % 100 == 0:
        print(avg_success, sender.last_loss, receiver.last_loss)
