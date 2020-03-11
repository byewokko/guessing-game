import logging
import game.game as game
import agent.pg_agent as agent
from tools.tools import load_emb
from keras.optimizers import Adam, SGD, Adagrad

log = logging.getLogger("game")
log.setLevel(logging.DEBUG)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


IMG_EMB_FILE = "data/espgame-resnet50-1920.txt"
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
                      gibbs_temp=50,
                      use_bias=False,
                      optimizer=Adam)
receiver = agent.Receiver(input_sizes=[IMG_SHAPE, IMG_SHAPE, (1,)],
                          output_size=N_IMAGES,
                          n_symbols=N_SYMBOLS,
                          embedding_size=50,
                          learning_rate=0.001,
                          gibbs_temp=50,
                          mode="dot",  # original with dot product output
                          # mode="dense", # dense layer + sigmoid instead
                          use_bias=False,
                          optimizer=Adam)
g = game.Game(images=embs,
              images_filenames=fnames,
              sender=sender,
              receiver=receiver,
              reward={"success": 1, "fail": 0})

avg_success = 0.5
for i in range(10000):
    avg_success += g.take_turn()
    avg_success /= 2
    if i % 100 == 0:
        print(avg_success, sender.last_loss, receiver.last_loss)
