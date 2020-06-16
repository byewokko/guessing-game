# Disable GPU (because CPU starts faster)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import numpy as np
import game.game as game
# import agent.reinforce_agent as agent
import agent.q_agent as agent
from utils.dataprep import load_emb_gz, make_categories
from keras.optimizers import Adam, SGD, Adagrad
import matplotlib.pyplot as plt

log = logging.getLogger("game")
log.setLevel(logging.DEBUG)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


IMG_EMB_FILE = "data/esp-10000-vgg19.emb.gz"
N_SYMBOLS = 2
N_CHOICES = 2
EMB_SIZE = 50
N_IMAGES = 10
BATCH_SIZE = 30


_, fnames, embs = load_emb_gz(IMG_EMB_FILE, N_IMAGES)
IMG_SHAPE = embs[0].shape
IMG_N = len(embs)

# sender = agent.Sender(input_sizes=[IMG_SHAPE, IMG_SHAPE],
#                       output_size=N_SYMBOLS,
#                       n_symbols=N_SYMBOLS,
#                       embedding_size=50,
#                       learning_rate=0.001,
#                       gibbs_temp=10,
#                       use_bias=True,
#                       optimizer=Adam)
sender = agent.SenderInformed(input_sizes=[IMG_SHAPE, IMG_SHAPE],
                              output_size=N_SYMBOLS,
                              n_symbols=N_SYMBOLS,
                              n_filters=20,
                              embedding_size=50,
                              learning_rate=0.001,
                              gibbs_temp=10,
                              use_bias=True,
                              optimizer=Adam)
g = game.Game(images=embs,
              images_filenames=fnames,
              reward_sender={"success": 1, "fail": 0},
              reward_receiver={"success": 1, "fail": 0})

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plt.ion()

fig.show()
fig.canvas.draw()

t = []
show_steps = 2500
batch_success = []
success_rate = []
sendr_loss = []
recvr_loss = []
success_rate_avg = 0.5
sendr_loss_avg = None
recvr_loss_avg = None
for i in range(10000):
    g.reset()
    sender_state = g.get_sender_state(n_images=N_CHOICES)
    sample_ids = g.get_sample_ids()
    # This tests whether the sender is able to remember two arbitrary groups of images
    if np.all(sample_ids > N_IMAGES/2) or np.all(sample_ids <= N_IMAGES/2):
        correct_action = 0
    else:
        correct_action = 1
    sender_action, sender_prob = sender.act(sender_state)
    if sender_action == correct_action:
        success = True
        sender_reward = 1
    else:
        success = False
        sender_reward = 0
    sender.remember(sender_state, sender_action, sender_reward)
    batch_success.append(success)

    # TRAIN
    if i and not i % BATCH_SIZE:
        avg_success = sum(batch_success)/len(batch_success)
        batch_success = []
        sender.batch_train()
        # PLOT PROGRESS
        t.append(i)
        success_rate.append(avg_success)
        sendr_loss.append(sender.last_loss)
        ax1.clear()
        ax2.clear()
        ax1.plot(t[-show_steps:], sendr_loss[-show_steps:], "g")
        ax2.plot(t[-show_steps:], success_rate[-show_steps:], "r")
        fig.canvas.draw()
        fig.canvas.flush_events()
