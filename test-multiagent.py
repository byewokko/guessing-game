# Disable GPU (because CPU starts faster)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import numpy as np
import game.game as game
import agent.q_agent as agent
# import agent.q_agent as agent
from utils.embeddings import load_emb_gz, make_categories
from keras.optimizers import Adam, SGD, Adagrad
import matplotlib.pyplot as plt

log = logging.getLogger("game")
log.setLevel(logging.DEBUG)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


IMG_EMB_FILE = "data/imagenet-4000-vgg19.emb.gz"
# IMG_EMB_FILE = "data/esp-10000-esp-vgg19.emb.gz"
# IMG_EMB_FILE = "data/esp-10000-esp-xception.emb.gz"
N_SYMBOLS = 10
N_CHOICES = 2
EMB_SIZE = 50
N_IMAGES = None
BATCH_SIZE = 30
N_EPISODES = 10000

_, fnames, embs = load_emb_gz(IMG_EMB_FILE, N_IMAGES)
categories = make_categories(fnames)
IMG_SHAPE = embs[0].shape
IMG_N = len(embs)

args = {
    "input_shapes": [IMG_SHAPE] * N_CHOICES + [(1,)],
    "output_size": N_SYMBOLS,
    "n_symbols": N_SYMBOLS,
    "embedding_size": 50,
    "learning_rate": 0.01,
    "optimizer": Adam
}

agent1 = agent.MultiAgent(role="sender", **args)
# sender = agent.SenderInformed(input_sizes=[IMG_SHAPE] * N_CHOICES,
#                               output_size=N_SYMBOLS,
#                               n_symbols=N_SYMBOLS,
#                               n_filters=20,
#                               embedding_size=50,
#                               learning_rate=0.02,
#                               gibbs_temp=10,
#                               use_bias=False,
#                               optimizer=Adam)
agent2 = agent.MultiAgent(role="receiver", **args)

g = game.Game(images=embs,
              images_filenames=fnames,
              categories=categories,
              reward_sender={"success": 1, "fail": 0},
              reward_receiver={"success": 1, "fail": 0})

fig = plt.figure()
plt.subplots_adjust(hspace=0.65)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plt.ion()

fig.show()
fig.canvas.draw()

t = []
show_steps = 2500
batch_success = []
success_rate = []
sendr1_loss = []
recvr1_loss = []
sendr2_loss = []
recvr2_loss = []
success_rate_avg = []
sendr_loss_avg = None
recvr_loss_avg = None
explore = "gibbs"
sender = agent1
receiver = agent2
for i in range(N_EPISODES):
    g.reset()
    sender_state = g.get_sender_state(n_images=N_CHOICES, unique_categories=True)
    sender_action, _ = sender.act(sender_state, explore=explore)
    receiver_state = g.get_receiver_state(sender_action)
    receiver_action, _ = receiver.act(receiver_state, explore=explore)
    sender_reward, receiver_reward, success = g.evaluate_guess(receiver_action)
    sender.remember(sender_state, sender_action, sender_reward)
    receiver.remember(receiver_state, receiver_action, receiver_reward)
    batch_success.append(success)

    if i == 6000:
        explore = False
    # TRAIN
    if i and not i % BATCH_SIZE:
        avg_success = sum(batch_success) / len(batch_success)
        batch_success = []
        sender.batch_train()
        receiver.batch_train()

        sender.switch_role()
        receiver.switch_role()
        tmp = sender
        sender = receiver
        receiver = tmp

        # PLOT PROGRESS
        t.append(i)
        # success_rate_avg = smooth_avg(success_rate_avg, avg_success, 0.1)
        # sendr_loss_avg = smooth_avg(sendr_loss_avg, sender.last_loss, 0.1)
        # recvr_loss_avg = smooth_avg(recvr_loss_avg, receiver.last_loss, 0.1)
        success_rate.append(avg_success)
        success_rate_avg.append(sum(success_rate[-10:]) / min(10, len(success_rate)))
        sendr1_loss.append(agent1.net["sender"].last_loss)
        sendr2_loss.append(agent2.net["sender"].last_loss)
        recvr1_loss.append(agent1.net["receiver"].last_loss)
        recvr2_loss.append(agent2.net["receiver"].last_loss)
        #       print(success, sender.last_loss, receiver.last_loss)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.set_title("Sender loss")
        ax2.set_title("Receiver loss")
        ax3.set_title("Average batch success rate")
        ax1.plot(t[-show_steps:], sendr1_loss[-show_steps:], "m", label="Agent 1")
        ax1.plot(t[-show_steps:], sendr2_loss[-show_steps:], "c", label="Agent 2")
        ax1.legend(loc="lower left")
        ax2.plot(t[-show_steps:], recvr1_loss[-show_steps:], "m", label="Agent 1")
        ax2.plot(t[-show_steps:], recvr2_loss[-show_steps:], "c", label="Agent 2")
        ax2.legend(loc="lower left")
        ax3.plot(t[-show_steps:], success_rate[-show_steps:], "r.")
        ax3.plot(t[-show_steps:], success_rate_avg[-show_steps:], "k")
        fig.canvas.draw()
        fig.canvas.flush_events()
