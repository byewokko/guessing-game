import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


def convert_to_variable(x, grad=True):
    return Variable(torch.FloatTensor(x), requires_grad=grad)


class Agent:
    input_type = "data"

    def __init__(self, input_sizes, output_size, n_symbols,
                 embedding_size=50, learning_rate=0.001, gibbs_temp=10,
                 optimizer=optim.Adam, use_bias=True,
                 **kwargs):
        self.n_input_images = 2
        self.input_sizes = input_sizes
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_symbols = n_symbols
        self.last_action = None
        self.last_loss = 0
        self.train_model = None
        self.predict_model = None
        self.learning_rate = learning_rate
        self.gibbs_temp = gibbs_temp
        self.optimizer = optimizer
        self.use_bias = use_bias

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError
        state = [np.expand_dims(st, 0) for st in state]
        act_probs = self.predict_model.predict(state)
        act_probs = np.squeeze(act_probs)
        action = np.random.choice(range(self.output_size), 1, p=act_probs)
        self.last_action = (state, action, act_probs)
        return action, act_probs

    def fit(self, state, action, reward):
        raise NotImplementedError
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        X = [np.expand_dims(st, 0) for st in state]
        Y = np.expand_dims(action_onehot, 0)
        self.last_loss = self.train_model.train_on_batch([*X, reward], Y)

    def load(self, name):
        raise NotImplementedError
        self.train_model.load_weights(name)

    def save(self, name):
        raise NotImplementedError
        self.train_model.save_weights(name)