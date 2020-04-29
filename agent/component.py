import numpy as np


class Net:
    def __init__(self, input_shapes=None, output_size=None, **kwargs):
        self.input_shapes = input_shapes
        self.output_size = output_size
        self.model = None
        self.batch_states = None
        self.batch_actions = None
        self.batch_rewards = None
        self.last_loss = None

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def train_on_batch(self, *args, **kwargs):
        return self.model.train_on_batch(*args, **kwargs)

    def batch_train(self):
        q_values = self.model.predict(self.batch_states)
        for i in range(len(self.batch_rewards)):
            q_values[i][self.batch_actions[i].astype("bool")] = self.batch_rewards[i]
        self.last_loss = self.model.train_on_batch(
            self.batch_states,
            q_values
        )
        self.reset_batch()
        return self.last_loss

    def remember(self, state, action, reward):
        for i in range(len(state)):
            self.batch_states[i].append(state[i])
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        self.batch_actions.append(action_onehot)
        self.batch_rewards.append(reward)

    def reset_batch(self):
        self.batch_states = [[] for _ in self.input_shapes]
        self.batch_actions = []
        self.batch_rewards = []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
