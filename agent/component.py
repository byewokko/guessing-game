import numpy as np

MAX_MEMORY = 2000


class Net:
    def __init__(self, input_shapes=None, output_size=None, max_memory=MAX_MEMORY, **kwargs):
        self.input_shapes = input_shapes
        self.output_size = output_size
        self.model = None
        self.batch_states = None
        self.batch_actions = None
        self.batch_rewards = None
        self.last_loss = None
        self.memory_states = None
        self.memory_actions = None
        self.memory_rewards = None
        self.max_memory = max_memory
        self.memory_sampling_dist = None

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
            self.memory_states[i].append(state[i])
        action_onehot = np.zeros([self.output_size])
        action_onehot[action] = 1
        self.memory_actions.append(action_onehot)
        self.memory_rewards.append(reward)

    def trim_memory(self, length=None):
        if not length:
            length = self.max_memory
        for i in range(len(self.memory_states)):
            self.memory_states[i] = self.memory_states[i][-length:]
        self.memory_rewards = self.memory_rewards[-length:]
        self.memory_actions = self.memory_actions[-length:]

    def make_distribution(self, mode: str = ""):
        d = np.linspace(0, 1, self.max_memory)
        if not mode or mode == "linear":
            self.memory_sampling_dist = d / d.sum()
        elif mode == "quadratic":
            d = d * d
            self.memory_sampling_dist = d / d.sum()
        else:
            raise ValueError(f"Invalid mode: '{mode}'")

    def prepare_batch(self, size: int, batch_mode: str = "last"):
        self.reset_batch()
        self.trim_memory()
        assert self.memory_sampling_dist is not None
        if batch_mode == "sample":
            if len(self.memory_rewards) == self.max_memory:
                indices = np.random.choice(np.arange(self.max_memory), size, p=self.memory_sampling_dist)
                for i in indices:
                    for s in range(len(self.batch_states)):
                        self.batch_states[s].append(self.memory_states[s][i])
                    self.batch_actions.append(self.memory_actions[i])
                    self.batch_rewards.append(self.memory_rewards[i])
            else:
                for s in range(len(self.batch_states)):
                    self.batch_states[s] = self.memory_states[s][-size:]
                self.batch_actions = self.memory_actions[-size:]
                self.batch_rewards = self.memory_rewards[-size:]
        elif batch_mode == "last":
            for s in range(len(self.batch_states)):
                self.batch_states[s] = self.memory_states[s][-size:]
            self.batch_actions = self.memory_actions[-size:]
            self.batch_rewards = self.memory_rewards[-size:]

    def reset_batch(self):
        self.batch_states = [[] for _ in self.input_shapes]
        self.batch_actions = []
        self.batch_rewards = []

    def reset_memory(self):
        self.memory_states = [[] for _ in self.input_shapes]
        self.memory_actions = []
        self.memory_rewards = []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
