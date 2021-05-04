import logging
import numpy as np


L = logging.getLogger(__name__)


def getshape(array):
	if isinstance(array, list):
		return [getshape(x) for x in array]
	else:
		try:
			return array.shape
		except Exception as e:
			L.error(f"Cannot get shape of array: {e}")
			return None


class Agent:
	def __init__(self, **kwargs):
		self.model, self.model_train = None, None
		self.memory_x = []
		self.memory_y = []

	def predict(self, state):
		x = state
		return self.model.predict_on_batch(x)

	def choose_action(self, probs):
		return np.random.choice(
			np.arange(len(probs)),
			p=probs
		)

	def update(self, state, action, target):
		x = [*state, action]
		return self.model_train.train_on_batch(x=x, y=target)

	def remember(self, state, action, action_probs, reward):
		x = [*state, action]
		self.memory_x.append(x)
		self.memory_y.append(reward)

	def reset_memory(self):
		self.memory_x = []
		self.memory_y = []

	def update_on_batch(self, batch_size: int, reset_after=True, **kwargs):
		loss = []
		for x, y in zip(self.memory_x[-batch_size:], self.memory_y[-batch_size:]):
			loss.append(self.model_train.train_on_batch(x=x, y=y))
		if reset_after:
			self.reset_memory()
		return loss

	def load(self, name: str):
		self.model.load_weights(name)

	def save(self, name: str):
		self.model.save_weights(name)


class MultiAgent(Agent):
	def __init__(self, active_role, **kwargs):
		self.components = {}
		if active_role not in ("sender", "receiver"):
			raise ValueError(f"Role must be either 'sender' or 'receiver', not '{active_role}'.")
		self.active_role = active_role

	def switch_role(self):
		if self.active_role == "sender":
			self.active_role = "receiver"
		else:
			self.active_role = "sender"
