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
	def __init__(self, temperature=1, **kwargs):
		self.temperature = temperature
		self.model, self.model_train = None, None
		self.memory_x = []
		self.memory_y = []

	def predict(self, state):
		x = [*state, np.array([self.temperature])]
		return self.model.predict_on_batch(x)

	def update(self, state, action, target):
		x = [*state, action, np.array(self.temperature)]
		return self.model_train.train_on_batch(x=x, y=target)

	def remember(self, state, action, target):
		x = [*state, action]
		self.memory_x.append(x)
		self.memory_y.append(target)

	def reset_memory(self):
		self.memory_x = []
		self.memory_y = []

	def update_on_batch(self, batch_size: int, reset_after=True, **kwargs):
		loss = []
		for x, y in zip(self.memory_x[-batch_size:], self.memory_y[-batch_size:]):
			x = [*x, np.array([self.temperature])]
			loss.append(self.model_train.train_on_batch(x=x, y=y))
		if reset_after:
			self.reset_memory()
		return loss

	def set_temperature(self, temperature):
		self.temperature = temperature

	def load(self, name: str):
		self.model.load_weights(name)

	def save(self, name: str):
		self.model.save_weights(name)
