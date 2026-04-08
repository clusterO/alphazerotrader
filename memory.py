import numpy as np
from collections import deque
import pickle
import os

import config

class Memory:
	def __init__(self, MEMORY_SIZE):
		self.MEMORY_SIZE = MEMORY_SIZE
		self.ltmemory = deque(maxlen=MEMORY_SIZE)
		self.stmemory = deque(maxlen=MEMORY_SIZE)
		self.metadata = {} # New persistent metadata field

	def save(self, path):
		with open(path, 'wb') as f:
			pickle.dump(self, f)

	def load(self, path):
		if os.path.exists(path):
			with open(path, 'rb') as f:
				return pickle.load(f)
		return self

	def commit_stmemory(self, identities, state, actionValues):
		for r in identities(state, actionValues):
			self.stmemory.append({
				'board': r[0].board
				, 'state': r[0]
				, 'id': r[0].id
				, 'AV': r[1]
				, 'playerTurn': r[0].playerTurn
				})

	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)
		