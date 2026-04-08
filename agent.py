# %matplotlib inline

import numpy as np
import random

import MCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl


class User():
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		action = input('Enter your chosen action: ')
		pi = np.zeros(self.action_size)
		pi[action] = 1
		value = None
		NN_value = None
		return (action, pi, value, NN_value)



class Agent():
	def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size
		self.cpuct = cpuct
		self.MCTSsimulations = mcts_simulations
		self.model = model
		self.mcts = None

		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []
		
		import yaml
		with open("config.yaml", 'r') as f:
			self.cfg = yaml.safe_load(f)

	def simulate(self):
		# Legacy single simulation - replaced by act()'s batched logic
		##### MOVE THE LEAF NODE
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
		##### EVALUATE THE LEAF NODE
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)
		##### BACKFILL THE VALUE THROUGH THE TREE
		self.mcts.backFill(leaf, value, breadcrumbs)

	def act(self, state, tau):
		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)

		# PHASE 4.2: BATCHED MCTS
		batch_size = self.cfg['rl'].get('mcts_batch_size', 8)
		num_batches = max(1, self.MCTSsimulations // batch_size)

		for b in range(num_batches):
			if b % 4 == 0: print(".", end="", flush=True)
			
			batch_leaves = []
			batch_breadcrumbs = []
			
			# 1. Selection with Virtual Loss
			for _ in range(batch_size):
				leaf, value, done, breadcrumbs = self.mcts.moveToLeaf(apply_virtual_loss=True)
				batch_leaves.append((leaf, value, done))
				batch_breadcrumbs.append(breadcrumbs)
			
			# 2. Batched Evaluation
			states_to_eval = []
			eval_indices = []
			
			for i, (leaf, value, done) in enumerate(batch_leaves):
				# Expansion Guard: Only expand if not already terminal
				if done == 0 and not leaf.state.isEndGame:
					states_to_eval.append(self.model.convertToModelInput(leaf.state))
					eval_indices.append(i)
			
			if states_to_eval:
				input_tensor = np.array(states_to_eval, dtype=np.float32)
				preds = self.model.model(input_tensor, training=False)
				vals = preds[0].numpy()
				pol_logits = preds[1].numpy()
				
				eval_idx = 0
				for i in eval_indices:
					leaf = batch_leaves[i][0]
					v = vals[eval_idx][0]
					logits = pol_logits[eval_idx]
					
					# Process logits to probs
					mask = np.ones(logits.shape, dtype=bool)
					mask[leaf.state.allowedActions] = False
					logits[mask] = -100
					odds = np.exp(logits)
					probs = odds / np.sum(odds)
					
					# Expand Node (GUARDED: only if not already expanded by another sim in the batch)
					if not leaf.edges:
						for idx, action in enumerate(leaf.state.allowedActions):
							newState, _, _ = leaf.state.takeAction(action)
							if newState.id not in self.mcts.tree:
								node = mc.Node(newState)
								self.mcts.addNode(node)
							else:
								node = self.mcts.tree[newState.id]
							
							newEdge = mc.Edge(leaf, node, float(probs[idx]), action)
							leaf.edges.append((action, newEdge))
					
					# Replace dummy value with real NN value
					batch_leaves[i] = (leaf, v, 0)
					eval_idx += 1

			# 3. Batched Backfill
			for i in range(batch_size):
				leaf, v, done = batch_leaves[i]
				self.mcts.backFill(leaf, v, batch_breadcrumbs[i], remove_virtual_loss=True)

		print("!", end="", flush=True)
		pi, values = self.getAV(1)
		action, value = self.chooseAction(pi, values, tau)
		
		# Log terminal value prediction
		nextState, _, _ = state.takeAction(action)
		NN_value = self.get_preds(nextState)[0]
		return (action, pi, value, NN_value)

	def get_preds(self, state):
		inputToModel = np.array([self.model.convertToModelInput(state)], dtype=np.float32)
		preds = self.model.model(inputToModel, training=False)
		value = preds[0].numpy()[0][0]
		logits = preds[1].numpy()[0]
		mask = np.ones(logits.shape,dtype=bool)
		mask[state.allowedActions] = False
		logits[mask] = -100
		odds = np.exp(logits)
		probs = odds / np.sum(odds)
		return (value, probs, state.allowedActions)

	def chooseAction(self, pi, values, tau):
		if tau == 0:
			actions = np.argwhere(pi == np.max(pi))
			action = random.choice(actions)[0]
		else:
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx==1)[0][0]
		return action, values[action]

	def getAV(self, tau):
		edges = self.mcts.root.edges
		pi = np.zeros(self.action_size, dtype=float)
		values = np.zeros(self.action_size, dtype=np.float32)
		for action, edge in edges:
			pi[action] = pow(edge.stats['N'], 1/tau) if tau > 0 else edge.stats['N']
			values[action] = edge.stats['Q']
		pi = pi / (np.sum(pi) + 1e-9)
		return pi, values

	def replay(self, ltmemory):
		import yaml
		with open("config.yaml", 'r') as f:
			cfg = yaml.safe_load(f)
		for i in range(cfg['rl']['training_loops']):
			minibatch = random.sample(ltmemory, min(cfg['rl']['batch_size'], len(ltmemory)))
			training_states = np.array([row['state'].binary for row in minibatch])
			training_targets = {'value_head': np.array([row['value'] for row in minibatch]), 
								'policy_head': np.array([row['AV'] for row in minibatch])} 
			self.model.fit(training_states, training_targets, epochs=cfg['rl']['epochs'], verbose=0, validation_split=0, batch_size=cfg['rl']['batch_size'])

	def buildMCTS(self, state):
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct, self.cfg)

	def changeRootMCTS(self, state):
		self.mcts.root = self.mcts.tree[state.id]
