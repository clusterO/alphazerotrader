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
		# PHASE 6: FAIR MODE (Non-Oracle Inference)
		# If MCTS sims are 0, we rely solely on the Neural Network's Policy and Value heads.
		# This prevents the agent from "searching" future price data during backtests.
		if self.MCTSsimulations == 0:
			v, probs, allowed = self.get_preds(state)
			action, value = self.chooseAction(probs, [v]*self.action_size, tau)
			# NN_value is the same as value for raw inference
			return (action, probs, v, v)

		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)

		# PHASE 4.2: BATCHED MCTS
		batch_size = self.cfg['rl'].get('mcts_batch_size', 8)
		num_batches = self.MCTSsimulations // batch_size
		
		# Ensure at least one batch if sims > 0 but less than batch_size
		if num_batches == 0 and self.MCTSsimulations > 0:
			num_batches = 1

		for b in range(num_batches):
			if self.MCTSsimulations > 0 and b % 4 == 0: print(".", end="", flush=True)
			
			batch_leaves = []
			batch_breadcrumbs = []
			
			# 1. Selection with Virtual Loss
			batch_depths = []
			for _ in range(batch_size):
				leaf, value, done, breadcrumbs = self.mcts.moveToLeaf(apply_virtual_loss=True)
				batch_leaves.append((leaf, value, done))
				batch_breadcrumbs.append(breadcrumbs)
				batch_depths.append(len(breadcrumbs))
			
			# Diagnostic: Track depths if near horizon
			rem = state.end_tick - state.current_tick
			if rem <= 20 and batch_depths:
				avg_d = sum(batch_depths) / len(batch_depths)
				lg.logger_mcts.debug(f"MCTS Selection Depth [Rem: {rem}]: Avg={avg_d:.1f}, Max={max(batch_depths)}")
			
			# 2. Batched Evaluation
			states_to_eval = []
			eval_indices = []
			
			for i, (leaf, value, done) in enumerate(batch_leaves):
				if not leaf.state.isEndGame:
					states_to_eval.append(self.model.convertToModelInput(leaf.state))
					eval_indices.append(i)
				else:
					# Terminal node: use actual value
					v = leaf.state.value[0]
					batch_leaves[i] = (leaf, v, 1)
			
			if states_to_eval:
				input_tensor = np.array(states_to_eval, dtype=np.float32)
				# PHASE 4.3: USE COMPILED GRAPH INFERENCE
				preds = self.model.predict_batch(input_tensor)
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

		if self.MCTSsimulations > 0: print("!", end="", flush=True)
		pi, values = self.getAV(1)
		action, value = self.chooseAction(pi, values, tau)
		
		# Log terminal value prediction (Safety Check for end of episode)
		if not state.isEndGame:
			nextState, _, _ = state.takeAction(action)
			NN_value = self.get_preds(nextState)[0]
		else:
			NN_value = state.value[0]
			
		# --- HORIZON COLLAPSE DIAGNOSTICS (Phase 6) ---
		remaining_ticks = state.end_tick - state.current_tick
		if remaining_ticks <= 20:
			# Get raw NN value for current state
			raw_nn_v = self.get_preds(state)[0]
			lg.logger_mcts.info(
				f"HORIZON DIAG [Tick {state.current_tick}/{state.end_tick}] "
				f"Remaining: {remaining_ticks} | "
				f"MCTS V: {value:.4f} | "
				f"NN V: {raw_nn_v:.4f} | "
				f"Diff: {abs(value - raw_nn_v):.4f}"
			)
			
		return (action, pi, value, NN_value)

	def get_preds(self, state):
		inputToModel = np.array([self.model.convertToModelInput(state)], dtype=np.float32)
		preds = self.model.predict_batch(inputToModel)
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
		
		batch_size = cfg['rl']['batch_size']
		for i in range(cfg['rl']['training_loops']):
			minibatch = random.sample(ltmemory, min(batch_size, len(ltmemory)))
			
			training_states = np.array([row['state'].binary for row in minibatch])
			
			# PHASE 8: VALUE-WEIGHTED POLICY TARGETS
			# Instead of just matching MCTS visits, we scale by the outcome (z)
			# target = pi * (1.0 + z) -> Moves in winning episodes get boosted
			raw_pi = np.array([row['AV'] for row in minibatch])
			z_values = np.array([row['value'] for row in minibatch])
			
			# Normalize z to [0, 2] range for weighting (z is -1 to 1)
			weights = 1.0 + z_values
			weighted_pi = raw_pi * weights[:, np.newaxis]
			# Re-normalize to ensure they are valid probabilities for cross-entropy
			weighted_pi = weighted_pi / (np.sum(weighted_pi, axis=1, keepdims=True) + 1e-9)

			# PHASE 8: ANTI-SATURATION (ENTROPY FLOOR)
			# Check for "Long" saturation (>75% bias)
			actions = np.argmax(weighted_pi, axis=1)
			long_pct = np.sum(actions == 1) / len(actions)
			
			if long_pct > 0.75:
				# Inject Dirichlet noise to force uncertainty
				epsilon = 0.25
				noise = np.random.dirichlet([0.3] * self.action_size, size=len(minibatch))
				weighted_pi = (1 - epsilon) * weighted_pi + epsilon * noise

			training_targets = {
				'value_head': z_values, 
				'policy_head': weighted_pi
			} 
			
			self.model.fit(training_states, training_targets, epochs=cfg['rl']['epochs'], verbose=0, validation_split=0, batch_size=batch_size)

	def buildMCTS(self, state):
		if self.mcts is not None:
			self.mcts.clear()
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct, self.cfg)

	def changeRootMCTS(self, state):
		new_root = self.mcts.tree[state.id]
		self.mcts.prune(new_root)
