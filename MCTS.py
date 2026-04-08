import numpy as np
import logging
import config

from utils import setup_logger
import loggers as lg

class Node():

	def __init__(self, state):
		self.state = state
		self.playerTurn = state.playerTurn
		self.id = state.id
		self.edges = []

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():

	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
		self.inNode = inNode
		self.outNode = outNode
		self.playerTurn = inNode.state.playerTurn
		self.action = action

		self.stats =  {
					'N': 0,
					'W': 0,
					'Q': 0,
					'P': prior,
				}
		self.virtual_loss = 0 # Phase 4.2: For batched MCTS

class MCTS():

	def __init__(self, root, cpuct, cfg):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.cfg = cfg
		self.addNode(root)
		self.root_nu = None # Lazy initialization

	def moveToLeaf(self, apply_virtual_loss=False):
		breadcrumbs = []
		currentNode = self.root

		done = 0
		value = 0
		depth = 0
		max_depth = self.cfg['rl'].get('mcts_max_depth', 1000)

		while not currentNode.isLeaf():
			depth += 1
			if depth > max_depth:
				lg.logger_mcts.warning(f"CIRCUIT BREAKER: MCTS depth {depth} exceeded max {max_depth}. Forcing leaf at {currentNode.id}")
				break

			maxQU = -99999

			if currentNode == self.root:
				epsilon = self.cfg['rl']['epsilon']
				# Calculate noise once when root is first expanded
				if self.root_nu is None or len(self.root_nu) != len(self.root.edges):
					self.root_nu = np.random.dirichlet([self.cfg['rl']['alpha']] * len(self.root.edges))
				nu = self.root_nu
			else:
				epsilon = 0
				nu = [0] * len(currentNode.edges)

			Nb = 0
			for action, edge in currentNode.edges:
				Nb = Nb + edge.stats['N'] + edge.virtual_loss

			simulationAction = None
			simulationEdge = None

			for idx, (action, edge) in enumerate(currentNode.edges):
				# Q is adjusted by virtual loss
				N = edge.stats['N'] + edge.virtual_loss
				W = edge.stats['W'] - edge.virtual_loss # Treat virtual loss as -1 value
				Q = W / (N + 1e-9)

				U = self.cpuct * \
					((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
					np.sqrt(Nb) / (1 + N)
					
				QU = Q + U
				if np.isnan(QU):
					QU = -np.inf

				if QU > maxQU:
					maxQU = QU
					simulationAction = action
					simulationEdge = edge

			if simulationAction is None:
				simulationAction, simulationEdge = currentNode.edges[0]

			if apply_virtual_loss:
				simulationEdge.virtual_loss += 1

			currentNode = simulationEdge.outNode
			breadcrumbs.append(simulationEdge)

		return currentNode, value, done, breadcrumbs

	def backFill(self, leaf, value, breadcrumbs, remove_virtual_loss=False):
		# If trading, direction is always 1
		direction = 1

		for edge in breadcrumbs:
			if remove_virtual_loss:
				edge.virtual_loss -= 1

			edge.stats['N'] = edge.stats['N'] + 1
			edge.stats['W'] = edge.stats['W'] + value * direction
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

	def clear(self):
		# Manually break circular references to help GC
		for node_id in list(self.tree.keys()):
			node = self.tree[node_id]
			for action, edge in node.edges:
				edge.inNode = None
				edge.outNode = None
			node.edges = []
		self.tree.clear()
		self.root = None

	def addNode(self, node):
		self.tree[node.id] = node

