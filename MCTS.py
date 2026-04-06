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
				

class MCTS():

	def __init__(self, root, cpuct, cfg):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.cfg = cfg
		self.addNode(root)
		self.root_nu = None # Lazy initialization

	def moveToLeaf(self):
		breadcrumbs = []
		currentNode = self.root

		done = 0
		value = 0

		while not currentNode.isLeaf():
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
				Nb = Nb + edge.stats['N']

			simulationAction = None
			simulationEdge = None

			for idx, (action, edge) in enumerate(currentNode.edges):

				U = self.cpuct * \
					((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
					np.sqrt(Nb) / (1 + edge.stats['N'])
					
				Q = edge.stats['Q']
				if isinstance(Q, np.ndarray):
					Q = Q.item()

				lg.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
					, int(action), int(action % 7), int(edge.stats['N']), float(edge.stats['P']), float(nu[idx]), float(((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] ))
					, float(edge.stats['W'].item() if isinstance(edge.stats['W'], np.ndarray) else edge.stats['W'])
					, float(Q), float(U.item() if isinstance(U, np.ndarray) else U), float((Q+U).item() if isinstance(Q+U, np.ndarray) else Q+U))

				QU = Q + U
				if np.isnan(QU):
					QU = -np.inf   # treat NaN as very bad

				if QU > maxQU:
					maxQU = QU
					simulationAction = action
					simulationEdge = edge

			if simulationAction is None:
				lg.logger_mcts.warning('All actions had NaN QU! Picking first edge by default.')
				simulationAction, simulationEdge = currentNode.edges[0]

			lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)

			newState, value, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
			currentNode = simulationEdge.outNode
			breadcrumbs.append(simulationEdge)

		lg.logger_mcts.info('DONE...%d', done)

		return currentNode, value, done, breadcrumbs



	def backFill(self, leaf, value, breadcrumbs):
		lg.logger_mcts.info('------DOING BACKFILL------')

		currentPlayer = leaf.state.playerTurn

		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn == currentPlayer:
				direction = 1
			else:
				direction = -1
			
			# If trading, we don't have an opponent, so we don't flip the value
			# In TradingGame, playerTurn is always 1 for now.
			if hasattr(config, 'SYMBOL'):
				direction = 1

			edge.stats['N'] = edge.stats['N'] + 1
			edge.stats['W'] = edge.stats['W'] + value * direction
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

			lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
				, float(value * direction)
				, int(playerTurn)
				, int(edge.stats['N'])
				, float(edge.stats['W'])
				, float(edge.stats['Q'])
				)

			edge.outNode.state.render(lg.logger_mcts)

	def addNode(self, node):
		self.tree[node.id] = node

