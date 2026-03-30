import numpy as np
import pandas as pd

class TradingGame:
    def __init__(self, data, window_size=50, initial_balance=1000):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.name = 'trading'
        
        # State size: window_size * 5 (OHLCV) + 3 (portfolio: position, pnl, balance)
        # We'll reshape this for the CNN
        self.grid_shape = (window_size, 5)
        self.input_shape = (window_size, 5, 2) # channels_last: (height, width, channels)
        self.state_size = window_size * 5 * 2
        self.action_size = 3 # 0: Flat, 1: Long, 2: Short
        
        self.reset()

    def reset(self):
        # Start at a random point in the data that allows for a full window and some future steps
        max_start = len(self.data) - self.window_size - 200 # Episode length 200
        self.start_tick = np.random.randint(self.window_size, max_start)
        self.current_tick = self.start_tick
        
        self.portfolio = {
            'balance': self.initial_balance,
            'position': 0, # 0: Flat, 1: Long, 2: Short
            'entry_price': 0,
            'units': 0
        }
        
        self.gameState = TradingGameState(
            self.data, 
            self.current_tick, 
            self.window_size, 
            self.portfolio,
            1 # AlphaZero single player turn
        )
        return self.gameState

    def step(self, action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        return (next_state, value, done, None)

    def identities(self, state, actionValues):
        return [(state, actionValues)]

class TradingGameState:
    def __init__(self, data, current_tick, window_size, portfolio, playerTurn):
        self.data = data
        self.current_tick = current_tick
        self.window_size = window_size
        self.portfolio = portfolio
        self.playerTurn = playerTurn
        self.pieces = {'1':'LONG', '0': 'FLAT', '2': 'SHORT'}
        
        self.allowedActions = [0, 1, 2] # Flat, Long, Short
        self.id = self._convertStateToId()
        self.binary = self._binary()
        self.board = self.data.iloc[self.current_tick - self.window_size : self.current_tick][['open', 'high', 'low', 'close', 'volume']].values
        
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = (0, 0)

    def _allowedActions(self):
        return [0, 1, 2]

    def _binary(self):
        # Extract window
        window = self.data.iloc[self.current_tick - self.window_size : self.current_tick]
        ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values
        
        # Normalize
        base_price = ohlcv[0, 3] # first close
        norm_ohlcv = ohlcv.copy()
        norm_ohlcv[:, :4] = norm_ohlcv[:, :4] / base_price - 1.0
        norm_ohlcv[:, 4] = norm_ohlcv[:, 4] / (norm_ohlcv[:, 4].mean() + 1e-9)
        
        # Channel 1: Market Data
        channel1 = norm_ohlcv
        
        # Channel 2: Portfolio Data
        channel2 = np.zeros_like(channel1)
        channel2[:, 0] = self.portfolio['position']
        if self.portfolio['position'] != 0:
            current_price = self.data.iloc[self.current_tick]['close']
            pnl = (current_price / self.portfolio['entry_price'] - 1.0) * self.portfolio['position']
            channel2[:, 1] = pnl
            
        # Reshape to (window_size, 5, 2)
        state = np.stack([channel1, channel2], axis=-1)
        return state

    def _convertStateToId(self):
        # A unique ID for the state. Since data is fixed, tick + position is enough.
        return f"{self.current_tick}_{self.portfolio['position']}"

    def _checkForEndGame(self):
        # End after 200 ticks or if bankrupt
        if self.portfolio['balance'] <= 0:
            return True
        # For now, we'll let the Game class handle episode length
        return False

    def _getValue(self):
        # AlphaZero value is between -1 and 1.
        # We can use normalized profit.
        if self.portfolio['balance'] <= 0:
            return (-1, -1, 1)
        return (0, 0, 0)

    def takeAction(self, action):
        new_portfolio = self.portfolio.copy()
        current_price = self.data.iloc[self.current_tick]['close']
        
        # Execute trade
        if action != self.portfolio['position']:
            # Close old position
            if self.portfolio['position'] != 0:
                pnl = (current_price / self.portfolio['entry_price'] - 1.0) * self.portfolio['position']
                new_portfolio['balance'] *= (1.0 + pnl)
            
            # Open new position
            new_portfolio['position'] = action
            new_portfolio['entry_price'] = current_price
            
        # Move to next tick
        new_tick = self.current_tick + 1
        done = 0
        if new_tick >= len(self.data) - 1:
            done = 1
        
        # We could also define a fixed episode length here
        
        newState = TradingGameState(self.data, new_tick, self.window_size, new_portfolio, self.playerTurn)
        
        # Reward/Value for the step
        value = 0
        if done:
            # Final profit relative to initial balance
            final_pnl = (new_portfolio['balance'] / 1000.0) - 1.0
            value = np.tanh(final_pnl) # Squeeze to [-1, 1]
            
        return (newState, value, done)

    def render(self, logger):
        logger.info(f"Tick: {self.current_tick}, Pos: {self.portfolio['position']}, Bal: {self.portfolio['balance']:.2f}")
