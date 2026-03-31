import numpy as np
import pandas as pd
import yaml

class TradingGame:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.name = 'trading'
        
        self.window_size = config['trading']['window_size']
        self.initial_balance = config['trading']['initial_balance']
        self.fee = config['trading']['fee']
        self.holding_penalty = config['trading']['holding_penalty']
        
        # Features: log_return, rsi, atr, vol_10, vol_30, rel_high_low, vol_delta (7)
        # Portfolio: position_type, position_age, realized_pnl, remaining_ratio, trade_count (5)
        self.feature_columns = ['log_return', 'rsi', 'atr', 'vol_10', 'vol_30', 'rel_high_low', 'vol_delta']
        self.n_features = len(self.feature_columns)
        self.n_portfolio = 5
        
        self.grid_shape = (self.window_size, self.n_features + self.n_portfolio)
        self.input_shape = (self.window_size, self.n_features + self.n_portfolio)
        self.action_size = 3 # 0: Flat, 1: Long, 2: Short
        self.state_size = self.window_size * (self.n_features + self.n_portfolio)
        
        self.reset()

    def reset(self):
        min_len, max_len = self.config['trading']['episode_length_range']
        self.episode_length = np.random.randint(min_len, max_len)
        
        max_start = len(self.data) - self.window_size - self.episode_length - 5
        self.start_tick = np.random.randint(self.window_size, max_start)
        self.current_tick = self.start_tick
        self.end_tick = self.start_tick + self.episode_length
        
        self.portfolio = {
            'balance': self.initial_balance,
            'position': 0, # 0: Flat, 1: Long, 2: Short
            'entry_price': 0,
            'trade_count': 0,
            'realized_pnl': 0.0,
            'step_count': 0
        }
        
        self.gameState = TradingGameState(self)
        return self.gameState

    def step(self, action):
        next_state, reward, done = self.gameState.takeAction(action)
        self.gameState = next_state
        return (next_state, reward, done, None)

    def identities(self, state, actionValues):
        return [(state, actionValues)]

class TradingGameState:
    def __init__(self, game):
        self.game = game
        self.data = game.data
        self.current_tick = game.current_tick
        self.portfolio = game.portfolio
        self.playerTurn = 1
        self.pieces = {'1':'LONG', '0': 'FLAT', '2': 'SHORT'}
        
        self.allowedActions = [0, 1, 2]
        self.id = f"{self.current_tick}_{self.portfolio['position']}_{self.portfolio['balance']:.2f}"
        self.binary = self._generate_state_tensor()
        self.board = self.binary # For memory compatibility
        
        self.isEndGame = (self.current_tick >= game.end_tick) or (self.portfolio['balance'] <= 0)
        self.value = self._get_value()
        self.score = (0, 0)

    def _generate_state_tensor(self):
        # 1. Extract Market Features
        start = self.current_tick - self.game.window_size
        end = self.current_tick
        window = self.data.iloc[max(0, start) : end]
        market_features = window[self.game.feature_columns].values
        
        # Robust padding if window is too small (e.g. at the very start of the data)
        if market_features.shape[0] < self.game.window_size:
            pad_width = self.game.window_size - market_features.shape[0]
            if market_features.shape[0] > 0:
                market_features = np.pad(market_features, ((pad_width, 0), (0, 0)), mode='edge')
            else:
                market_features = np.zeros((self.game.window_size, self.game.n_features))
        
        # Simple Z-Score normalization per window
        market_features = (market_features - np.mean(market_features, axis=0)) / (np.std(market_features, axis=0) + 1e-9)
        
        # 2. Extract Portfolio Features
        p = self.portfolio
        remaining_ratio = (self.game.end_tick - self.current_tick) / self.game.episode_length
        
        portfolio_row = np.array([
            p['position'],
            p['step_count'] / self.game.episode_length, # position_age approx
            p['realized_pnl'],
            remaining_ratio,
            p['trade_count'] / 50.0 # Normalized trade count
        ])
        
        # Broadcast portfolio info across the temporal window
        portfolio_features = np.tile(portfolio_row, (self.game.window_size, 1))
        
        # Concatenate to (window_size, n_features + n_portfolio)
        return np.concatenate([market_features, portfolio_features], axis=1)

    def _get_value(self):
        if self.isEndGame:
            pnl = (self.portfolio['balance'] / self.game.initial_balance) - 1.0
            return (np.tanh(pnl), 0, 0)
        return (0, 0, 0)

    def takeAction(self, action):
        new_p = self.portfolio.copy()
        current_price = self.data.iloc[self.current_tick]['close']
        reward = 0.0
        
        # 1. Execute Trade & Costs
        if action != self.portfolio['position']:
            # Close cost
            if self.portfolio['position'] != 0:
                new_p['balance'] -= new_p['balance'] * self.game.fee
            
            # Open cost
            if action != 0:
                new_p['balance'] -= new_p['balance'] * self.game.fee
                new_p['entry_price'] = current_price
                new_p['step_count'] = 0
                new_p['trade_count'] += 1
            
            new_p['position'] = action
        else:
            new_p['step_count'] += 1

        # 2. Calculate PnL Delta (Reward)
        if self.portfolio['position'] != 0:
            prev_price = self.data.iloc[self.current_tick - 1]['close']
            price_change = (current_price / prev_price) - 1.0
            pnl_delta = price_change * self.portfolio['position']
            
            realized_delta = new_p['balance'] * pnl_delta
            new_p['balance'] += realized_delta
            new_p['realized_pnl'] += pnl_delta
            
            reward = pnl_delta # Basic reward signal
        
        # 3. Holding Penalty
        if action == 0:
            reward -= self.game.holding_penalty
            
        # Move to next tick
        self.game.current_tick += 1
        self.game.portfolio = new_p
        
        new_state = TradingGameState(self.game)
        done = 1 if new_state.isEndGame else 0
        
        return (new_state, reward, done)

    def render(self, logger):
        logger.info(f"T:{self.current_tick} | P:{self.portfolio['position']} | B:{self.portfolio['balance']:.2f} | PnL:{self.portfolio['realized_pnl']:.4f}")
