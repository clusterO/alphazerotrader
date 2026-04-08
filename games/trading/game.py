import numpy as np
import pandas as pd
import copy

class TradingGame:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.name = 'trading'
        
        self.window_size = config['trading']['window_size']
        self.initial_balance = config['trading']['initial_balance']
        self.fee = config['trading']['fee']
        
        self.feature_columns = ['log_return', 'rsi', 'atr', 'vol_10', 'vol_30', 'rel_high_low', 'vol_delta']
        self.n_features = len(self.feature_columns)
        self.n_portfolio = 5
        
        self.grid_shape = (self.window_size, self.n_features + self.n_portfolio)
        self.input_shape = (self.window_size, self.n_features + self.n_portfolio)
        self.action_size = 3
        self.state_size = self.window_size * (self.n_features + self.n_portfolio)
        
        # Static config dict to pass to states (prevents mutating the game runner)
        self.game_config = {
            'window_size': self.window_size,
            'initial_balance': self.initial_balance,
            'fee': self.fee,
            'tanh_scale': config['trading'].get('tanh_scale', 10.0),
            'feature_columns': self.feature_columns,
            'n_features': self.n_features,
            'n_portfolio': self.n_portfolio
        }
        
        self.reset()

    def reset(self):
        min_len, max_len = self.config['trading']['episode_length_range']
        available_data = len(self.data) - self.window_size - 10
        
        # Safe bounds for episode length
        self.episode_length = np.random.randint(min_len, max(min_len + 1, min(max_len, available_data + 1)))
        
        # Safe bounds for start tick
        min_start = self.window_size
        max_start = max(min_start + 1, len(self.data) - self.episode_length - 5)
        self.start_tick = np.random.randint(min_start, max_start)
        
        portfolio = {
            'balance': float(self.initial_balance),
            'position': 0,
            'entry_price': 0.0,
            'trade_count': 0,
            'realized_pnl': 0.0,
            'step_count': 0
        }
        
        self.gameState = TradingGameState(
            self.data, self.start_tick, self.start_tick + self.episode_length,
            portfolio, self.game_config
        )
        return self.gameState

    def step(self, action):
        next_state, reward, done = self.gameState.takeAction(action)
        self.gameState = next_state
        return (next_state, reward, done, None)

    def identities(self, state, actionValues):
        return [(state, actionValues)]

class TradingGameState:
    def __init__(self, data, current_tick, end_tick, portfolio, game_config):
        # All inputs are set once. No mutation allowed after __init__.
        self.data = data
        self.current_tick = current_tick
        self.end_tick = end_tick
        self.portfolio = portfolio # Expects a dict
        self.game_config = game_config 
        
        self.playerTurn = 1
        self.pieces = {'1':'LONG', '0': 'FLAT', '2': 'SHORT'}
        self.allowedActions = [0, 1, 2]
        
        # Pre-generate state data
        self.binary = self._generate_state_tensor()
        self.board = self.binary
        self.id = f"{self.current_tick}_{self.portfolio['position']}_{self.portfolio['balance']:.4f}"
        
        self.isEndGame = (self.current_tick >= self.end_tick) or (self.portfolio['balance'] <= (self.game_config['initial_balance'] * 0.1))
        self.value = self._get_value()
        self.score = (0, 0)

    def _generate_state_tensor(self):
        start = self.current_tick - self.game_config['window_size']
        end = self.current_tick
        window = self.data.iloc[max(0, start) : end]
        market_features = window[self.game_config['feature_columns']].values
        
        if market_features.shape[0] < self.game_config['window_size']:
            pad_width = self.game_config['window_size'] - market_features.shape[0]
            if market_features.shape[0] > 0:
                market_features = np.pad(market_features, ((pad_width, 0), (0, 0)), mode='edge')
            else:
                market_features = np.zeros((self.game_config['window_size'], self.game_config['n_features']))
        
        # Local window normalization
        market_features = (market_features - np.mean(market_features, axis=0)) / (np.std(market_features, axis=0) + 1e-9)
        
        p = self.portfolio
        ep_len = self.end_tick - (self.current_tick - p['step_count'])
        remaining_ratio = (self.end_tick - self.current_tick) / max(1, ep_len)
        
        portfolio_row = np.array([
            float(p['position']),
            float(p['step_count'] / 100.0),
            float(p['realized_pnl']),
            float(remaining_ratio),
            float(p['trade_count'] / 50.0)
        ])
        
        portfolio_features = np.tile(portfolio_row, (self.game_config['window_size'], 1))
        return np.concatenate([market_features, portfolio_features], axis=1).astype(np.float32)

    def _get_value(self):
        if self.isEndGame:
            pnl_pct = (self.portfolio['balance'] / self.game_config['initial_balance']) - 1.0
            # Apply TANH_SCALE from config
            z = np.tanh(self.game_config['tanh_scale'] * pnl_pct)
            return (z, 0, 0)
        return (0, 0, 0)

    def takeAction(self, action):
        # 1. IMMUTABILITY ASSERTION
        state_snapshot = (self.current_tick, self.portfolio['position'], self.portfolio['balance'])
        
        # 2. Logic
        new_p = self.portfolio.copy()
        price_at_t = self.data.iloc[self.current_tick]['close']
        balance_before_step = new_p['balance']
        
        # Execute Trade & Fees (applied to balance at T)
        if action != self.portfolio['position']:
            # Close cost
            if self.portfolio['position'] != 0:
                new_p['balance'] -= new_p['balance'] * self.game_config['fee']
            
            # Open cost
            if action != 0:
                new_p['balance'] -= new_p['balance'] * self.game_config['fee']
                new_p['entry_price'] = price_at_t
                new_p['trade_count'] += 1
            
            new_p['position'] = action
            new_p['step_count'] = 0
        else:
            new_p['step_count'] += 1

        # Advance Time strictly to T+1 (Ensures unique ID)
        next_tick = self.current_tick + 1
        
        # Safety: Check if we just stepped past the absolute end of data
        if next_tick >= len(self.data):
            # Create terminal state at the edge
            new_state = TradingGameState(
                self.data, self.current_tick, self.end_tick,
                new_p, self.game_config
            )
            new_state.isEndGame = True # Force termination
            return (new_state, 0.0, 1)

        price_at_t_plus_1 = self.data.iloc[next_tick]['close']
        
        # Calculate PnL for the move [T -> T+1] using the position held during that interval
        if new_p['position'] != 0:
            price_change_pct = (price_at_t_plus_1 / price_at_t) - 1.0
            pnl_delta_pct = price_change_pct * new_p['position']
            
            new_p['balance'] += new_p['balance'] * pnl_delta_pct
            new_p['realized_pnl'] += pnl_delta_pct
            
        # Step Reward: delta in total portfolio value normalized by starting balance
        reward = (new_p['balance'] - balance_before_step) / balance_before_step

        # 3. Create NEW independent state at T+1
        new_state = TradingGameState(
            self.data, next_tick, self.end_tick,
            new_p, self.game_config
        )
        
        # Verify Immutability
        assert state_snapshot == (self.current_tick, self.portfolio['position'], self.portfolio['balance']), "CRITICAL: State mutation detected in takeAction!"
        
        done = 1 if new_state.isEndGame else 0
        return (new_state, reward, done)

    def render(self, logger):
        logger.info(f"T:{self.current_tick} | P:{self.portfolio['position']} | B:{self.portfolio['balance']:.2f} | PnL:{self.portfolio['realized_pnl']:.4f}")
