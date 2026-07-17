import numpy as np
import pandas as pd
import pandas_ta as ta
import os
import yaml

class TradingGame:
    def __init__(self, data, config):
        # data can be a single DataFrame or a list of DataFrames for Multi-Regime training
        self.data_list = data if isinstance(data, list) else [data]
        self.config = config
        self.name = 'trading'
        
        self.window_size = config['trading']['window_size']
        self.initial_balance = config['trading']['initial_balance']
        self.fee = config['trading']['fee']
        
        self.feature_columns = [
            'log_return', 'rsi', 'atr', 'vol_10', 'vol_30', 'rel_high_low', 'vol_delta',
            'trend_20', 'trend_50', 'dist_ma_200', 'vol_skew', 'momentum_5', 'momentum_10', 'vol_regime'
        ]
        self.n_features = len(self.feature_columns)
        self.n_portfolio = 5 # pos, step_count, pnl, remaining_ratio, current_dd
        
        self.input_shape = (self.window_size, self.n_features + self.n_portfolio)
        self.action_size = 3 # 0: Flat, 1: Long, 2: Short
        self.state_size = self.window_size * (self.n_features + self.n_portfolio)
        self.grid_shape = (1, self.action_size) # For compatibility with legacy loggers
        
        self.regimes = []
        self.preprocess_data()
        
        self.game_config = {
            'window_size': self.window_size,
            'initial_balance': self.initial_balance,
            'fee': self.fee,
            'tanh_scale': config['trading'].get('tanh_scale', 3.0),
            'dd_threshold': config['trading'].get('drawdown_threshold', 0.12),
            'idling_penalty': config['trading'].get('idling_penalty', 0.0001)
        }

    def preprocess_data(self):
        for df_raw in self.data_list:
            df = df_raw.copy()
            
            # Log returns
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Technical Indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14) / (df['close'] + 1e-9)
            
            # Volatility
            df['vol_10'] = df['log_return'].rolling(window=10).std()
            df['vol_30'] = df['log_return'].rolling(window=30).std()
            
            # Relative High/Low (30 period)
            df['high_30'] = df['high'].rolling(window=30).max()
            df['low_30'] = df['low'].rolling(window=30).min()
            df['rel_high_low'] = (df['close'] - df['low_30']) / (df['high_30'] - df['low_30'] + 1e-9)
            
            # Volume Delta
            df['vol_delta'] = df['volume'].diff() / (df['volume'].shift(1) + 1e-9)
            
            # --- REGIME AWARE FEATURES ---
            close = df['close']
            returns = df['log_return']
            realized_vol = returns.rolling(20).std()

            # 1. Trend direction
            df['trend_20'] = close.pct_change(20) / (realized_vol + 1e-8)
            df['trend_50'] = close.pct_change(50) / (realized_vol + 1e-8)

            # 2. Distance from long-term MA
            df['dist_ma_200'] = (close - close.rolling(200).mean()) / (close.rolling(20).std() + 1e-8)

            # 3. Volatility skew
            neg_returns = returns.clip(upper=0)
            pos_returns = returns.clip(lower=0)
            df['vol_skew'] = (neg_returns.rolling(20).std() / (pos_returns.rolling(20).std() + 1e-8))

            # 4. Momentum
            df['momentum_5'] = close.pct_change(5) / (realized_vol + 1e-8)
            df['momentum_10'] = close.pct_change(10) / (realized_vol + 1e-8)

            # 5. Volatility regime
            vol_ma = realized_vol.rolling(50).mean()
            df['vol_regime'] = (realized_vol - vol_ma) / (vol_ma + 1e-8)

            df.dropna(inplace=True)
            
            market_data_array = df[self.feature_columns].values.astype(np.float32)
            close_prices = df['close'].values.astype(np.float32)
            self.regimes.append((market_data_array, close_prices))
        
        # For legacy compatibility or backtests with single data
        self.market_data_array = self.regimes[0][0]
        self.close_prices = self.regimes[0][1]

    def reset(self, start_tick=None, end_tick=None):
        # Pick a random regime for this episode
        regime_idx = np.random.randint(0, len(self.regimes))
        market_data_array, close_prices = self.regimes[regime_idx]
        
        # Also update pointers for current episode
        self.market_data_array = market_data_array
        self.close_prices = close_prices

        # EPISODE RANDOMIZATION
        min_len, max_len = self.config['trading']['episode_length_range']
        
        # PHASE 6: LOOKAHEAD BUFFER for MCTS Horizon
        LOOKAHEAD_BUFFER = 50
        
        # Total usable ticks after dropping NaNs and allowing for window_size
        total_usable = len(market_data_array) 
        
        # Max steps we can actually take, leaving room for buffer
        max_possible_ep_len = total_usable - self.window_size - 1 - LOOKAHEAD_BUFFER
        
        if start_tick is not None:
            self.start_tick = start_tick
            self.episode_length = (end_tick - start_tick) if end_tick else min(max_len, max_possible_ep_len)
        else:
            if max_possible_ep_len <= 5:
                 # Critical fallback for tiny datasets
                 self.episode_length = max(1, max_possible_ep_len)
                 self.start_tick = self.window_size
            else:
                 self.episode_length = np.random.randint(min(min_len, max_possible_ep_len), min(max_len, max_possible_ep_len) + 1)
                 max_start = total_usable - self.episode_length - 1 - LOOKAHEAD_BUFFER
                 self.start_tick = np.random.randint(self.window_size, max(self.window_size + 1, max_start))
        
        self.episode_end_tick = self.start_tick + self.episode_length
        actual_end_tick = min(self.episode_end_tick + LOOKAHEAD_BUFFER, total_usable - 1)
        
        portfolio = {
            'balance': float(self.initial_balance),
            'peak_balance': float(self.initial_balance),
            'max_drawdown': 0.0,
            'position': 0,
            'entry_price': 0.0,
            'trade_count': 0,
            'realized_pnl': 0.0,
            'step_count': 0,
            'sum_returns': 0.0,
            'sum_sq_returns': 0.0,
            'returns_count': 0
        }
        
        self.gameState = GameState(
            market_data_array, close_prices, self.start_tick, actual_end_tick,
            portfolio, self.game_config, reward_tick=self.episode_end_tick
        )
        return self.gameState

    def step(self, action):
        next_state, reward, done = self.gameState.takeAction(action)
        # FORCE TERMINATION at official episode length (The "Lookahead Buffer" is MCTS-only)
        if next_state.current_tick >= self.episode_end_tick:
            done = True
        self.gameState = next_state
        return (next_state, reward, done, None)

    def identities(self, state, actionValues):
        return [(state, actionValues)]

class GameState:
    def __init__(self, market_data, close_prices, current_tick, end_tick, portfolio, game_config, reward_tick=None, custom_binary=None, wm_depth=0):
        self.market_data = market_data
        self.close_prices = close_prices
        
        self.current_tick = current_tick
        self.end_tick = end_tick
        self.reward_tick = reward_tick or end_tick
        self.portfolio = portfolio
        self.game_config = game_config 
        self.wm_depth = wm_depth
        
        self.playerTurn = 1
        self.pieces = {'1':'LONG', '0': 'FLAT', '2': 'SHORT'}
        self.allowedActions = [0, 1, 2]
        
        if custom_binary is not None:
            self.binary = custom_binary
        else:
            self.binary = self._generate_state_tensor()
            
        self.board = self.binary
        self.id = f"{self.current_tick}_{self.portfolio['position']}_{self.portfolio['balance']:.4f}"
        
        # Terminate if DD exceeds threshold or reached end tick
        self.isEndGame = (self.current_tick >= self.end_tick) or (self.portfolio['max_drawdown'] > self.game_config['dd_threshold'])
        
        self.value = self._get_value()
        self.score = (0, 0)

    def render(self, logger):
        p = self.portfolio
        logger.info(f"Tick: {self.current_tick} | Bal: {p['balance']:.2f} | Pos: {p['position']} | PnL: {p['realized_pnl']:.4f} | DD: {p['max_drawdown']:.4%}")

    def _generate_state_tensor(self):
        start = self.current_tick - self.game_config['window_size'] + 1
        end = self.current_tick + 1
        market_features = self.market_data[start:end].copy()
        
        # Sign-preserving normalization: Scale by mean absolute value per feature in window
        # Store scale for potential denormalization in fantasy rollouts
        self.norm_scale = np.mean(np.abs(market_features), axis=0) + 1e-9
        market_features = market_features / self.norm_scale
        
        p = self.portfolio
        ep_len = self.reward_tick - (self.current_tick - p['step_count'])
        remaining_ratio = (self.reward_tick - self.current_tick) / max(1, ep_len)
        current_dd = (p['peak_balance'] - p['balance']) / (p['peak_balance'] + 1e-9)
        
        portfolio_row = np.array([
            float(p['position']),
            float(p['step_count'] / 100.0),
            float(p['realized_pnl']),
            float(remaining_ratio),
            float(current_dd) 
        ], dtype=np.float32)
        
        portfolio_features = np.tile(portfolio_row, (self.game_config['window_size'], 1))
        return np.concatenate([market_features, portfolio_features], axis=1).astype(np.float32)

    def takeActionWM(self, action, world_model):
        """
        Modified takeAction for World Model Rollouts.
        Uses Transition Model instead of historical data.
        Enforces a 5-step depth cap.
        """
        if self.wm_depth >= 5:
            # Expansion refused/Terminal for this branch
            return self, 0.0, True

        # 1. Predict next state tensor
        next_binary = world_model.predict(self.binary, action)
        
        return self.takeActionWM_FromPrediction(action, next_binary)

    def takeActionWM_FromPrediction(self, action, next_binary):
        """
        Analytics part of takeActionWM, allowing for batched predictions.
        """
        # 2. Analytic Reward Calculation
        # log_return is at index 0. We need to denormalize it using the last known scale.
        scale = getattr(self, 'norm_scale', np.ones(19))
        predicted_log_return = next_binary[-1, 0] * scale[0]
        
        # ret = exp(log_ret) - 1
        step_return = np.exp(predicted_log_return) - 1.0
        
        # Adjust based on action (Mirroring takeAction logic)
        if action == 2: # SHORT
            step_return = 1.0 - np.exp(predicted_log_return)
        elif action == 0: # FLAT
            step_return = 0.0
            
        # Subtract fees if position changed
        if action != self.portfolio['position']:
            step_return -= self.game_config['fee']
            
        # Update portfolio (Fantasy)
        new_portfolio = self.portfolio.copy()
        new_portfolio['step_count'] += 1
        new_portfolio['balance'] *= (1 + step_return)
        new_portfolio['position'] = action
        
        # Idling penalty
        if action == 0 and self.portfolio['position'] == 0:
            new_portfolio['balance'] *= (1 - self.game_config['idling_penalty'])

        # Max Drawdown update
        new_portfolio['peak_balance'] = max(new_portfolio['peak_balance'], new_portfolio['balance'])
        drawdown = (new_portfolio['peak_balance'] - new_portfolio['balance']) / (new_portfolio['peak_balance'] + 1e-9)
        new_portfolio['max_drawdown'] = max(new_portfolio['max_drawdown'], drawdown)
        
        # Accumulate fantasy returns
        new_portfolio['sum_returns'] += step_return
        new_portfolio['sum_sq_returns'] += step_return ** 2
        new_portfolio['returns_count'] += 1
        
        next_tick = self.current_tick + 1
        
        new_state = GameState(
            self.market_data, self.close_prices, next_tick, self.end_tick, 
            new_portfolio, self.game_config, reward_tick=self.reward_tick,
            custom_binary=next_binary, wm_depth=self.wm_depth + 1
        )
        new_state.norm_scale = scale # Persist normalization scale
        
        return new_state, float(step_return), new_state.isEndGame

    def _get_value(self):
        # Calculate terminal value if we hit lookahead boundary, DD boundary, or passed reward_tick
        if self.isEndGame or self.current_tick >= self.reward_tick:
            if self.portfolio['max_drawdown'] > self.game_config['dd_threshold']:
                return (-1.0, 0, 0) # Maximum punishment for deep DD
            
            n = self.portfolio['returns_count']
            if n < 20: return (0, 0, 0) # Neutral for tiny episodes
            
            mu = self.portfolio['sum_returns'] / n
            var = (self.portfolio['sum_sq_returns'] / n) - (mu ** 2)
            sigma = np.sqrt(max(0, var)) + 1e-9
            
            # Sharpe-based reward (Raw episode Sharpe, no annualization)
            sharpe = (mu / sigma)
            # Use tanh_scale from config as a multiplier for better reward distribution
            z = np.tanh(sharpe * self.game_config['tanh_scale']) 
            return (z, 0, 0)
        
        return (0, 0, 0)

    def takeAction(self, action):
        new_portfolio = self.portfolio.copy()
        
        # PHASE 6: FORCED FLAT BUFFER (Lookahead)
        # If we are past the official reward_tick, we stop trading and lock performance
        if self.current_tick >= self.reward_tick:
            # We move forward in time but no more returns are accumulated
            next_tick = self.current_tick + 1
            # Next tick must not exceed absolute data boundary
            if next_tick >= len(self.close_prices):
                next_tick = self.current_tick
            
            new_portfolio['step_count'] += 1
            new_portfolio['position'] = 0 # Forced flat in buffer
            
            new_state = GameState(
                self.market_data, self.close_prices, next_tick, self.end_tick, 
                new_portfolio, self.game_config, reward_tick=self.reward_tick
            )
            return new_state, 0.0, new_state.isEndGame

        new_portfolio['step_count'] += 1
        current_price = self.close_prices[self.current_tick]
        next_tick = self.current_tick + 1
        
        if next_tick >= len(self.close_prices):
            # If we hit the absolute end of data, return a terminal state at the current tick.
            return GameState(self.market_data, self.close_prices, self.current_tick, self.current_tick, new_portfolio, self.game_config, reward_tick=self.reward_tick), 0.0, True

        next_price = self.close_prices[next_tick]
        
        # Calculate step return based on the NEW action (Removes 1-tick lag)
        step_return = 0.0
        if action == 1: # LONG
            step_return = (next_price - current_price) / current_price
        elif action == 2: # SHORT
            step_return = (current_price - next_price) / current_price
        
        # Subtract fees if position changed
        if action != self.portfolio['position']:
            step_return -= self.game_config['fee']
            new_portfolio['trade_count'] += 1
            
        # Update balance
        new_portfolio['balance'] *= (1 + step_return)
        
        # LIQUIDATION CHECK: Prevent balance sign flips/explosions
        if new_portfolio['balance'] <= self.game_config['initial_balance'] * 0.05:
            new_portfolio['balance'] = 0.0
            new_portfolio['position'] = 0
            new_state = GameState(self.market_data, self.close_prices, next_tick, next_tick, new_portfolio, self.game_config, reward_tick=self.reward_tick)
            return new_state, -1.0, True

        # Idling penalty to prevent "do nothing" strategy in trending markets
        if action == 0 and self.portfolio['position'] == 0:
            new_portfolio['balance'] *= (1 - self.game_config['idling_penalty'])

        # Update peak and drawdown
        new_portfolio['peak_balance'] = max(new_portfolio['peak_balance'], new_portfolio['balance'])
        drawdown = (new_portfolio['peak_balance'] - new_portfolio['balance']) / (new_portfolio['peak_balance'] + 1e-9)
        new_portfolio['max_drawdown'] = max(new_portfolio['max_drawdown'], drawdown)
        
        # Accumulate returns for terminal reward
        new_portfolio['sum_returns'] += step_return
        new_portfolio['sum_sq_returns'] += step_return ** 2
        new_portfolio['returns_count'] += 1
        
        new_portfolio['position'] = action
        
        new_state = GameState(self.market_data, self.close_prices, next_tick, self.end_tick, new_portfolio, self.game_config, reward_tick=self.reward_tick)
        
        return new_state, float(step_return), new_state.isEndGame
