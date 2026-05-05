import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
import time
import os
from datetime import datetime
from game import GameState

class LiveTradingGame:
    def __init__(self, config, exchange_config=None):
        self.config = config
        self.name = 'trading_live'
        
        self.window_size = config['trading']['window_size']
        self.symbol = config['trading']['symbol']
        self.timeframe = config['trading']['timeframe']
        
        # Determine Market Type from ENV or Config
        self.market_type = os.getenv('BINANCE_MARKET_TYPE', 'spot').lower()
        
        self.feature_columns = ['log_return', 'rsi', 'atr', 'vol_10', 'vol_30', 'rel_high_low', 'vol_delta']
        self.n_features = len(self.feature_columns)
        self.n_portfolio = 5
        
        self.input_shape = (self.window_size, self.n_features + self.n_portfolio)
        self.action_size = 3
        self.state_size = self.window_size * (self.n_features + self.n_portfolio)
        
        # 1. Public Exchange (For high-quality historical data)
        self.public_exchange = ccxt.binance({'enableRateLimit': True})
        
        # 2. Private Exchange (For NEW Demo Trading / Paper Trading)
        if exchange_config:
            params = exchange_config.copy()
            params['options'] = params.get('options', {})
            
            # Set market type
            params['options']['defaultType'] = 'future' if self.market_type == 'future' else 'spot'
            
            # Check for testnet status
            is_testnet = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true' or config['trading'].get('testnet', False)
            
            if is_testnet:
                print(f"🔧 Initializing NATIVE Binance {self.market_type.upper()} Demo Trading...")
                # 1. Internal flag ( Silences SAPI margin calls during load_markets )
                params['options']['enableDemoTrading'] = True
                params['options']['fetchMargins'] = False
                params['options']['fetchBalanceMethod'] = 'privateGetAccount'
            
            # Initialize
            self.exchange = ccxt.binance(params)
            
            if is_testnet:
                # 2. URL Swap ( Mandatory in CCXT 4.5.x as flag doesn't auto-switch URLs )
                self.exchange.urls['api'] = self.exchange.urls['demo']
                
                print(f"DEBUG: Market Type: {self.market_type}")
                print(f"DEBUG: Base URL Set To: {self.exchange.urls['api']['public']}")
                if self.market_type == 'future':
                    print(f"DEBUG FAPI URL: {self.exchange.urls['api'].get('fapiPrivate', 'NOT SET')}")
        else:
            self.exchange = self.public_exchange

        self.game_config = {
            'window_size': self.window_size,
            'initial_balance': config['trading']['initial_balance'],
            'fee': config['trading']['fee'],
            'tanh_scale': config['trading'].get('tanh_scale', 3.0),
            'dd_threshold': config['trading'].get('drawdown_threshold', 0.12),
            'idling_penalty': config['trading'].get('idling_penalty', 0.0001),
            'feature_columns': self.feature_columns,
            'n_features': self.n_features,
            'n_portfolio': self.n_portfolio
        }

    def fetch_latest_state(self):
        """
        Fetches live data and returns a GameState object.
        """
        try:
            # 1. Fetch OHLCV from Public Exchange
            limit = 500 
            ohlcv = self.public_exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                print(f"❌ CRITICAL: Public exchange returned 0 candles for {self.symbol}")
                raise ValueError("No data returned from exchange")
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            print(f"✅ Fetched {len(df)} candles from Public API.")
            
            # 2. Calculate Features
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['vol_10'] = df['log_return'].rolling(window=10).std()
            df['vol_30'] = df['log_return'].rolling(window=30).std()
            df['high_30'] = df['high'].rolling(window=30).max()
            df['low_30'] = df['low'].rolling(window=30).min()
            df['rel_high_low'] = (df['close'] - df['low_30']) / (df['high_30'] - df['low_30'] + 1e-9)
            df['vol_delta'] = df['volume'].diff() / (df['volume'].shift(1) + 1e-9)
            
            initial_count = len(df)
            df.dropna(inplace=True)
            print(f"📊 After dropna: {len(df)} rows remaining (Dropped {initial_count - len(df)} rows).")

            if len(df) < self.window_size:
                print(f"❌ ERROR: Not enough valid data rows ({len(df)}) to fill window_size ({self.window_size})")
                raise ValueError("Insufficient data after indicator calculation")

            # Pad with 20 dummy rows of "future" data (copy of last row) for MCTS lookahead
            lookahead_padding = 20
            latest_data = df.tail(self.window_size)
            
            market_data_array = latest_data[self.feature_columns].values.astype(np.float32)
            padding = np.tile(market_data_array[-1:], (lookahead_padding, 1))
            market_data_padded = np.concatenate([market_data_array, padding], axis=0)
            
            close_prices = latest_data['close'].values.astype(np.float32)
            price_padding = np.full(lookahead_padding, close_prices[-1], dtype=np.float32)
            close_prices_padded = np.concatenate([close_prices, price_padding], axis=0)
            
            print(f"✅ Market data array shape: {market_data_array.shape}, Padded: {market_data_padded.shape}")
            
            # 3. Get Portfolio State
            total_balance = self.game_config['initial_balance']
            position = 0
            
            if self.exchange.apiKey and self.exchange.secret:
                # RECURRING DEBUG
                print(f"DEBUG: Market Type: {self.market_type.upper()}")
                print(f"DEBUG PUBLIC URL: {self.exchange.urls['api'].get('public', 'NOT SET')}")
                if self.market_type == 'future':
                    print(f"DEBUG FAPI URL: {self.exchange.urls['api'].get('fapiPrivate', 'NOT SET')}")
                
                try:
                    if self.market_type == 'spot':
                        raw = self.exchange.privateGetAccount()
                        balance = {
                            'total': {b['asset']: float(b['free']) + float(b['locked']) 
                                      for b in raw['balances'] if float(b['free']) + float(b['locked']) > 0}
                        }
                    else:
                        balance = self.exchange.fetch_balance({'type': 'future'})
                    
                    base, quote = self.symbol.split('/')

                    if self.market_type == 'future':
                        total_balance = balance['total'].get('USDT', 0.0)
                        positions = balance.get('info', {}).get('positions', [])
                        for pos in positions:
                            if pos['symbol'] == self.symbol.replace('/', ''):
                                amt = float(pos['positionAmt'])
                                if amt > 0: position = 1
                                elif amt < 0: position = 2
                    else:
                        total_balance = balance['total'].get(quote, 0.0)
                        btc_held = balance['total'].get(base, 0.0)
                        current_price = close_prices[-1]
                        total_balance += btc_held * current_price
                        if btc_held * current_price > 10:
                            position = 1
                except Exception as e:
                    if "2015" in str(e):
                        print(f"❌ BINANCE ERROR 2015: Invalid API Key or Permissions for {self.market_type.upper()}.")
                    else:
                        print(f"⚠️ Warning: Private balance fetch failed ({e}). using mock.")
            portfolio = {
                'balance': float(total_balance),
                'peak_balance': float(total_balance),
                'max_drawdown': 0.0,
                'position': position,
                'entry_price': 0.0,
                'trade_count': 0,
                'realized_pnl': 0.0,
                'step_count': 0,
                'sum_returns': 0.0,
                'sum_sq_returns': 0.0,
                'returns_count': 0
            }
            
            # Reset current_tick to window_size - 1 for live session
            state = GameState(
                market_data_padded, close_prices_padded, self.window_size - 1, self.window_size + lookahead_padding - 1,
                portfolio, self.game_config
            )
            return state
            
        except Exception as e:
            print(f"❌ Error in fetch_latest_state: {e}")
            raise e

    def execute_action(self, action):
        """
        Executes an order on Binance.
        action: 0=Flat, 1=Long, 2=Short
        """
        if not self.exchange.apiKey or not self.exchange.secret:
            print("SKIPPING ORDER: No API keys configured in .env")
            return

        try:
            # 1. Get current balance and holdings
            if self.market_type == 'spot':
                raw = self.exchange.privateGetAccount()
                balance = {
                    'total': {b['asset']: float(b['free']) + float(b['locked']) 
                              for b in raw['balances'] if float(b['free']) + float(b['locked']) > 0}
                }
            else:
                balance = self.exchange.fetch_balance({'type': 'future'})
            
            base, quote = self.symbol.split('/')
            symbol_unified = self.symbol
            symbol_exchange = self.symbol.replace('/', '') if self.market_type == 'future' else self.symbol
            
            if self.market_type == 'future':
                # --- FUTURES EXECUTION ---
                pos_amt = 0.0
                positions = balance.get('info', {}).get('positions', [])
                for pos in positions:
                    if pos['symbol'] == symbol_exchange:
                        pos_amt = float(pos['positionAmt'])
                        break
                
                if action == 0: # WANT FLAT
                    if pos_amt > 0: self.exchange.create_market_sell_order(symbol_unified, pos_amt, {'reduceOnly': True})
                    elif pos_amt < 0: self.exchange.create_market_buy_order(symbol_unified, abs(pos_amt), {'reduceOnly': True})
                
                elif action == 1: # WANT LONG
                    if pos_amt <= 0:
                        if pos_amt < 0: self.exchange.create_market_buy_order(symbol_unified, abs(pos_amt), {'reduceOnly': True})
                        usdt = balance['total'].get('USDT', 0.0)
                        price = self.exchange.fetch_ticker(symbol_unified)['last']
                        amount = (usdt * 0.95) / price
                        self.exchange.create_market_buy_order(symbol_unified, amount)
                
                elif action == 2: # WANT SHORT
                    if pos_amt >= 0:
                        if pos_amt > 0: self.exchange.create_market_sell_order(symbol_unified, pos_amt, {'reduceOnly': True})
                        usdt = balance['total'].get('USDT', 0.0)
                        price = self.exchange.fetch_ticker(symbol_unified)['last']
                        amount = (usdt * 0.95) / price
                        self.exchange.create_market_sell_order(symbol_unified, amount)

            else:
                # --- SPOT EXECUTION ---
                btc_held = balance['total'].get(base, 0.0)
                usdt_held = balance['total'].get(quote, 0.0)
                
                if action == 1: # WANT LONG
                    if btc_held * self.exchange.fetch_ticker(self.symbol)['last'] < 10:
                        print(f"Opening LONG position on {self.symbol}...")
                        price = self.exchange.fetch_ticker(self.symbol)['last']
                        amount = (usdt_held * 0.98) / price
                        if amount > 0:
                            self.exchange.create_market_buy_order(self.symbol, amount)
                else: 
                    if btc_held * self.exchange.fetch_ticker(self.symbol)['last'] > 10:
                        print(f"Closing position on {self.symbol}...")
                        self.exchange.create_market_sell_order(self.symbol, btc_held)
                    
        except Exception as e:
            print(f"CCXT Execution Error: {e}")
