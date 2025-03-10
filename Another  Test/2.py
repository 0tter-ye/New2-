import time
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from pybit.unified_trading import HTTP
import logging
import MetaTrader5 as mt5

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from config import API_KEY, API_SECRET  # Ensure config.py exists

# Initialize Bybit API
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
symbol = "BTCUSDT"

# Initialize MT5 connection
if not mt5.initialize():
    logging.error(f"Failed to initialize MT5: {mt5.last_error()}")
    exit()

mt5_login = 3263156
mt5_password = "Bl1SrRhFb0JP@E4"
mt5_server = "Bybit-Demo"
if not mt5.login(mt5_login, mt5_password, mt5_server):
    logging.error(f"Failed to login to MT5: {mt5.last_error()}")
    exit()
logging.info("Connected to MT5 demo account")

account_info = mt5.account_info()
if account_info is None:
    logging.error("Failed to fetch MT5 account info")
    exit()
initial_balance = account_info.balance
logging.info(f"MT5 Account Balance: {initial_balance} USDT")

symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    logging.warning(f"{symbol} not found. Searching for BTCUSDT variant...")
    all_symbols = [s.name for s in mt5.symbols_get() if "BTCUSDT" in s.name]
    if not all_symbols:
        logging.error("No BTCUSDT variant found in MT5. Check Market Watch.")
        exit()
    symbol = all_symbols[0]
    logging.info(f"Using symbol: {symbol}")
    symbol_info = mt5.symbol_info(symbol)

raw_point = symbol_info.point
min_volume = symbol_info.volume_min
max_volume = symbol_info.volume_max
volume_step = symbol_info.volume_step

class Wallet:
    def __init__(self, max_risk_per_trade=0.005, max_drawdown=0.2):
        account_info = mt5.account_info()
        self.balance = account_info.balance if account_info else 0
        self.initial_balance = self.balance
        self.positions = {}
        self.trade_history = []
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.paused = False
        self.max_profit = {}  # Still tracking max_profit for potential future use, but not for trailing stop

    def sync_balance(self):
        account_info = mt5.account_info()
        if account_info is not None:
            self.balance = account_info.balance
            drawdown = (self.initial_balance - self.balance) / self.initial_balance
            if drawdown >= self.max_drawdown:
                self.paused = True
                logging.warning(f"Max drawdown ({self.max_drawdown*100}%) reached. Trading paused.")
            total_unrealized_pnl = sum(self.calculate_unrealized_pnl(sym, mt5.symbol_info_tick(sym).last) for sym in self.positions)
            if total_unrealized_pnl < -0.05 * self.initial_balance:
                self.paused = True
                logging.warning("Total unrealized PNL <-5% of initial balance. Trading paused.")
        else:
            logging.warning("Failed to sync balance with MT5")

    def calculate_position_size(self, price, risk_distance, fixed_qty=None, volume_multiplier=1.0):
        if fixed_qty is not None:
            return adjust_volume(fixed_qty, min_volume, max_volume, volume_step)
        risk_amount = self.balance * self.max_risk_per_trade
        qty = (risk_amount / risk_distance) * volume_multiplier
        return adjust_volume(qty, min_volume, max_volume, volume_step)

    def open_position(self, symbol, side, qty, price, df, stop_loss=None, take_profit=None):
        if self.paused:
            logging.warning("Trading paused due to drawdown or PNL limit.")
            return False
        cost = qty * price
        self.sync_balance()
        if self.balance >= cost:
            self.balance -= cost
            self.positions[symbol] = {
                'qty': qty, 'entry_price': price, 'side': side,
                'open_time': datetime.now(),
                'entry_imbalance': calculate_bid_ask_imbalance(*fetch_order_book(symbol, df)[1:3]),
                'entry_volume_spike': df["volume_spike_1m"].iloc[-1] if "volume_spike_1m" in df.columns else 0
            }
            self.max_profit[symbol] = 0
            logging.info(f"Opened {side} position: {qty} {symbol} @ {price}")
            return True
        logging.warning(f"Insufficient funds: {cost} > {self.balance}")
        return False

    def calculate_unrealized_pnl(self, symbol, current_price):
        if symbol not in self.positions:
            return 0
        pos = self.positions[symbol]
        qty = pos['qty']
        entry_price = pos['entry_price']
        side = pos['side']
        if side == "Buy":
            return qty * (current_price - entry_price)
        else:
            return qty * (entry_price - current_price)

    def close_position(self, symbol, price):
        if symbol in self.positions:
            pos = self.positions[symbol]
            unrealized_pnl = self.calculate_unrealized_pnl(symbol, price)
            self.balance += (pos['qty'] * price if pos['side'] == "Buy" else pos['qty'] * (pos['entry_price'] * 2 - price))
            trade = {
                'symbol': symbol, 'side': pos['side'], 'qty': pos['qty'],
                'entry_price': pos['entry_price'], 'exit_price': price,
                'profit': unrealized_pnl, 'timestamp': datetime.now(),
                'entry_imbalance': pos['entry_imbalance'],
                'entry_volume_spike': pos['entry_volume_spike']
            }
            self.trade_history.append(trade)
            logging.info(f"Closed {pos['side']} position: {pos['qty']} {symbol} @ {price}, Profit: {unrealized_pnl}")
            del self.positions[symbol]
            del self.max_profit[symbol]
            self.sync_balance()

    def get_performance_summary(self, trades=None):
        if trades is None:
            trades = self.trade_history
        self.sync_balance()
        total_profit = sum(trade['profit'] for trade in trades)
        total_trades = len(trades)
        final_value = self.initial_balance + total_profit
        total_return = (final_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        win_rate = len([t for t in trades if t['profit'] > 0]) / total_trades * 100 if total_trades > 0 else 0
        return {
            'start_value': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return * 100,
            'total_trades': total_trades,
            'win_rate': win_rate
        }

wallet = Wallet(max_risk_per_trade=0.005)

class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.transition_matrix = transition_matrix
        self.current_state = random.choice(states)
        self.stationary_dist = self.compute_stationary_distribution()

    def compute_stationary_distribution(self):
        P = np.array(self.transition_matrix)
        n = P.shape[0]
        A = P.T - np.eye(n)
        A[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.clip(pi, 0, None)
        pi /= pi.sum()
        return pi.tolist()

    def next_state(self):
        try:
            probabilities = self.transition_matrix[self.states.index(self.current_state)]
            probabilities = np.clip(probabilities, 0, None)
            prob_sum = np.sum(probabilities)
            if prob_sum == 0:
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities = probabilities / prob_sum
            self.current_state = np.random.choice(self.states, p=probabilities)
            return self.current_state, self.stationary_dist
        except Exception as e:
            logging.error(f"Error in Markov Chain transition: {e}")
            self.current_state = random.choice(self.states)
            return self.current_state, [0.5, 0.5]

states = ["Loss", "Win"]
transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float)
markov_chain = MarkovChain(states, transition_matrix)
logging.info(f"Markov Stationary Dist: {markov_chain.stationary_dist}")

def fetch_data(symbol, timeframe, limit=200):
    try:
        response = client.get_kline(category="linear", symbol=symbol, interval=timeframe, limit=limit)
        if "result" not in response or "list" not in response["result"]:
            logging.error(f"Invalid API response structure for {timeframe} timeframe")
            return pd.DataFrame()
        
        data = response["result"]["list"]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
        df[["open", "high", "low", "close", "volume", "turnover"]] = df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)
        df["returns"] = df["close"].pct_change()
        df["high_low"] = df["high"] - df["low"]
        df["high_close_prev"] = abs(df["high"] - df["close"].shift(1))
        df["low_close_prev"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["high_low", "high_close_prev", "low_close_prev"]].max(axis=1)
        df["atr"] = df["tr"].rolling(window=14).mean()
        df["volume_mean"] = df["volume"].rolling(window=20).mean()
        df["volume_std"] = df["volume"].rolling(window=20).std()
        df["volume_spike"] = (df["volume"] > df["volume_mean"] + 2 * df["volume_std"]).astype(int)
        df = df.drop(columns=["high_low", "high_close_prev", "low_close_prev", "tr"])
        df = df.dropna()
        df.columns = [f"{col}_{timeframe}m" if col not in ["timestamp"] else col for col in df.columns]
        df = df.sort_values("timestamp")
        return df
    except Exception as e:
        logging.error(f"Error fetching {timeframe}min data: {e}")
        return pd.DataFrame()

def calculate_sma(df, period, timeframe="5m"):
    return df[f"close_{timeframe}"].rolling(window=period).mean()

def calculate_rsi(df, period=14, timeframe="5m"):
    delta = df[f"close_{timeframe}"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def is_trend_reversal(df, short_period=10, long_period=30, timeframe="5m"):
    sma_short = calculate_sma(df, short_period, timeframe)
    sma_long = calculate_sma(df, long_period, timeframe)
    rsi = calculate_rsi(df, timeframe="5m")
    
    last_sma_short = sma_short.iloc[-1]
    last_sma_long = sma_long.iloc[-1]
    prev_sma_short = sma_short.iloc[-2]
    prev_sma_long = sma_long.iloc[-2]
    last_rsi = rsi.iloc[-1]
    
    bullish_cross = prev_sma_short <= prev_sma_long and last_sma_short > last_sma_long and last_rsi > 50
    bearish_cross = prev_sma_short >= prev_sma_long and last_sma_short < last_sma_long and last_rsi < 50
    return bullish_cross, bearish_cross

def calculate_support_resistance(df, window=20, timeframe="5m"):
    df = df.copy()
    df["pivot_high"] = df[f"high_{timeframe}"].rolling(window=window, center=True).max()
    df["pivot_low"] = df[f"low_{timeframe}"].rolling(window=window, center=True).min()
    
    resistance_levels = df[df[f"high_{timeframe}"] == df["pivot_high"]][f"high_{timeframe}"].value_counts()
    resistance_levels = resistance_levels[resistance_levels >= 2].index.tolist()
    resistance_levels = sorted(resistance_levels, reverse=True)[:3]
    
    support_levels = df[df[f"low_{timeframe}"] == df["pivot_low"]][f"low_{timeframe}"].value_counts()
    support_levels = support_levels[support_levels >= 2].index.tolist()
    support_levels = sorted(support_levels)[:3]
    
    return support_levels, resistance_levels

def fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=200):
    dfs = {}
    for tf in timeframes:
        df = fetch_data(symbol, tf, limit)
        if not df.empty:
            dfs[tf] = df
        else:
            logging.warning(f"Failed to fetch {tf}min data")
    
    if not dfs:
        logging.error("No data fetched for any timeframe. Exiting.")
        return pd.DataFrame()
    
    base_tf = min(timeframes, key=int)
    combined_df = dfs[base_tf]
    for tf in timeframes:
        if tf != base_tf:
            combined_df = pd.merge_asof(
                combined_df,
                dfs[tf],
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(minutes=int(tf)),
                suffixes=("", f"_{tf}m")
            )
    combined_df = combined_df.dropna()
    combined_df["sma_short_5m"] = calculate_sma(combined_df, 10, "5m")
    combined_df["sma_long_5m"] = calculate_sma(combined_df, 30, "5m")
    combined_df["rsi_5m"] = calculate_rsi(combined_df, 14, "5m")
    combined_df["macd_5m"] = combined_df["close_5m"].ewm(span=12).mean() - combined_df["close_5m"].ewm(span=26).mean()
    combined_df = combined_df.dropna()
    logging.info(f"Combined data shape: {combined_df.shape}, Timeframes: {timeframes}")
    return combined_df

def fetch_order_book(symbol, df, depth=10):
    try:
        response = client.get_orderbook(category="linear", symbol=symbol, limit=depth)
        if "result" not in response or "b" not in response["result"] or "a" not in response["result"]:
            logging.warning(f"Failed to fetch order book for {symbol}. Using fallback.")
            return [100.0], [10], [100.1], [10]
        
        bids = response["result"]["b"]
        asks = response["result"]["a"]
        bid_prices = [float(b[0]) for b in bids]
        bid_volumes = [float(b[1]) for b in bids]
        ask_prices = [float(a[0]) for a in asks]
        ask_volumes = [float(a[1]) for a in asks]
        
        current_price = (max(bid_prices) + min(ask_prices)) / 2
        last_close = df["close_5m"].iloc[-1] if not df.empty else None
        if last_close and abs(current_price - last_close) / last_close > 0.5:
            logging.warning(f"Order book price {current_price} deviates significantly from close {last_close}. Using close price.")
            current_price = last_close
        
        logging.debug(f"Order book fetched: {len(bid_prices)} bids, {len(ask_prices)} asks")
        return bid_prices, bid_volumes, ask_prices, ask_volumes
    except Exception as e:
        logging.error(f"Error fetching order book: {e}")
        return [100.0], [10], [100.1], [10]

def calculate_bid_ask_imbalance(bid_volumes, ask_volumes):
    total_bid_volume = sum(bid_volumes)
    total_ask_volume = sum(ask_volumes)
    return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0

def adjust_volume(volume, min_vol, max_vol, step):
    volume = max(min_vol, min(max_vol, volume))
    volume = round(volume / step) * step
    return round(volume, 6)

def train_models(df):
    if df.empty or not any(col.endswith("_5m") for col in df.columns):
        logging.error("DataFrame is empty or missing required 5m columns")
        return None, None, None
    
    feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
    X = df[feature_cols].iloc[:-1]
    y = (df["close_5m"].shift(-1) > df["close_5m"]).astype(int).iloc[:-1]
    if len(y) < 10:
        logging.warning("Not enough data for training")
        return None, None, None
    
    logging.info(f"Training data class distribution: {np.bincount(y)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = rf_model.score(X_test_scaled, y_test)

    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_accuracy = dt_model.score(X_test_scaled, y_test)

    logging.info(f"RF Model Accuracy: {rf_accuracy:.2f}")
    logging.info(f"DT Model Accuracy: {dt_accuracy:.2f}, Max Depth: {dt_model.get_depth()}, Leaves: {dt_model.get_n_leaves()}")
    return rf_model, dt_model, scaler

rf_model, dt_model, scaler = None, None, None

df_initial = fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=1000)
if df_initial.empty:
    logging.error("Initial data fetch failed. Exiting.")
    exit()
rf_model, dt_model, scaler = train_models(df_initial)
if rf_model is None or dt_model is None or scaler is None:
    logging.error("Model training failed. Exiting.")
    exit()

last_entry_time = None

def execute_trade(prediction, symbol, df, confidence_threshold=0.60, markov_win_prob=0.0):
    global last_entry_time
    if wallet.paused:
        logging.warning("Trading paused.")
        return
    
    side = "Buy" if prediction == 1 else "Sell"
    try:
        bid_prices, bid_volumes, ask_prices, ask_volumes = fetch_order_book(symbol, df, depth=10)
        current_price = (min(ask_prices) + max(bid_prices)) / 2
        if current_price == 100.05:
            current_price = df["close_5m"].iloc[-1]
            logging.warning(f"Using fallback price: {current_price}")

        atr_5m = df["atr_5m"].iloc[-1]
        volume_spike_1m = df["volume_spike_1m"].iloc[-1] if "volume_spike_1m" in df.columns else 0
        feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
        X_latest = pd.DataFrame(df[feature_cols].iloc[-1]).T
        X_latest_scaled = scaler.transform(X_latest)
        rf_confidence = rf_model.predict_proba(X_latest_scaled)[0][1] if prediction == 1 else 1 - rf_model.predict_proba(X_latest_scaled)[0][1]
        dt_confidence = dt_model.predict_proba(X_latest_scaled)[0][1] if prediction == 1 else 1 - dt_model.predict_proba(X_latest_scaled)[0][1]
        
        imbalance = calculate_bid_ask_imbalance(bid_volumes, ask_volumes)
        bullish_reversal, bearish_reversal = is_trend_reversal(df)
        
        adjusted_threshold = 0.65 if side == "Sell" else max(0.50, 0.60 + confidence_adjustment)
        
        if (side == "Sell" and (imbalance >= -0.3 or not bearish_reversal)) or \
           (side == "Buy" and (imbalance <= 0.3 or not bullish_reversal)):
            logging.debug(f"Entry skipped: {side} - Imbalance {imbalance:.2f}, Reversal {bullish_reversal}/{bearish_reversal}")
            return
        
        now = datetime.now()
        if last_entry_time and (now - last_entry_time).total_seconds() < 300 and not volume_spike_1m:
            logging.debug(f"Skipping trade: Too soon since last entry ({(now - last_entry_time).total_seconds()}s)")
            return
        
        big_move_factor = 1.0
        if volume_spike_1m and imbalance > 0.3 and side == "Buy":
            big_move_factor = 1.2
        elif volume_spike_1m and imbalance < -0.3 and side == "Sell":
            big_move_factor = 1.2
        
        total_score = (0.35 * max(0, (rf_confidence - adjusted_threshold) / (1 - adjusted_threshold)) +
                       0.15 * max(0, (dt_confidence - adjusted_threshold) / (1 - adjusted_threshold)) +
                       0.2 * max(0, (markov_win_prob / 100 - 0.60) / 0.40)) * big_move_factor
        total_score = np.clip(total_score, 0, 1)
        
        volume_range = [0.01, 0.02, 0.03, 0.04, 0.05]
        trade_qty = volume_range[min(int(total_score * (len(volume_range) - 1)), len(volume_range) - 1)]
        volume_multiplier = 1.2 if side == "Buy" and df["rsi_5m"].iloc[-1] < 30 else 1.0
        adjusted_qty = wallet.calculate_position_size(current_price, atr_5m * 8, fixed_qty=trade_qty, volume_multiplier=volume_multiplier)

        if adjusted_qty < min_volume:
            logging.warning(f"Trade qty {adjusted_qty} below minimum {min_volume}. Skipping.")
            return

        cost = adjusted_qty * current_price
        wallet.sync_balance()
        if cost > wallet.balance:
            logging.warning(f"Trade cost {cost:.2f} USDT exceeds balance {wallet.balance:.2f} USDT. Skipping.")
            return

        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": adjusted_qty,
            "type": order_type,
            "price": current_price,
            "deviation": 20,
            "magic": 123456,
            "comment": f"{side} via Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed: {result.comment} (retcode: {result.retcode})")
            return

        logging.info(f"Placed {side} order: {adjusted_qty} {symbol} @ {current_price}, Imbalance: {imbalance:.2f}, Volume Spike: {volume_spike_1m}")
        wallet.open_position(symbol, side, adjusted_qty, current_price, df)
        last_entry_time = now

    except Exception as e:
        logging.error(f"Error executing trade: {e}")

def main_loop():
    global rf_model, dt_model, scaler, last_entry_time
    iteration = 0
    start_time = datetime.now()
    last_retrain = datetime.now()
    retrain_interval = timedelta(minutes=30)
    confidence_adjustment = 0.0

    while True:
        try:
            df = fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=200)
            if df.empty:
                logging.warning("No data fetched, skipping iteration.")
                time.sleep(5)
                continue

            logging.debug(f"Iteration {iteration} - Data columns: {df.columns.tolist()}")

            if datetime.now() - last_retrain >= retrain_interval:
                rf_model, dt_model, scaler = train_models(df)
                last_retrain = datetime.now()

            current_state, stationary_dist = markov_chain.next_state()
            markov_win_prob = stationary_dist[1] * 100
            feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
            X_latest = pd.DataFrame(df[feature_cols].iloc[-1]).T
            X_latest_scaled = scaler.transform(X_latest)
            
            rf_pred = rf_model.predict_proba(X_latest_scaled)[0][1]
            dt_pred = dt_model.predict_proba(X_latest_scaled)[0][1]
            logging.debug(f"Iteration {iteration} - Markov State: {current_state}, Win Prob: {markov_win_prob:.2f}%, RF Pred: {rf_pred:.2f}, DT Pred: {dt_pred:.2f}")
            
            if iteration == 1:
                prediction = 1 if rf_pred > 0.5 else 0
                logging.info(f"Iteration {iteration} - Forcing test trade: RF Pred: {rf_pred:.2f}, DT Pred: {dt_pred:.2f}, Prediction: {prediction}")
                execute_trade(prediction, symbol, df, confidence_threshold=0.50, markov_win_prob=markov_win_prob)
            elif current_state == "Win" and markov_win_prob >= 40:
                prediction = 1 if rf_pred > 0.30 or dt_pred > 0.30 else 0
                adjusted_threshold = max(0.50, 0.60 + confidence_adjustment)
                logging.info(f"Iteration {iteration} - RF Win Prob: {rf_pred*100:.2f}%, DT Win Prob: {dt_pred*100:.2f}%, Prediction: {prediction}, Adjusted Threshold: {adjusted_threshold}")
                execute_trade(prediction, symbol, df, confidence_threshold=adjusted_threshold, markov_win_prob=markov_win_prob)
            else:
                logging.debug(f"Iteration {iteration} - Entry conditions not met: State={current_state}, Markov Prob={markov_win_prob:.2f}%")

            current_price = df["close_5m"].iloc[-1]
            bid_prices, bid_volumes, ask_prices, ask_volumes = fetch_order_book(symbol, df, depth=10)
            imbalance = calculate_bid_ask_imbalance(bid_volumes, ask_volumes)
            
            wallet_summary = wallet.get_performance_summary()
            total_return = wallet_summary["total_return"]
            base_profit_target = 3.0
            atr_5m = df["atr_5m"].iloc[-1]
            rsi_5m = df["rsi_5m"].iloc[-1]
            
            volatility_factor = atr_5m / df["close_5m"].iloc[-1] * 100
            dynamic_profit_target = base_profit_target * (1 + volatility_factor / 100)
            if total_return > 5:
                dynamic_profit_target += (total_return / 10)
                logging.info(f"Wallet up {total_return:.2f}%, Volatility-Adjusted Profit Target: {dynamic_profit_target:.2f}%")
            elif total_return < -5:
                dynamic_profit_target = max(1.0, base_profit_target - 1.0)
                logging.info(f"Wallet down {total_return:.2f}%, Reduced Profit Target: {dynamic_profit_target:.2f}%")

            bullish_reversal, bearish_reversal = is_trend_reversal(df)
            support_levels, resistance_levels = calculate_support_resistance(df, window=20, timeframe="5m")
            logging.debug(f"Iteration {iteration} - Support Levels: {support_levels}, Resistance Levels: {resistance_levels}")

            atr_mean = df["atr_5m"].rolling(window=20).mean().iloc[-1]
            high_volatility = atr_5m > 2 * atr_mean

            for sym, pos in list(wallet.positions.items()):
                unrealized_pnl = wallet.calculate_unrealized_pnl(sym, current_price)
                entry_price = pos["entry_price"]
                side = pos["side"]
                time_open = (datetime.now() - pos["open_time"]).total_seconds() / 60

                profit_percent = (unrealized_pnl / (pos["qty"] * entry_price)) * 100
                loss_percent = -profit_percent if profit_percent < 0 else 0

                wallet.max_profit[sym] = max(wallet.max_profit[sym], profit_percent)  # Still tracked but not used for trailing stop

                near_resistance = any(abs(current_price - r) / current_price < 0.005 for r in resistance_levels)
                near_support = any(abs(current_price - s) / current_price < 0.005 for s in support_levels)
                
                if side == "Buy" and near_resistance and (imbalance < -0.2 or profit_percent > 0):
                    logging.info(f"Iteration {iteration} - Near resistance {resistance_levels}, Imbalance: {imbalance:.2f}, closing Buy with profit: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)
                elif side == "Sell" and near_support and (imbalance > 0.2 or profit_percent > 0):
                    logging.info(f"Iteration {iteration} - Near support {support_levels}, Imbalance: {imbalance:.2f}, closing Sell with profit: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)
                elif profit_percent >= dynamic_profit_target:
                    logging.info(f"Iteration {iteration} - Dynamic profit target hit: {profit_percent:.2f}% (Target: {dynamic_profit_target:.2f}%)")
                    wallet.close_position(sym, current_price)
                elif profit_percent >= 1.5 and rf_model.predict_proba(X_latest_scaled)[0][1 if side == "Buy" else 0] < 0.5:
                    logging.info(f"Iteration {iteration} - Profit secured due to reversal signal: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)
                elif side == "Buy" and rsi_5m > 70 and profit_percent > 0:
                    logging.info(f"Iteration {iteration} - Overbought (RSI: {rsi_5m:.2f}), closing Buy with profit: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)
                elif side == "Sell" and rsi_5m < 30 and profit_percent > 0:
                    logging.info(f"Iteration {iteration} - Oversold (RSI: {rsi_5m:.2f}), closing Sell with profit: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)
                elif side == "Buy" and bearish_reversal and profit_percent > 0:
                    logging.info(f"Iteration {iteration} - Bearish reversal detected, closing Buy with profit: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)
                elif side == "Sell" and bullish_reversal and profit_percent > 0:
                    logging.info(f"Iteration {iteration} - Bullish reversal detected, closing Sell with profit: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)
                elif side == "Buy" and imbalance < -0.5 and df["volume_spike_1m"].iloc[-1]:
                    logging.info(f"Iteration {iteration} - Big sell detected, closing Buy: Imbalance {imbalance:.2f}")
                    wallet.close_position(sym, current_price)
                elif side == "Sell" and imbalance > 0.5 and df["volume_spike_1m"].iloc[-1]:
                    logging.info(f"Iteration {iteration} - Big buy detected, closing Sell: Imbalance {imbalance:.2f}")
                    wallet.close_position(sym, current_price)
                elif time_open >= (5 if side == "Sell" else 10):
                    logging.info(f"Iteration {iteration} - Time limit hit: {time_open:.1f} minutes, Profit: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)
                elif side == "Sell" and high_volatility and profit_percent > 0:
                    logging.info(f"Iteration {iteration} - High volatility (ATR: {atr_5m:.2f}), closing Sell with profit: {profit_percent:.2f}%")
                    wallet.close_position(sym, current_price)

            iteration += 1
            if iteration % 10 == 0:
                summary = wallet.get_performance_summary()
                logging.info(f"Iteration {iteration} - Overall Performance: Start Value: {summary['start_value']:.2f}, "
                             f"Final Value: {summary['final_value']:.2f}, Total Return: {summary['total_return']:.2f}%, "
                             f"Trades: {summary['total_trades']}, Win Rate: {summary['win_rate']:.2f}%")

            time.sleep(5)
        except Exception as e:
            logging.error(f"Iteration {iteration} - Unexpected error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main_loop()
