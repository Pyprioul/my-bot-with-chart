import pandas as pd
from binance.client import Client
import time
from loguru import logger
import sys
import os
from datetime import datetime

# Fetch Binance API credentials from environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize Binance Client
client = Client(API_KEY, API_SECRET)

# Set up logger for both file and console
log_file = "/home/ubuntu/combined_paper_trading_log.txt"
logger.add(log_file, format="{message}", level="INFO")
logger.add(sys.stdout, format="{message}", level="INFO")

# Define the initial balance and trade details
initial_balance = 10000  # Initial balance
current_balance = initial_balance
overall_drawdown = 0  # Declare drawdown globally, initializing to 0
max_drawdown_limit = 30  # Maximum allowed drawdown (30%)
entry_price = None  # Variable to store entry price after a BUY signal
stop_loss = None  # Variable to store stop-loss price
cumulative_profit = 0  # Track overall profit

# Paper Trading Tracker
open_position = None

# Trade parameters
trade_size_multiplier = 0.05  # 5% of current balance per trade
atr_multiplier = 1.5  # ATR multiplier for stop-loss calculation
transaction_fee = 0.001  # Binance fee (0.1%)
max_trade_size = 50000  # Maximum trade size

# To track max profit and loss
max_profit = float('-inf')
max_loss = float('inf')


# Function to fetch real-time market data from Binance
def get_binance_data(symbol):
    try:
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=100)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav',
                                            'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None


# Calculate RSI
def calculate_rsi(df, period=5):
    if len(df) < period:
        logger.warning(f"Not enough data to calculate RSI for period {period}. Needed: {period}, Available: {len(df)}")
        df['rsi'] = float('nan')
        return df

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    loss = loss.replace(0, 1e-10)  # Avoid division by zero
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


# Calculate ATR
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=period).mean()
    return df


# Ichimoku components
def ichimoku(df, period1=9, period2=26, period3=52):
    high_9 = df['high'].rolling(window=period1).max()
    low_9 = df['low'].rolling(window=period1).min()
    tenkan_sen = (high_9 + low_9) / 2

    high_26 = df['high'].rolling(window=period2).max()
    low_26 = df['low'].rolling(window=period2).min()
    kijun_sen = (high_26 + low_26) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period2)
    high_52 = df['high'].rolling(window=period3).max()
    low_52 = df['low'].rolling(window=period3).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(period2)

    chikou_span = df['close'].shift(-period2)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


# Function to generate Ichimoku signals (no logging of Ichimoku values)
def ichimoku_signal(index, df):
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = ichimoku(df)

    if tenkan_sen.iloc[index] > kijun_sen.iloc[index] and df['close'].iloc[index] > senkou_span_a.iloc[index]:
        return "BUY"
    elif tenkan_sen.iloc[index] < kijun_sen.iloc[index] and df['close'].iloc[index] < senkou_span_a.iloc[index]:
        return "SELL"
    return None


# Function to simulate trade execution
def execute_trade(row, balance, atr_value, atr_multiplier):
    global current_balance, overall_drawdown, max_profit, max_loss, entry_price, stop_loss, cumulative_profit

    price = row['close']
    trade_value = balance * trade_size_multiplier
    trade_value = min(trade_value, max_trade_size)

    # Simulate BUY signal
    if row['signal'] == "BUY" and entry_price is None:
        entry_price = price
        stop_loss = entry_price - (atr_value * atr_multiplier)
        current_balance -= trade_value
        log_trade('BUY', entry_price, None, trade_value, stop_loss, current_balance)

    # Simulate SELL signal or hitting stop-loss, but only if entry_price is not None
    elif (row['signal'] == "SELL" or (stop_loss is not None and price <= stop_loss)) and entry_price is not None:
        profit_loss = trade_value * (price / entry_price - 1)
        current_balance += trade_value + profit_loss
        cumulative_profit += profit_loss
        log_trade('SELL', entry_price, price, trade_value, stop_loss, current_balance)
        log_pnl(profit_loss)  # Log PnL for the trade
        log_drawdown()  # Log drawdown only when a trade happens

        entry_price = None
        stop_loss = None

    return True


# Function to log trades
def log_trade(action, entry_price, closing_price, trade_value, stop_loss, balance):
    logger.info(
        f"{action} | Entry Price: {entry_price} | Closing Price: {closing_price or 'N/A'} | Trade Value: {trade_value} | Stop Loss: {stop_loss} | Balance: {balance}")


# Function to log drawdown only when a trade happens
def log_drawdown():
    global overall_drawdown, current_balance, initial_balance
    drawdown = 100 * (initial_balance - current_balance) / initial_balance
    overall_drawdown = max(overall_drawdown, drawdown)
    logger.info(f"Drawdown after this trade: {drawdown:.2f}%")


# Function to log Profit and Loss (PnL) after each trade
def log_pnl(profit_loss):
    logger.info(f"Profit/Loss for this trade: {profit_loss:.2f} USDT")


# Paper trading function
def paper_trading(symbol):
    while True:
        data = get_binance_data(symbol)
        if data is None or len(data) < 52:  # Ensure enough data for both RSI and Ichimoku
            logger.warning("Insufficient data to proceed with trading")
            time.sleep(60)
            continue

        # Calculate indicators
        data = calculate_rsi(data, period=5)
        data = calculate_atr(data, period=14)

        current_price = data['close'].iloc[-1]
        logger.info(f"Current Price of {symbol}: {current_price}")

        # No logging for RSI
        data['signal'] = ichimoku_signal(-1, data)

        # Execute trade
        atr_value = data['atr'].iloc[-1]
        execute_trade(data.iloc[-1], current_balance, atr_value, atr_multiplier)

        time.sleep(60)


# Run paper trading bot for BTC/USDT pair
if __name__ == "__main__":
    paper_trading('BTCUSDT')
