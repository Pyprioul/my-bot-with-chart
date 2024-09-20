import pandas as pd
from binance.client import Client
import time
from loguru import logger
import sys
import random
import os
from datetime import datetime

# Fetch Binance API credentials from environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize Binance Client
client = Client(API_KEY, API_SECRET)

# Set up logger for both file and console
log_file = "/home/ubuntu/bot_output.log"
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

# To handle real-world trading (e.g., slippage, gap risk)
slippage_percent_base = 0.001  # 0.1% slippage base

# Function to fetch real-time market data from Binance
def get_binance_data(symbol):
    try:
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=100)
        for candle in candles:
            return {
                'timestamp': candle[0],
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            }
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

# Calculate RSI
def calculate_rsi(df, period=5):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
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

# Function to calculate Ichimoku signals (buy/sell decisions)
def ichimoku_signal(index, df):
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = ichimoku(df)
    if tenkan_sen.iloc[index] > kijun_sen.iloc[index] and df['close'].iloc[index] > senkou_span_a.iloc[index]:
        return "BUY"
    elif tenkan_sen.iloc[index] < kijun_sen.iloc[index] and df['close'].iloc[index] < senkou_span_a.iloc[index]:
        return "SELL"
    return None

# Function to calculate Ichimoku components
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

# Function to log trade details
def log_trade(trade_action, entry_price, closing_price, trade_value, slippage, stop_loss, realized_pl, cumulative_pl, final_balance):
    closing_price_str = f"{closing_price:,.2f}" if closing_price is not None else "N/A"
    stop_loss_str = f"{stop_loss:,.2f}" if stop_loss is not None else "N/A"

    logger.info(
        f"{trade_action:<10} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Entry Price: {entry_price:,.2f} | "
        f"Closing Price: {closing_price_str} | Trade Value: {trade_value:,.2f} | Slippage: {slippage:,.2f} | "
        f"Stop Loss: {stop_loss_str} | Realized P/L: {realized_pl:,.2f} | Cumulative P/L: {cumulative_pl:,.2f} | "
        f"Final Balance: {final_balance:,.2f}"
    )

# Simulate trade execution (buy/sell)
def execute_trade(row, balance, atr_value, atr_multiplier):
    global current_balance, overall_drawdown, max_profit, max_loss, entry_price, stop_loss, cumulative_profit

    price = row['close']
    trade_value = balance * trade_size_multiplier
    trade_value = min(trade_value, max_trade_size)  # Ensure trade value doesn't exceed liquidity constraint

    if row['rsi'] < 30:  # Example: Increase trade size for strong buy signal
        trade_value *= 1.5
    elif row['rsi'] > 70:  # Example: Reduce trade size for overbought condition
        trade_value *= 0.75

    # Simulate BUY signal
    if row['signal'] == "BUY" and entry_price is None:
        entry_price = price
        stop_loss = entry_price - (atr_value * atr_multiplier)  # ATR-based stop-loss
        stop_loss = apply_gap_risk(price, stop_loss)
        slippage = calculate_dynamic_slippage(trade_value, atr_value)
        current_balance -= trade_value + slippage
        current_balance -= trade_value * transaction_fee
        log_trade('BUY', entry_price, None, trade_value, slippage, stop_loss, 0, cumulative_profit, current_balance)

    # Simulate SELL signal or hitting stop-loss
    elif (row['signal'] == "SELL" or (stop_loss is not None and price <= stop_loss)) and entry_price is not None:
        profit_loss = trade_value * (price / entry_price - 1)
        current_balance += trade_value + profit_loss
        current_balance -= trade_value * transaction_fee
        cumulative_profit += profit_loss
        log_trade('SELL', entry_price, price, trade_value, 0, stop_loss, profit_loss, cumulative_profit, current_balance)
        max_profit = max(max_profit, profit_loss)
        max_loss = min(max_loss, profit_loss)
        entry_price = None
        stop_loss = None

    # Calculate drawdown and track overall max drawdown
    drawdown = 100 * (initial_balance - current_balance) / initial_balance
    overall_drawdown = max(overall_drawdown, drawdown)

    if drawdown > max_drawdown_limit:
        logger.info(f"Drawdown limit reached: {drawdown}% - Stopping trading.")
        return False

    return True

# Function to simulate slippage based on trade size and volatility
def calculate_dynamic_slippage(trade_value, atr_value):
    slippage_percent = slippage_percent_base + (trade_value / max_trade_size) * 0.002
    slippage = trade_value * slippage_percent
    return slippage

# Simulate potential gap risk (slippage beyond stop-loss)
def apply_gap_risk(price, stop_loss):
    gap_chance = random.random()
    if gap_chance < 0.05:  # 5% chance of a gap risk
        gap_percentage = random.uniform(0.02, 0.10)  # Random gap size between 2% and 10%
        new_stop_loss = stop_loss * (1 - gap_percentage)
        logger.info(f"Gap risk applied: new stop-loss at {new_stop_loss}")
        return new_stop_loss
    return stop_loss

def simulate_trade(action, symbol, price, trade_size):
    # Example trade simulation logic
    logger.info(f"Simulating {action} trade for {symbol} at price {price:.2f} with trade size {trade_size:.2f}")

def paper_trading(symbol):
    global stop_loss  # Reference the global stop_loss

    while True:
        try:
            # Fetch data
            data = get_binance_data(symbol)

            if data:
                # Convert the data (dict) into a DataFrame
                row = pd.DataFrame([data])  # Ensure data is wrapped in a list to avoid errors

                # Calculate indicators
                row['rsi'] = calculate_rsi(row, period=5)['rsi']  # RSI period = 5
                row['atr'] = calculate_atr(row, period=14)['atr']  # ATR period

                current_price = row['close'].iloc[0]
                logger.info(f"Current Price of {symbol}: {current_price:.2f}")

                # Check for stop-loss hit
                if open_position and current_price <= stop_loss:
                    logger.info(f"Stop-loss hit at {current_price:.2f}")
                    simulate_trade('SELL', symbol, current_price, trade_size_multiplier)

                # Generate signals
                row['signal'] = ichimoku_signal(0, row)
                atr_value = row['atr'].iloc[0]

                trade_continues = execute_trade(row.iloc[0], current_balance, atr_value, atr_multiplier)
                if not trade_continues:
                    break

        except Exception as e:
            logger.error(f"Error occurred: {e}")
        time.sleep(60)


# Run paper trading bot for BTC/USDT pair
if __name__ == "__main__":
    paper_trading('BTCUSDT')

