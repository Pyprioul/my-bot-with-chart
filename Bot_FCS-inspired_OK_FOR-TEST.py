import pandas as pd
import numpy as np
import talib as ta

# Load historical data (OHLCV)
data = pd.read_csv('data/cleaned_BTCUSDT_1h_last_5_years.csv')

# Ensure the column names are lowercase to match your dataset
data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

# Parameters for EMA, RSI, ATR, MACD, and Fibonacci levels
EMA_short_period = 9
SMA_long_period = 26
RSI_period = 5
ATR_period = 14
Fibonacci_levels = [0.382, 0.618, 0.786]

# Calculate EMA, SMA, RSI, ATR, and MACD
data['EMA_short'] = ta.EMA(data['Close'], timeperiod=EMA_short_period)
data['SMA_long'] = ta.SMA(data['Close'], timeperiod=SMA_long_period)
data['RSI'] = ta.RSI(data['Close'], timeperiod=RSI_period)
data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=ATR_period)
data['MACD'], data['MACD_signal'], _ = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Generate buy/sell signals based on strategy conditions
data['Buy_Signal'] = np.where((data['EMA_short'] > data['SMA_long']) & (data['RSI'] < 60), 1, 0)
data['Sell_Signal'] = np.where((data['EMA_short'] < data['SMA_long']) & (data['RSI'] > 40), 1, 0)

# Dynamic stop-loss and take-profit using ATR
data['Stop_Loss'] = data['Low'].rolling(window=ATR_period).min() - data['ATR']
data['Take_Profit'] = data['High'].rolling(window=ATR_period).max() + data['ATR']

# Initialize variables
initial_balance = 10000.0
current_balance = initial_balance
peak_balance = initial_balance
risk_per_trade = 0.02
drawdown_limit = 0.30
trade_log = []

def execute_trades(data):
    global current_balance, peak_balance
    position = 0  # 1 for long, 0 for no position
    entry_price = 0
    trailing_stop = 0

    for i in range(1, len(data)):
        if data['Buy_Signal'][i] == 1 and position == 0:
            # Enter buy trade
            entry_price = data['Close'][i]
            stop_loss = data['Stop_Loss'][i]
            take_profit = data['Take_Profit'][i]
            position_size = round((current_balance * risk_per_trade) / abs(entry_price - stop_loss), 2)
            trailing_stop = data['Close'][i] - data['ATR'][i]  # Initial trailing stop

            trade_log.append({
                'Date': data['timestamp'][i],
                'Type': 'Buy',
                'Price': round(entry_price, 2),
                'Balance': round(current_balance, 2),
                'Drawdown': round((peak_balance - current_balance) / peak_balance, 4) * 100
            })
            position = 1

        elif position == 1:
            # Adjust trailing stop or execute sell
            trailing_stop = max(trailing_stop, data['Close'][i] - data['ATR'][i] * 1.5)
            if data['Close'][i] <= trailing_stop or data['Sell_Signal'][i] == 1:
                # Exit position
                exit_price = data['Close'][i]
                pnl = round((exit_price - entry_price) * position_size, 2)
                current_balance += pnl
                peak_balance = max(peak_balance, current_balance)

                trade_log.append({
                    'Date': data['timestamp'][i],
                    'Type': 'Sell',
                    'Price': round(exit_price, 2),
                    'Balance': round(current_balance, 2),
                    'PnL': round(pnl, 2),
                    'Drawdown': round((peak_balance - current_balance) / peak_balance, 4) * 100
                })
                position = 0

        # Check for max drawdown and exit if limit hit
        drawdown = (peak_balance - current_balance) / peak_balance
        if drawdown >= drawdown_limit:
            print("Max drawdown hit. Stopping trading.")
            break

# Execute trades and log the results
execute_trades(data)

# Save the trade log to CSV
pd.DataFrame(trade_log).to_csv('bot_output.log', index=False)
