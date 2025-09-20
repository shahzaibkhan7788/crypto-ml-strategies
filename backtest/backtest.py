import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.binance.data_downloader import data_downloader
from utils import store_signal_stegnth_data, table_exists
import psycopg2
import pandas as pd


class BackTest:
    def __init__(self,signal_df,ohlvc_df, fee, tp_threshold, sl_threshold):
        self.tp_threshold= tp_threshold
        self.sl_threshold= sl_threshold
        self.fee_in_percentage= fee
        self.initial_balance= 1000
        self.new_schema="Ledger"
        self.strategy_table_name= "strategies_indicators"
        self.signal_df=signal_df
        self.ohlvc_df= ohlvc_df
        self.merged_df= self.merge_data()
        
        print("name of the columns in the singal df:",self.signal_df.columns)
        print("name of the columns in the singal df:",self.ohlvc_df.columns)
        
       

    
    def merge_data(self):
        self.ohlvc_df.rename(columns={'timestamp': 'datetime'}, inplace=True)
        self.signal_df.rename(columns={'timestamp': 'datetime'}, inplace=True)
        self.ohlvc_df['datetime'] = pd.to_datetime(self.ohlvc_df['datetime'])

        self.signal_df['datetime'] = pd.to_datetime(self.signal_df['datetime'])

        # Now merge safely
        merged_df = pd.merge(
            self.ohlvc_df,
            self.signal_df,
            on="datetime",
            how="left"
        ).sort_values(by="datetime")
        merged_df.rename(columns={'aggregated_signal':'predicted_direction'}, inplace=True)
        print("here is the merged df:",merged_df)
        return merged_df
    
    def calculate_trade_action(self,merged_df=None):
        print("Calcualte trade action")
        
        if merged_df is None:
            df = self.merged_df.copy()
        else:
            df= merged_df.copy()
        result = []

        balance = self.initial_balance
        fee = self.fee_in_percentage / 100  # 0.05% = 0.0005
        tp_threshold = self.tp_threshold / 100
        sl_threshold = self.sl_threshold / 100

        trade_open = False
        direction = None
        entry_price = None
        pnl_sum = 0

        for idx, row in df.iterrows():
            timestamp = row['datetime']
            signal = row['predicted_direction']
            open_price = row['open']
            high = row['high']
            low = row['low']
            close = row['close']

            action = None
            buy_price = 0
            sell_price = 0
            pnl_percent = 0

            # ───── Trade Opening ───── #
            if not trade_open:
                if signal == 1:
                    action = "buy"
                    direction = "long"
                    entry_price = open_price
                    balance *= (1 - fee)
                    pnl_percent = -fee * 100
                    pnl_sum += pnl_percent
                    trade_open = True
                    buy_price = entry_price

                    result.append({
                        'datetime': timestamp,
                        'predicted_direction': direction,
                        'action': action,
                        'buy_price': buy_price,
                        'sell_price': 0,
                        'balance': round(balance, 2),
                        'pnl': round(pnl_percent, 2),
                        'pnl_sum': round(pnl_sum, 2)
                    })

                elif signal == -1:
                    action = "sell"
                    direction = "short"
                    entry_price = open_price
                    balance *= (1 - fee)
                    pnl_percent = -fee * 100
                    pnl_sum += pnl_percent
                    trade_open = True
                    sell_price = entry_price

                    result.append({
                        'datetime': timestamp,
                        'predicted_direction': direction,
                        'action': action,
                        'buy_price': 0,
                        'sell_price': sell_price,
                        'balance': round(balance, 2),
                        'pnl': round(pnl_percent, 2),
                        'pnl_sum': round(pnl_sum, 2)
                    })

            # ───── Trade Monitoring ───── #
            else:
                if direction == "long":
                    # Take Profit
                    if high >= entry_price * (1 + tp_threshold):
                        exit_price = entry_price * (1 + tp_threshold)
                        raw_profit = ((exit_price - entry_price) / entry_price) * 100
                        pnl_percent = raw_profit - fee * 100
                        balance *= (1 + raw_profit / 100) * (1 - fee)
                        pnl_sum += pnl_percent
                        action = "sell - take_profit"
                        sell_price = exit_price
                        trade_open = False

                    # Stop Loss
                    elif low <= entry_price * (1 - sl_threshold):
                        exit_price = entry_price * (1 - sl_threshold)
                        raw_loss = -((entry_price - exit_price) / entry_price) * 100
                        pnl_percent = raw_loss - fee * 100
                        balance *= (1 + raw_loss / 100) * (1 - fee)
                        pnl_sum += pnl_percent
                        action = "sell - stop_loss"
                        sell_price = exit_price
                        trade_open = False

                    # Direction Change
                    elif signal == -1:
                        exit_price = open_price
                        raw_change = ((exit_price - entry_price) / entry_price) * 100
                        pnl_percent = raw_change - fee * 100
                        balance *= (1 + raw_change / 100) * (1 - fee)
                        pnl_sum += pnl_percent
                        action = "sell - direction change"
                        sell_price = exit_price

                        # Append exit trade
                        result.append({
                            'datetime': timestamp,
                            'predicted_direction': direction,
                            'action': action,
                            'buy_price': 0,
                            'sell_price': sell_price,
                            'balance': round(balance, 2),
                            'pnl': round(pnl_percent, 2),
                            'pnl_sum': round(pnl_sum, 2)
                        })

                        # Start new short trade
                        action = "sell"
                        direction = "short"
                        entry_price = open_price
                        balance *= (1 - fee)
                        pnl_percent = -fee * 100
                        pnl_sum += pnl_percent
                        sell_price = entry_price
                        trade_open = True

                        result.append({
                            'datetime': timestamp,
                            'predicted_direction': direction,
                            'action': action,
                            'buy_price': 0,
                            'sell_price': sell_price,
                            'balance': round(balance, 2),
                            'pnl': round(pnl_percent, 2),
                            'pnl_sum': round(pnl_sum, 2)
                        })
                        continue  # Skip rest to avoid double append

                elif direction == "short":
                    # Take Profit
                    if low <= entry_price * (1 - tp_threshold):
                        exit_price = entry_price * (1 - tp_threshold)
                        raw_profit = ((entry_price - exit_price) / entry_price) * 100
                        pnl_percent = raw_profit - fee * 100
                        balance *= (1 + raw_profit / 100) * (1 - fee)
                        pnl_sum += pnl_percent
                        action = "buy - take_profit"
                        buy_price = exit_price
                        trade_open = False

                    # Stop Loss
                    elif high >= entry_price * (1 + sl_threshold):
                        exit_price = entry_price * (1 + sl_threshold)
                        raw_loss = -((exit_price - entry_price) / entry_price) * 100
                        pnl_percent = raw_loss - fee * 100
                        balance *= (1 + raw_loss / 100) * (1 - fee)
                        pnl_sum += pnl_percent
                        action = "buy - stop_loss"
                        buy_price = exit_price
                        trade_open = False

                    # Direction Change
                    elif signal == 1:
                        exit_price = open_price
                        raw_change = ((entry_price - exit_price) / entry_price) * 100
                        pnl_percent = raw_change - fee * 100
                        balance *= (1 + raw_change / 100) * (1 - fee)
                        pnl_sum += pnl_percent
                        action = "buy - direction change"
                        buy_price = exit_price

                        # Append exit trade
                        result.append({
                            'datetime': timestamp,
                            'predicted_direction': direction,
                            'action': action,
                            'buy_price': buy_price,
                            'sell_price': 0,
                            'balance': round(balance, 2),
                            'pnl': round(pnl_percent, 2),
                            'pnl_sum': round(pnl_sum, 2)
                        })

                        # Start new long trade
                        action = "buy"
                        direction = "long"
                        entry_price = open_price
                        balance *= (1 - fee)
                        pnl_percent = -fee * 100
                        pnl_sum += pnl_percent
                        buy_price = entry_price
                        trade_open = True

                        result.append({
                            'datetime': timestamp,
                            'predicted_direction': direction,
                            'action': action,
                            'buy_price': buy_price,
                            'sell_price': 0,
                            'balance': round(balance, 2),
                            'pnl': round(pnl_percent, 2),
                            'pnl_sum': round(pnl_sum, 2)
                        })
                        continue

                if action:
                    result.append({
                        'datetime': timestamp,
                        'predicted_direction': direction,
                        'action': action,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'balance': round(balance, 2),
                        'pnl': round(pnl_percent, 2),
                        'pnl_sum': round(pnl_sum, 2)
                    })

        self.trade_df = pd.DataFrame(result)
        return self.trade_df


    def schema(self,backtest_df,backtest_table_name):
        store_signal_stegnth_data(self.new_schema,backtest_table_name,backtest_df)
    
    def calculate_signal_strength(self):
        print("the signal calculation calss called")
        self.merge_data()
        df= self.calculate_trade_action()
        print("columns of the calcualte signal strength:",df.columns)
        print("here is the pnl sum pdf:",df)
        last_pnl_sum = df['pnl_sum'].iloc[-1]
        print("here is the pnl sum:",last_pnl_sum)
        return {"pnl_sum": last_pnl_sum},df
    
    
    
    
    
    
    
    