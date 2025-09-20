import os
import sys
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from data.binance.data_downloader import data_downloader
from utils import extract_categories_of_true_indicators, table_exists
from datetime import timedelta
from datetime import datetime
import psycopg2
import numpy as np
import pandas as pd
import sqlite3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from backtest import BackTest_Demo
from utils import store_signal_stegnth_data



class strategy_reader:
    def __init__(self,search_table):
        self.search_table= search_table
        self.schema= "demo_trading"
        print("strategy name:",self.search_table)
    
    def load_ohlvc_signal_data(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database="exchange",
            user="postgres",
            password="shah7788",
            port=5432
        )

        if table_exists(self.conn, "strategies_indicators", "public"):
            meta_df = pd.read_sql("SELECT * FROM public.strategies_indicators", self.conn)
       

            if self.search_table in meta_df['strategy_name'].values:
                match_row = meta_df[meta_df['strategy_name'] == self.search_table].iloc[0]

                self.exchange = match_row['exchange']
                self.symbol = match_row['symbol']
                self.time_in_hour = str(match_row['time_horizon']) 
                
                if self.time_in_hour.endswith("h"):
                    self.time_horizon = int(self.time_in_hour.rstrip("h"))*60
                    
                                    
                print("Time horizon for strategy:", self.time_horizon)
                print("Time for a strategy:", self.time_in_hour)


               
                self.true_columns = [col for col, val in match_row.items()
                                if isinstance(val, (bool, np.bool_)) and val == True]

     
                numeric_suffix_columns = []

                filtered_true_columns = []

                for prefix in self.true_columns:
                    # Check if there's a column like prefix_period (or prefix_something)
                    for col_name, val in match_row.items():
                        if col_name.startswith(prefix + "_") and isinstance(val, (int, float, np.integer, np.floating)):
                            if val == 1.0:
                                break  # Skip this prefix if value is 1.0
                    else:
                        filtered_true_columns.append(prefix)
                        
                numeric_suffix_columns = []
                true_boolean_prefixes = [
                    col_name for col_name, val in match_row.items()
                    if isinstance(val, (bool, np.bool_)) and bool(val)  # works for both types
                ]
              
                # Step 2: For each true boolean prefix, find numeric suffix columns
                for prefix in true_boolean_prefixes:
                    for col_name, val in match_row.items():
                        if col_name.startswith(prefix + "_") and isinstance(val, (int, float, np.integer, np.floating)):
                            numeric_suffix_columns.append((col_name, val))
                


                # Step 3: Build params dictionary (no filtering — keeps all found columns)
                params = defaultdict(dict)
                for col_name, val in numeric_suffix_columns:
                    try:
                        prefix, suffix = col_name.split('_', 1)
                        params[prefix][suffix] = val
                    except ValueError:
                        continue  # Skip malformed names

                params = dict(params)

                # Step 4: Extract ALL numeric suffix column names into a list
                numeric_values = [col for col, _ in numeric_suffix_columns]


                for prefix in self.true_columns:
                    suffix_cols = [col for col in match_row.index if col.startswith(prefix + "_")]
                    for col in suffix_cols:
                        val = match_row[col]
                        if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                            numeric_values.append(val)
                            
           
                
                self.true_indicators,self.true_indicators_with_categories= extract_categories_of_true_indicators(params)
                self.paramters= params
             
                now = datetime.now()
                
                                # Step 4: Find max numeric value
                if numeric_suffix_columns:
                    numeric_values_only = [int(val) for _, val in numeric_suffix_columns]  # ensure Python int
                    self.overall_max = max(numeric_values_only)
                    print("Max parameter window:", self.overall_max)
                    print("self.time horizon:",self.time_horizon)

                    # Step 5: Calculate times
                    total_shift_minutes = self.time_horizon * self.overall_max + 3*self.time_horizon
                    total_shift_minutes = int(total_shift_minutes)  # ensure Python int
                    print("Total shift value (minutes):", total_shift_minutes)

                    self.floored_initial_time = (now - timedelta(minutes=total_shift_minutes)).replace(second=0, microsecond=0)
                else:
                    self.floored_initial_time = (now - timedelta(minutes=self.time_horizon)).replace(second=0, microsecond=0)
                    
                
                                # Step 6: Ceil final time (always runs)
                if now.second > 0 or now.microsecond > 0:
                    self.ceiled_final_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
                else:
                    self.ceiled_final_time = now.replace(second=0, microsecond=0)

                # Step 7: Final print
                print(f"Initial Time: {self.floored_initial_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Final Time:   {self.ceiled_final_time.strftime('%Y-%m-%d %H:%M:%S')}")

                self.ohlvc_table = f"{self.exchange}_{self.symbol}usdt_1min".lower()
                result= data_downloader(self.exchange,self.symbol,self.time_in_hour,self.floored_initial_time,self.ceiled_final_time)
                
                self.resample_df,self.ohlvc = result[1],result[0] if isinstance(result, tuple) else result
                print("Resample df:",self.resample_df)
                print("ohlvc df:",self.ohlvc)

            else:
                print(f"Table '{self.search_table}' not found in public.strategies_indicators.")
        else:
            print("Schema or table does not exist: public.strategies_indicators.")
            
    
    def aggregate_signal(self):
        from signalgenerator.technical_indicator_signal.signal_generator import SignalGenerator
        from strategies.strategy_pipeline.utils.indicator_utils import aggregated_dataframe
        from indicator.indicator_calculator import technical_indicator
        
        indicator = technical_indicator(self.resample_df, 
                                        self.true_indicators_with_categories, 
                                        self.true_indicators, 
                                        self.paramters)
        
        indicators_dict = indicator.calculate_indicators(self.overall_max)
        signalgenerator = SignalGenerator(indicators_dict)
        signal_generator_df = signalgenerator.generate_singal()
        self.aggregated_df= aggregated_dataframe(signal_generator_df)
        # Keep only the last (max timestamp) row
        self.aggregated_df = self.aggregated_df.loc[self.aggregated_df['timestamp'] == self.aggregated_df['timestamp'].max()]

        print("aggregated df (last timestamp only):", self.aggregated_df)

        backtester = BackTest_Demo(
            signal_df=self.aggregated_df,
            ohlvc_df=self.ohlvc,
            fee=0.05,  # Example: 0.05% fee
            tp_threshold=5.0,  # 1% take-profit
            sl_threshold=3,
            symbol=self.symbol,
            time_horizon=self.time_in_hour,
            exchange= self.exchange
            )
        
        backtester.run_strategy(quantity=0.2)
    


if __name__=="__main__":
    search_table= "strategy_28"
    strategy= strategy_reader(search_table)
    strategy.load_ohlvc_signal_data()
    strategy.aggregate_signal()
  

    
