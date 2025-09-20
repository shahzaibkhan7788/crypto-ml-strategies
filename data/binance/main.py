import sys
import os
import configparser
import re
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import psycopg2
from crypto_dataloader import crypto
from utils import * 

config_path = os.path.join(os.path.dirname(__file__), 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)
print("path of configration file",config_path)


valid_intervals = {
        "binance": ["1m", "1h", "1d", "1w", "1M"],
        "bybit": ["1", "D", "W", "M"]}
    

def main():
    exchange= config['Data'].get('exchange')
    symbols = config['Data'].get("symbol")
    time_horizon = config['Data'].get('time_horizon')
    start_time = config['Data'].get('start_time')
    interpolation = config['Data'].getboolean('interpolatation')
    end_time = config['Data'].get('end_time')
    
    print("exchange",exchange)
    print("symbol",symbols)
    print("time horizon",time_horizon)
    print("start time",start_time)
    print("interpolation",interpolation)
    print("end time",end_time)

    if end_time == 'now':
        end_time = datetime.now().strftime('%Y-%m-%d')


    symbol_list = [s.strip() for s in symbols.split(',')]
    exchange_list = [s.strip().lower() for s in exchange.split(',')]
    
    print("symbol list:",symbol_list)
    print("exchange list:",exchange_list)


    interval_validation(exchange, time_horizon)
    for ex in exchange_list:
        for symbol in symbol_list:
            standard_time = normalize_timeframe(ex, time_horizon)
            data = crypto(ex, symbol, standard_time, start_time,end_time)
            df = None 
            try:
                if ex == "binance" and standard_time in valid_intervals['binance']:
                    df = data.fetch_binance_data()
                    print("binance data fetch call and its length is:",len(df))
                elif ex == "bybit" and standard_time in valid_intervals['bybit']:
                    df = data.fetch_bybit_data()
                    print("bybit data fetch length",len(df))
            except Exception as e:
                print(f"Error fetching data for {symbol} on {ex}: {e}")
                continue 

            if df is not None and not df.empty:
                df = data_interpolation(df, interpolation)
                data.store_ohlcv_data(df)
            else:
                print(f"No data returned for {symbol} on {ex} with interval {standard_time}")

                

if __name__ == '__main__':
    main()
