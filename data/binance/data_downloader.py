import sys
import os
import configparser
from datetime import datetime 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import psycopg2
from data.binance.crypto_dataloader import crypto
from utils import interval_validation,table_exists, data_interpolation, normalize_timeframe

from configparser import ConfigParser

from utility.config_loader import load_config
config = load_config()



def data_downloader(exchange, symbol, time_horizon,start_time=None,end_time=None):
    conn = psycopg2.connect(
        host="localhost",
        database="exchange",
        user="postgres",
        password="shah7788",
        port=5432)

    table_name = f"{exchange}_{symbol}usdt_1min".lower()
    if start_time==None:
        start_time = config['Data'].get('start_time')
        print("start time from config:",start_time)
    if end_time==None:
        end_time= config['Data'].get("end_time")
        if end_time == 'now':
            end_time = datetime.now().strftime('%Y-%m-%d')
        
        interpolation = config['Interpolation'].getboolean('interpolation')
        print("interpolation:",interpolation)
        print("end time:",end_time)
                

  
    standard_time= normalize_timeframe(exchange,time_horizon)
    print("start time at data downloader:",start_time)
    print("end time at data downloader:",end_time)

    interval_validation(exchange, time_horizon)
    query = f"""SELECT * FROM {exchange}.{table_name}
    WHERE timestamp >= '{start_time}' 
    AND timestamp <= '{end_time}'
    ORDER BY timestamp ASC;"""

    if table_exists(conn, table_name, exchange):
        print("Table exists:", f"{exchange}.{table_name}")
        df = pd.read_sql(query, conn)
        if isinstance(time_horizon, str) and time_horizon.lower() in ["1min", "1m"]:
            return df
        else:
            data = crypto(exchange, symbol, standard_time,start_time,end_time)
            resampled_df= data.resample_ohlcv(df, exchange, time_horizon)
            print("resample df:",resampled_df)
            return df, resampled_df

    else:
        print("Table does not exist, data will fetch.")
        data = crypto(exchange, symbol, standard_time,start_time)
        if exchange=="binance":
            df = data.fetch_binance_data()
         
        if exchange=="bybit":
            df= data.fetch_bybit_data()
          
        df= data_interpolation(df, interpolation)
        data.store_ohlcv_data(df)
        df = pd.read_sql(query, conn)
        
        if isinstance(time_horizon, str) and time_horizon.lower() in ["1min", "1m"]:
            return df
        else:
            resampled_df= data.resample_ohlcv(df, exchange, time_horizon)
            return df, resampled_df
    conn.close()

