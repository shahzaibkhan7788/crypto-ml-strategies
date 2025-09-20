from datetime import datetime
import psycopg2
from data.binance.crypto_dataloader import crypto
from utils import normalize_timeframe


conn = psycopg2.connect(
                host="localhost",
                database="exchange",
                user="postgres",
                password="shah7788",
                port=5432
            )
cursor = conn.cursor()



if __name__ == '__main__':
    exchange_list= ["binance","bybit"]
    symbol_list= ["BTC","ETH","SOL"]
    time_horizon= "1min"
    for ex in exchange_list:
        for symbol in symbol_list:
            standard_time = normalize_timeframe(ex, time_horizon)
    
            table_name = f"{ex}.{ex}_{symbol}usdt_1min".lower()
            query = f"SELECT MAX(timestamp) FROM {table_name};"
            cursor.execute(query)
            print("table name to fetch data:",table_name)
            start_time = cursor.fetchone()[0]
            end_time = datetime.now().replace(second=0, microsecond=0)
            data= crypto(ex,symbol,standard_time,start_time,end_time)
            df = None 
            try:
                if ex == "binance":
                    df = data.fetch_binance_data()
                    print("binance data fetch call and its length is:",len(df))
                if ex == "bybit":
                    df = data.fetch_bybit_data()
                    print("bybit data fetch length",len(df))
            except Exception as e:
                print(f"Error fetching data for {symbol} on {ex}: {e}")
                continue 

            if df is not None and not df.empty:
                data.store_ohlcv_data(df)
            else:
                print(f"No data returned for {symbol} on {ex} with interval {standard_time}")
    cursor.close()
