import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
from pybit.unified_trading import HTTP
import re
import os
import psycopg2
from psycopg2.extras import execute_values
import json
import time
from binance.exceptions import BinanceAPIException, BinanceRequestException


from strategies.strategy_pipeline.utils.indicator_utils import technical_indicators_dict_random, flatten_indicators
from strategies.strategy_pipeline.utils.indicator_utils import compute_signals
from utils import map_to_pandas


class crypto:
    
    def __init__(self, exchange, symbol, time_horizon, start_time=None, end_time=datetime.now().strftime('%Y-%m-%d')):
        self.exchange = exchange
        self.symbol = f"{symbol}usdt".upper()
        self.time_horizon = time_horizon
        self.start_time = start_time
        self.end_time = end_time

        self.table_name= f"{self.exchange}_{self.symbol}_1min".lower()

    def fetch_binance_data(self):
        client = Client(api_key='ACrNpQI9BAWGaEcA9z45Ny5rvbhNPzczrAGO73YO9iWz65cFsj4zFqLSol8xCQfZ',
                        api_secret='S5fbz1T8CSQEGM4Zriwc5WA6QspECypwkyVyn3wd269q5ICxqPfdErjXkOkt6WJZ')

        symbol = self.symbol
        interval = self.time_horizon  
        limit = 1000

        start_ms = int(pd.to_datetime(self.start_time, utc=True).timestamp() * 1000)
        end_ms = int(pd.to_datetime(self.end_time, utc=True).timestamp() * 1000)

        start_dt = pd.to_datetime(start_ms, unit='ms')
        end_dt = pd.to_datetime(end_ms, unit='ms') if end_ms else "∞"
        print(f"[DEBUG] Requesting Binance {interval} klines from {start_dt} to {end_dt}")

        all_data = []
        total_calls = 0
        t0 = time.time()

        while True:
            print(f"[DEBUG] API CALL #{total_calls + 1} | Start: {pd.to_datetime(start_ms, unit='ms')}")
            total_calls += 1

            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ms,
                endTime=end_ms,
                limit=limit
            )

            fetched = len(klines)
            print(f"[DEBUG] API returned {fetched} bars")

            if not klines:
                print("[DEBUG] No data returned, stopping.")
                break

            klines.sort(key=lambda x: int(x[0]))
            all_data.extend(klines)

            first_ts = int(klines[0][0])
            last_ts = int(klines[-1][0])
            print(f"[DEBUG] This batch covers: {pd.to_datetime(first_ts, unit='ms')} → {pd.to_datetime(last_ts, unit='ms')}")

            if end_ms and last_ts >= end_ms:
                if any(int(kline[0]) == end_ms for kline in klines):
                    print(f"[DEBUG] Final batch includes exact end timestamp: {pd.to_datetime(end_ms, unit='ms')}")
                print(f"[DEBUG] Reached or passed end_date: {pd.to_datetime(last_ts, unit='ms')} >= {pd.to_datetime(end_ms, unit='ms')}")
                break

            start_ms = last_ts + 60_000

            if fetched < limit:
                print("[DEBUG] Fetched fewer than limit, assuming no more data.")
                break

            print(f"[DEBUG] Total bars fetched so far: {len(all_data)} in {total_calls} calls")

        if not all_data:
            print("[DEBUG] No data fetched.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"].astype('int64'), unit='ms')
        df = df.astype({
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float
        })

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        print(f"[DEBUG] FINAL RANGE FETCHED: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print(f"[DEBUG] TOTAL BARS: {len(df)} | TOTAL TIME: {round(time.time() - t0, 2)}s | API CALLS: {total_calls}")

        return df

    def fetch_bybit_data(self):
 
        bybit_client = HTTP(
            api_key='8mgPMyFUprIKuBjNtI',
            api_secret='p0czyRpGkRmHDXdGQi4ufRlNdq6Tt5OrNpgi',
            timeout=30
        )

        limit = 1000
        start_ms = int(pd.to_datetime(self.start_time, utc=True).timestamp() * 1000)
        end_ms = int(pd.to_datetime(self.end_time, utc=True).timestamp() * 1000)

        start_dt = pd.to_datetime(start_ms, unit='ms')
        end_dt = pd.to_datetime(end_ms, unit='ms') if end_ms is not None else "∞"
        print(f"[DEBUG] Requesting 1-min klines from {start_dt} to {end_dt}")

        all_data = []
        total_calls = 0
        t0 = time.time()

        while True:
            print(f"[DEBUG] API CALL #{total_calls + 1} | Start: {pd.to_datetime(start_ms, unit='ms')}")
            total_calls += 1

            resp = bybit_client.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.time_horizon,
                start=start_ms,
                limit=limit
            )

            if not resp or resp.get('retCode') != 0:
                print(f"[ERROR] API response bad: {resp}")
                return pd.DataFrame()

            klines = resp['result']['list']
            fetched = len(klines)
            print(f"[DEBUG] API returned {fetched} bars")

            if not klines:
                print("[DEBUG] No data returned, stopping.")
                break

            klines.sort(key=lambda x: int(x[0]))
            all_data.extend(klines)

            first_ts = int(klines[0][0])
            last_ts = int(klines[-1][0])
            print(f"[DEBUG] This batch covers: {pd.to_datetime(first_ts, unit='ms')} → {pd.to_datetime(last_ts, unit='ms')}")

            # Stop if we’ve reached or passed the end time
            if end_ms and last_ts >= end_ms:
                if any(int(kline[0]) == end_ms for kline in klines):
                    print(f"[DEBUG] Final batch includes exact end timestamp: {pd.to_datetime(end_ms, unit='ms')}")
                print(f"[DEBUG] Reached or passed end_date: {pd.to_datetime(last_ts, unit='ms')} >= {pd.to_datetime(end_ms, unit='ms')}")
                break

            start_ms = last_ts + 60_000  # move to next minute

            if fetched < limit:
                print("[DEBUG] Fetched fewer than limit, assuming no more data.")
                break

            print(f"[DEBUG] Total bars fetched so far: {len(all_data)} in {total_calls} calls")

        if not all_data:
            print("[DEBUG] No data fetched.")
            return pd.DataFrame()

        # Final filtering: include bars <= end_ms
        all_data = [bar for bar in all_data if int(bar[0]) <= end_ms and int(bar[0]) < 10**13]
        print(f"[DEBUG] After filtering, bars count: {len(all_data)}")

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype('int64'), unit='ms')
        df = df.astype({
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "turnover": float
        })

        print(f"[DEBUG] FINAL RANGE FETCHED: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print(f"[DEBUG] TOTAL BARS: {len(df)} | TOTAL TIME: {round(time.time() - t0, 2)}s | API CALLS: {total_calls}")

        return df

    def store_ohlcv_data(self, df=None,return_latest=False):
        try:
            connection = psycopg2.connect(
                host="localhost",
                database="exchange",
                user="postgres",
                password="shah7788",
                port=5432
            )
            cursor = connection.cursor()

           
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.exchange};")
            connection.commit()

            # Create table if not exists
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.exchange}.{self.table_name} (
                    timestamp TEXT PRIMARY KEY,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume FLOAT
                );
            """)
            connection.commit()

            # Insert rows with ON CONFLICT DO NOTHING to avoid duplicates
            for _, row in df.iterrows():
                cursor.execute(f"""
                    INSERT INTO {self.exchange}.{self.table_name} (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO NOTHING;
                """, (
                    row['timestamp'],
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                ))

            connection.commit()
            print(f"[INFO] Data inserted successfully into {self.exchange}.{self.table_name}")
            
            if return_latest:
                cursor.execute(f"SELECT MAX(timestamp) FROM {self.exchange}.{self.table_name};")
                latest_time = cursor.fetchone()[0]
                return latest_time

        except Exception as e:
            print(f"[ERROR] Failed to store data: {e}")

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("[INFO] PostgreSQL connection closed")
                
                
    def resample_ohlcv(self, df, exchange, interval):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        resample_rule = map_to_pandas(interval)
        print("Resample rule:",resample_rule)

        df_resampled = df.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        
        df_resampled.reset_index(inplace=True)

        return df_resampled




class StrategyHandling:
    def __init__(self,strategy_name,exchange,symbol,time_horizon,aggregated_df=None):
        self.exchange= exchange
        self.symbol= symbol
        print("name of a symbol in strategy handling init:",self.symbol)
        #here print the symbol......is it 
        self.time_horizon= time_horizon
        self.strategy_name= strategy_name
        self.aggregated_df= aggregated_df
        print("aggregated df columns in strategy handling",aggregated_df.columns) 
        if 'datetime' in aggregated_df.columns and 'timestamp' not in aggregated_df.columns:
            aggregated_df.rename(columns={'datetime': 'timestamp'}, inplace=True)

        print("strategy handling:",self.exchange,self.symbol,self.time_horizon,self.strategy_name)
    
    
    def store_strategy_metadata_and_signal(self,flat_indicators, best_params, indicator_config):
        print("here is the aggregated_df:",self.aggregated_df)
        
        print("store meta data and signal calculated")
        try:
            connection = psycopg2.connect(
                host="localhost",
                database="exchange",
                user="postgres",
                password="shah7788",
                port=5432
            )
            cursor = connection.cursor()

            if best_params is None:
                best_params = {}

            if indicator_config is None:
                indicator_config = {}

            # ========== ✅ STEP 1–5: Construct extended_indicators ==========
            extended_indicators = {}

            for indicator, is_active in flat_indicators.items():
                # Step 1: Include flatten indicators as boolean
                extended_indicators[indicator] = is_active

                # Step 2: In flatten but not in config → *_period = 1
                if is_active and indicator not in indicator_config:
                    extended_indicators[f"{indicator}_period"] = 1

                # Step 3: In best_params → use best values
                elif is_active and indicator in best_params:
                    params = best_params[indicator]
                    if isinstance(params, dict):
                        for param_name, value in params.items():
                            col_name = f"{indicator}_{param_name}"
                            extended_indicators[col_name] = value

                # Step 4: In flatten and config, but not in best_params → use 0s
                elif is_active and indicator in indicator_config:
                    config_params = indicator_config[indicator]
                    if isinstance(config_params, dict):
                        for param_name in config_params:
                            col_name = f"{indicator}_{param_name}"
                            extended_indicators[col_name] = 0

                # ✅ Step 5: If indicator is inactive → assign 0 to all config params
                elif not is_active and indicator in indicator_config:
                    config_params = indicator_config[indicator]
                    if isinstance(config_params, dict):
                        for param_name in config_params:
                            col_name = f"{indicator}_{param_name}"
                            extended_indicators[col_name] = 0

            # ========== ✅ STEP 6: Ensure Table Exists BEFORE Altering =======ss===
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.strategies_indicators (
                    strategy_name TEXT PRIMARY KEY,
                    exchange TEXT,
                    symbol TEXT,
                    time_horizon TEXT
                );
            """)
            connection.commit()

            # ========== 🔧 Add Missing Columns ==========
            def add_missing_columns(cursor, table_name, columns):
                cursor.execute(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = '{table_name}' AND table_schema = 'public';
                """)
                existing_columns = set(row[0] for row in cursor.fetchall())
                for col, val in columns.items():
                    if col not in existing_columns:
                        if isinstance(val, bool):
                            pg_type = "BOOLEAN"
                        elif isinstance(val, int):
                            pg_type = "INTEGER"
                        elif isinstance(val, float):
                            pg_type = "REAL"
                        else:
                            pg_type = "TEXT"
                        alter_stmt = f"ALTER TABLE public.{table_name} ADD COLUMN {col} {pg_type};"
                        print("[INFO] Adding missing column:", alter_stmt)
                        cursor.execute(alter_stmt)

            add_missing_columns(cursor, "strategies_indicators", extended_indicators)
            connection.commit()

            # ========== ✅ Insert Metadata ==========
            base_columns = ['strategy_name', 'exchange', 'symbol', 'time_horizon']
            all_columns = base_columns + list(extended_indicators.keys())
            placeholders = ', '.join(['%s'] * len(all_columns))
            columns_str = ', '.join(all_columns)
            values = [
                self.strategy_name,
                self.exchange,
                self.symbol,
                self.time_horizon
            ] + [extended_indicators[col] for col in all_columns[4:]]

            insert_query = f"""
                INSERT INTO public.strategies_indicators ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT (strategy_name) DO NOTHING;
            """
            cursor.execute(insert_query, values)
            connection.commit()

            # ✅ Replace all NULL values in the table with 0
            # ✅ Replace all NULL values (only for non-boolean columns) with 0
            
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'strategies_indicators' 
                AND table_schema = 'public'
                AND data_type != 'boolean';
            """)
            non_boolean_columns = [row[0] for row in cursor.fetchall() if row[0] in extended_indicators]

            if non_boolean_columns:
                update_stmt = f"""
                    UPDATE public.strategies_indicators
                    SET {', '.join([f"{col} = COALESCE({col}, 0)" for col in non_boolean_columns])}
                    WHERE strategy_name = %s;
                """
                cursor.execute(update_stmt, (self.strategy_name,))
                connection.commit()


            print("[INFO] Metadata inserted for strategy:", self.strategy_name)
            
            if self.aggregated_df is None:
                self.aggregated_df = compute_signals(flat_indicators, self.exchange, self.symbol, self.time_horizon)
                print("after executing compute signals in strategy handling", len(self.aggregated_df))
            
            print("before inserting data into signal schema, the dataframe is:",self.aggregated_df)
            # Ensure table exists before inserting
           
            cursor.execute(f"""
                CREATE SCHEMA IF NOT EXISTS signal;
                CREATE TABLE IF NOT EXISTS signal.{self.strategy_name} (
                    timestamp TIMESTAMP PRIMARY KEY,
                    aggregated_signal INTEGER
                );
            """)
            connection.commit()

            # Insert rows
            for _, row in self.aggregated_df.iterrows():
                cursor.execute(f"""
                    INSERT INTO signal.{self.strategy_name} (timestamp, aggregated_signal)
                    VALUES (%s, %s)
                    ON CONFLICT (timestamp) DO NOTHING;
                """, (row['timestamp'], row['aggregated_signal']))
            connection.commit()


            print(f"[INFO] Strategy and signals stored successfully for: {self.strategy_name}")

        except psycopg2.Error as e:
            print("[ERROR] PostgreSQL Error:", e)

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("[INFO] PostgreSQL connection closed")
                
    
    
    def store_strategy_signal(self):
        
        print("store meta data and signal calculated")
        try:
            connection = psycopg2.connect(
                host="localhost",
                database="exchange",
                user="postgres",
                password="shah7788",
                port=5432
            )
            cursor = connection.cursor()

    
            technical_indicators = technical_indicators_dict_random()
            flat_indicators = flatten_indicators(technical_indicators)



            # ========== ✅ STEP 6: Ensure Table Exists BEFORE Altering =======ss===
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.strategies_indicators (
                    strategy_name TEXT PRIMARY KEY,
                    exchange TEXT,
                    symbol TEXT,
                    time_horizon TEXT
                );
            """)
            connection.commit()


            # ========== ✅ Insert Metadata ==========
            base_columns = ['strategy_name', 'exchange', 'symbol', 'time_horizon']
            all_columns = base_columns + list(flat_indicators.keys())
            placeholders = ', '.join(['%s'] * len(all_columns))
            columns_str = ', '.join(all_columns)
            
            values = [
                self.strategy_name,
                self.exchange,
                self.symbol,
                self.time_horizon
            ] + list(flat_indicators.values())


            insert_query = f"""
                INSERT INTO public.strategies_indicators ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT (strategy_name) DO NOTHING;
            """
            cursor.execute(insert_query, values)
            connection.commit()

            print("[INFO] Metadata inserted for strategy:", self.strategy_name)

            # ========== ✅ SIGNAL TABLE SETUP ==========
            cursor.execute("CREATE SCHEMA IF NOT EXISTS signal;")
            connection.commit()

            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS signal.{self.strategy_name} (
                    timestamp TEXT PRIMARY KEY,
                    aggregated_signal INTEGER
                );
            """)
            connection.commit()

            # ========== ✅ COMPUTE AND STORE SIGNALS ==========
            if self.aggregated_df is None:
                self.aggregated_df = compute_signals(flat_indicators, self.exchange, self.symbol, self.time_horizon)
                print("after executing compute signals in strategy handling", len(self.aggregated_df))

            for _, row in self.aggregated_df.iterrows():
                cursor.execute(f"""
                    INSERT INTO signal.{self.strategy_name} (timestamp, aggregated_signal)
                    VALUES (%s, %s)
                    ON CONFLICT (timestamp) DO NOTHING;
                """, (row['timestamp'], row['aggregated_signal']))
            connection.commit()

            print(f"[INFO] Strategy and signals stored successfully for: {self.strategy_name}")

        except psycopg2.Error as e:
            print("[ERROR] PostgreSQL Error:", e)

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("[INFO] PostgreSQL connection closed")