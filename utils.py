import pandas as pd
import re
import psycopg2
    

def table_exists(conn, table_name, exchange):
    cursor = conn.cursor()
    cursor.execute(f"""SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = %s 
        AND table_name = %s);""", (exchange, table_name))

    exists = cursor.fetchone()[0]
    cursor.close()
    return exists


def interval_validation(exchange, time_horizon):
    if isinstance(exchange, tuple):
        exchange = exchange[0]
    if not isinstance(exchange, str):
        raise TypeError("Exchange must be a string.")

    exchange = exchange.lower()

    pattern = r"^[0-9]+(min|m|h|d|w|M)$"
    # this has been modified bybit_pattern = r"^[0-9]+(min|D|W|M)?$"
    time_horizon = str(time_horizon)

    if exchange == "binance":
        if not re.match(pattern, time_horizon):
            raise ValueError("Invalid Binance interval format. Use formats like '3m', '1min', '1h', '7d', '1w', '1M'.")
    
    elif exchange == "bybit":
         if not re.match(pattern, time_horizon):
            raise ValueError("Invalid Binance interval format. Use formats like '3m', '1min', '1h', '7d', '1w', '1M'.")
    


def data_interpolation(df, Interpolation=True):
 
    has_missing = df.isna().any().any()
    
    duplicate_count = df.index.duplicated().sum()

    if has_missing and Interpolation:
        print("Missing values detected. Applying interpolation...")
        Null_rows = df[df.isna().any(axis=1)]
        print("Null rows:")
        print(Null_rows)

        df = df.interpolate(method='linear')
        remaining_na = df.isna().sum().sum()
        print(f"Remaining missing values after interpolation: {remaining_na}")

    if duplicate_count > 0:
        print(f"Duplicate timestamps detected: {duplicate_count}. Dropping duplicates...")
        df = df[~df.index.duplicated(keep='first')]
        remaining_duplicates = df.index.duplicated().sum()
        print(f"Remaining duplicated indices: {remaining_duplicates}")

    if not has_missing and duplicate_count == 0:
        print("No missing values or duplicates. Returning original dataframe.")

    return df


## change name of this function into..
## def standardize_time_horizon(exchange, time_horizon): the above is the acutal one
def normalize_timeframe(exchange, time_horizon):
    exchange = exchange.lower()
    time_horizon = str(time_horizon).strip().lower() 

    if exchange == "binance":
        if time_horizon.endswith("m") or time_horizon.endswith("min"):
            return "1m"
        elif time_horizon.endswith("h"):
            return "1h"
        elif time_horizon.endswith("d"):
            return "1d"
        elif time_horizon.endswith("w"):
            return "1w"
        elif time_horizon.endswith("m") and time_horizon != "m":  # e.g. '1M' (month)
            return "1M"

    elif exchange == "bybit":
        if time_horizon.endswith("m") or time_horizon.endswith("min") or time_horizon.endswith("h"):
            return "1"
        elif time_horizon.endswith('d'):
            return "D"
        elif time_horizon.endswith("w"):
            return "W"
        elif time_horizon.endswith("m"):
            return "M"



## no need of sending interval..so make sure to update it...
def map_to_pandas(interval: str) -> str:
    interval = str(interval)
    
    mapping = {
            'm': 'min',
            'h': 'H',
            'd': 'D',
            'w': 'W',
            'M': 'M',
        }
    num = ''.join(filter(str.isdigit, interval))
    unit = ''.join(filter(str.isalpha, interval))
    if unit not in mapping:
        raise ValueError(f"Unsupported Binance unit: {unit}")
    return f"{num}{mapping[unit]}"


import psycopg2

def store_signal_stegnth_data(schema, table_name, df):
    print("columns of df:",df.columns)
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="exchange",
            user="postgres",
            password="shah7788",
            port=5432
        )
        cursor = connection.cursor()


        # Create schema and table if they don't exist
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        connection.commit()
        print("✅ Schema created successfully.")

                # Create table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                datetime TEXT PRIMARY KEY,
                predicted_direction TEXT,
                action TEXT,
                buy_price REAL,
                sell_price REAL,
                balance REAL,
                pnl REAL,
                pnl_sum REAL
            );
        """)
        connection.commit()
        print("✅ Table created successfully.",table_name)

        # Insert query (corrected syntax)
        insert_query = f"""
            INSERT INTO {schema}.{table_name}
            (datetime, predicted_direction, action, buy_price, sell_price, balance, pnl, pnl_sum)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (datetime) DO NOTHING;
        """

        # Data to insert
        data_to_insert = [
            (
                str(row['datetime']),
                row['predicted_direction'],
                row['action'],
                row['buy_price'],
                row['sell_price'],
                row['balance'],
                row['pnl'],
                row['pnl_sum']
            )
            for _, row in df.iterrows()
        ]

        cursor.executemany(insert_query, data_to_insert)
        connection.commit()
        print(f"✅ {len(data_to_insert)} rows inserted successfully.")

    except psycopg2.Error as e:
        print("❌ PostgreSQL error:", e)

    finally:
        if connection:
            cursor.close()
            connection.close()
            print("🔒 PostgreSQL connection closed successfully.")



def convert_to_minutes(interval: str, platform: str):
    conversion_factors = {
        'h': 60,
        'd': 24 * 60,
        'w': 7 * 24 * 60,
        'month': 30 * 24 * 60
        }

    num = ''
    unit = ''
    for char in interval:
        if char.isdigit():
            num += char
        else:
            unit += char

    num = int(num) if num else 1

    # Support "mo" or "month"
    if unit in ['mo', 'month']:
        unit = 'month'

    if unit not in conversion_factors:
        raise ValueError(f"Unsupported unit '{unit}'.")

    total_minutes = num * conversion_factors[unit]
    
    # Format output
    if platform.lower() == "binance":
        return f"{total_minutes}m"
    elif platform.lower() == "bybit":
        return total_minutes
    else:
        raise ValueError("Platform must be either 'binance' or 'bybit'.")
    
    


def convert_from_minutes(value):
    # If input is a string with 'm' or 'min', strip non-digits
    if isinstance(value, str):
        value = ''.join([c for c in value if c.isdigit()])
    
    # Convert to integer
    minutes = int(value)

    if minutes % (30 * 24 * 60) == 0:
        return f"{minutes // (30 * 24 * 60)}month"
    elif minutes % (7 * 24 * 60) == 0:
        return f"{minutes // (7 * 24 * 60)}w"
    elif minutes % (24 * 60) == 0:
        return f"{minutes // (24 * 60)}d"
    elif minutes % 60 == 0:
        return f"{minutes // 60}h"
    else:
        return f"{minutes}m"



#this function will be used for scripting
def extract_categories_of_true_indicators(true_technical_indicators):
    from signalgenerator.technical_indicator_signal.signal_generator import config_reader
    
    # Load from config
    data, indicators_categories, _ = config_reader()
    
    # Extract just the names of true indicators
    true_indicators = list(true_technical_indicators.keys())

    # Filter categories for true indicators
    true_indicators_with_categories = {
        ind: indicators_categories.get(ind, 'unknown') for ind in true_indicators
    }


    return true_indicators, true_indicators_with_categories

    


def extract_true_indicators_by_category(flat_indicators):
    from signalgenerator.technical_indicator_signal.signal_generator import config_reader
    true_technical_indicators = [k for k, v in flat_indicators.items() if v]
    
    data, indicators_categories, true_indicators= config_reader()
    
    print("✅ Indicator config loaded successfully")

    updated_dict = {
        k: indicators_categories[k]
        for k, v in flat_indicators.items()
        if v and k in indicators_categories}
    
    return true_technical_indicators, updated_dict
    


def indicator_range_config(trial,active_indicators):
    indicator_config = {
        # Trend Indicators
        'sma': {
                    'window1': trial.suggest_int('sma_window1', 5, 40),
                    'window2': trial.suggest_int('sma_window2', 40, 100)
                },
        
        'ema': {
            'window1': trial.suggest_int('ema_window1', 5, 40),
            'window2': trial.suggest_int('ema_window2', 30, 80)
                },

        'macd': {
            'fastperiod': trial.suggest_int('macd_fastperiod', 5, 20),
            'slowperiod': trial.suggest_int('macd_slowperiod', 20, 50),
            'signalperiod': trial.suggest_int('macd_signalperiod', 5, 15)
                },
        
        'adx': {'window': trial.suggest_int('adx_window', 10, 30)},
        'sar': {
            'acceleration': trial.suggest_float('sar_acceleration', 0.01, 0.05),
            'maximum': trial.suggest_float('sar_maximum', 0.1, 0.5)
        },

        # Volume Indicators
        'obv': {'window': trial.suggest_int('obv_window', 10, 30)},
        'ad': {'window': trial.suggest_int('ad_window', 5, 20)},
        'adosc': {
            'fastperiod': trial.suggest_int('adosc_fastperiod', 2, 10),
            'slowperiod': trial.suggest_int('adosc_slowperiod', 10, 30)
                 },
        'mfi': {'window': trial.suggest_int('mfi_window', 10, 30)},
        'custom_volume': {'window': trial.suggest_int('custom_volume_window', 10, 40)},

        # Momentum Indicators
        'rsi': {'window': trial.suggest_int('rsi_window', 7, 30)},
        'stoch': {
            'fastk_period': trial.suggest_int('stoch_fastk', 7, 21),
            'slowk_period': trial.suggest_int('stoch_slowk', 2, 5),
            'slowk_matype': trial.suggest_int('stoch_slowk_matype', 0, 4),
            'slowd_period': trial.suggest_int('stoch_slowd', 2, 5),
            'slowd_matype': trial.suggest_int('stoch_slowd_matype', 0, 4)
                 },
        'cci': {'window': trial.suggest_int('cci_window', 10, 30)},
        'willr': {'window': trial.suggest_int('willr_window', 10, 30)},
        'mom': {'window': trial.suggest_int('mom_window', 5, 20)},
        'bop': {},  # No parameters
        'cmo': {'window': trial.suggest_int('cmo_window', 10, 30)},
        'dx': {'window': trial.suggest_int('dx_window', 10, 30)},
        'macdext': {
            'fastperiod': trial.suggest_int('macdext_fast', 5, 20),
            'fastmatype': trial.suggest_int('macdext_fast_matype', 0, 4),
            'slowperiod': trial.suggest_int('macdext_slow', 20, 50),
            'slowmatype': trial.suggest_int('macdext_slow_matype', 0, 4),
            'singalperiod': trial.suggest_int('macdext_signal', 5, 15),
            'singalmatype': trial.suggest_int('macdext_signal_matype', 0, 4)
                   },
        'macdfix': {'window': trial.suggest_int('macdfix_window', 7, 15)},
        'minus_di': {'window': trial.suggest_int('minus_di_window', 10, 30)},
        'minus_dm': {'window': trial.suggest_int('minus_dm_window', 10, 30)},
        'plus_di': {'window': trial.suggest_int('plus_di_window', 10, 30)},
        'plus_dm': {'window': trial.suggest_int('plus_dm_window', 10, 30)},
        'ppo': {
            'fastperiod': trial.suggest_int('ppo_fast', 5, 20),
            'slowperiod': trial.suggest_int('ppo_slow', 20, 50),
            'matype': trial.suggest_int('ppo_matype', 0, 4)
               },
        'roc': {'window': trial.suggest_int('roc_window', 5, 20)},
        'rocp': {'window': trial.suggest_int('rocp_window', 5, 20)},
        'rocr': {'window': trial.suggest_int('rocr_window', 5, 20)},
        'rocr100': {'window': trial.suggest_int('rocr100_window', 5, 20)},
        'stochf': {
            'fastk_period': trial.suggest_int('stochf_fastk', 7, 21),
            'fastd_period': trial.suggest_int('stochf_fastd', 2, 5),
            'fastd_matype': trial.suggest_int('stochf_fastd_matype', 0, 4)
        },
        'stochrsi': {
            'timeperiod': trial.suggest_int('stochrsi_time', 10, 30),
            'fastk_period': trial.suggest_int('stochrsi_fastk', 2, 5),
            'fastd_period': trial.suggest_int('stochrsi_fastd', 2, 5),
            'fastd_matype': trial.suggest_int('stochrsi_fastd_matype', 0, 4)
        },
        'trix': {'window': trial.suggest_int('trix_window', 10, 30)},
        'ultosc': {
            'timperiod1': trial.suggest_int('ultosc_p1', 5, 10),
            'timeperiod2': trial.suggest_int('ultosc_p2', 10, 20),
            'timeperiod3': trial.suggest_int('ultosc_p3', 20, 40)
                   },

        # Volatility Indicators
        'atr': {'window': trial.suggest_int('atr_window', 10, 30)},
        'natr': {'window': trial.suggest_int('natr_window', 10, 30)},
        'trange': {'window': trial.suggest_int('trange_window', 10, 30)},
        'bbands': {
            'timeperiod': trial.suggest_int('bbands_timeperiod', 10, 30),
            'nbdevup': trial.suggest_float('bbands_nbdevup', 1.5, 3.0),
            'nbdevdn': trial.suggest_float('bbands_nbdevdn', 1.5, 3.0),
            'matype': trial.suggest_int('bbands_matype', 0, 4)
                   },
        'stddev': {'window': trial.suggest_int('stddev_window', 10, 30)}
    }
    
     # Filter only the indicators that are active
    filtered_config = {
        key: value for key, value in indicator_config.items() if key in active_indicators
    }
    
    return filtered_config, indicator_config
