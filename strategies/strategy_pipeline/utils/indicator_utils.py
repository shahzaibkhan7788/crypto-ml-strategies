import configparser
import os
import random
from utils import extract_true_indicators_by_category
config = configparser.ConfigParser()

from utility.config_loader import load_config
config = load_config()


def technical_indicators_dict_random():
    indicator_config = {}

    for section in ['price', 'volatility', 'trend', 'momentum', 'cycle', 'pattern', 'volume']:
        if section in config:
            indicators = {}
            for key in config[section]:
                indicators[key] = random.choice([True, False])
                indicator_config[section] = indicators
 
    return indicator_config


def flatten_indicators(indicator_dict):
    flat_dict = {}
    for category, indicators in indicator_dict.items():
        for key, value in indicators.items():
            flat_dict[key] = value
    return flat_dict



def compute_signals(flat_indicators, exchange, symbol, time_horizon):
    print("🔄 Starting signal computation...")

    from data.binance.data_downloader import data_downloader
    from signalgenerator.technical_indicator_signal.signal_generator import SignalGenerator, config_reader
    from indicator.indicator_calculator import technical_indicator
    
    #problem occur here

    df, aggregated_df = data_downloader(exchange, symbol, time_horizon)
    print(f"📊 Raw data columns from data_downloader: {df.columns.tolist()}")

    true_technical_indicators, updated_dict=  extract_true_indicators_by_category(flat_indicators)
    indicator = technical_indicator(aggregated_df, updated_dict, true_technical_indicators)
    print("🧮 Calculating technical indicators...")

    indicators_dict = indicator.calculate_indicators()
   
    signalgenerator = SignalGenerator(indicators_dict)
    signal_generator_df = signalgenerator.generate_singal()
    print("✅ Signal generation completed")
    
    aggregated_df=aggregated_dataframe(signal_generator_df)
    
    return aggregated_df
    



def aggregated_dataframe(signal_generator_df):
    print("columns:",signal_generator_df)
    
    # Step 1: Get signal columns before renaming
    signal_cols = [col for col in signal_generator_df.columns if col.endswith('signal')]

    # Step 2: Keep a lowercase mapping
    signal_generator_df.columns = [col.lower() for col in signal_generator_df.columns]

    # Step 3: Convert signal_cols to lowercase too
    signal_cols = [col.lower() for col in signal_cols]

    # Step 4: Drop NA and round only existing columns
    signal_generator_df.dropna(inplace=True)
    existing_signal_cols = [col for col in signal_cols if col in signal_generator_df.columns]
    if existing_signal_cols:
        signal_generator_df[existing_signal_cols] = signal_generator_df[existing_signal_cols].round()
    else:
        print("⚠️ No matching signal columns found for rounding.")

    print("✅ After signal generation and drop")

    # Step 5: Aggregated signal logic
    count_1 = (signal_generator_df[existing_signal_cols] == 1.0).sum(axis=1)
    count_neg1 = (signal_generator_df[existing_signal_cols] == -1.0).sum(axis=1)

    signal_generator_df['aggregated_signal'] = 0
    signal_generator_df.loc[count_1 > count_neg1, 'aggregated_signal'] = 1
    signal_generator_df.loc[count_neg1 > count_1, 'aggregated_signal'] = -1

    print("📈 Signal aggregation completed successfully",signal_generator_df.columns)
    return signal_generator_df[['timestamp', 'aggregated_signal']]

