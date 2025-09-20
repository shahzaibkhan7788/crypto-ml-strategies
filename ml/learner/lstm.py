import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.binance.data_downloader import data_downloader
from backtest.backtest import BackTest
from base import setup_trainer_directory
from utils import store_signal_stegnth_data


# =========================
# Helper Functions
# =========================
def generate_labels(df, lookahead, threshold):
    """
    Generate signals using future return between close[t] and close[t+lookahead].
    Returns 1 if return > threshold, -1 if < -threshold, else 0.
    """
    df = df.copy()
    df['future_close'] = df['close'].shift(-lookahead)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    conditions = [
        df['future_return'] > threshold,
        df['future_return'] < -threshold
    ]
    choices = [1, -1]
    df['signal'] = np.select(conditions, choices, default=0)
    df.dropna(inplace=True)
    return df


# =========================
# LSTM Model Class
# =========================
class LSTMOHLCV:
    def __init__(self, df, lookback=60):
        self.df = df.copy()
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.model = None
        self.X, self.y = None, None

    def preprocess_data(self):
        features = ['open', 'high', 'low', 'close', 'volume']
        data = self.df[features].values
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i])
            y.append(self.df['signal'].iloc[i])  # Using generated signals as labels
        self.X, self.y = np.array(X), np.array(y)

    def build_model(self, units=50, dropout=0.2):
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True,
                       input_shape=(self.X.shape[1], self.X.shape[2])))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='tanh'))  # Output between -1 and 1
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def train(self, epochs=10, batch_size=32):
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self):
        raw_preds = self.model.predict(self.X, verbose=0).flatten()
        preds = np.where(raw_preds > 0, 1, np.where(raw_preds < 0, -1, 0))

        result_df = self.df.iloc[self.lookback:].copy()
        result_df['aggregated_signal'] = preds

        if 'timestamp' not in result_df.columns:
            result_df = result_df.reset_index()

        return result_df[['timestamp', 'aggregated_signal']]


# =========================
# Trial Runner
# =========================
def run_trial(trial_number, df, aggregated_df):
    lookback = int(np.random.choice(range(20, 71, 10)))
    units = int(np.random.choice(range(32, 129, 16)))
    dropout = float(round(np.random.uniform(0.1, 0.5), 1))
    epochs = int(np.random.choice(range(5, 31, 5)))
    batch_size = int(np.random.choice([16, 32, 64]))
    lookahead = int(np.random.choice(range(1, 4)))
    threshold = float(round(np.random.uniform(0.001, 0.01), 4))

    print(f"\n=== Trial {trial_number} ===")
    print(f"Params: lookback={lookback}, units={units}, dropout={dropout}, "
          f"epochs={epochs}, batch_size={batch_size}, lookahead={lookahead}, threshold={threshold}")

    df_labeled = generate_labels(aggregated_df, lookahead, threshold)

    lstm = LSTMOHLCV(df_labeled, lookback=lookback)
    lstm.preprocess_data()
    lstm.build_model(units=units, dropout=dropout)
    lstm.train(epochs=epochs, batch_size=batch_size)

    signals_df = lstm.predict().dropna()

    backtest = BackTest(signals_df, df, fee=0.005, tp_threshold=5, sl_threshold=3)
    pnl_dict, pnl_df = backtest.calculate_signal_strength()

    print(f"PnL Sum: {pnl_dict['pnl_sum']}")

    return pnl_dict, pnl_df, {
        "lookback": lookback,
        "units": units,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "lookahead": lookahead,
        "threshold": threshold
    }, lstm.model


# =========================
# Main Script
# =========================
if __name__ == "__main__":
    
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')

    exchanges = config.get('Data', 'exchange').split(',')
    symbols = config.get('Data', 'symbol').split(',')
    time_horizons = config.get('Data', 'time_horizon').split(',')

    exchange_list = [e.strip() for e in exchanges]
    symbol_list = [s.strip() for s in symbols]
    time_horizon_list = [t.strip() for t in time_horizons]

    for exch in exchange_list:
        for sym in symbol_list:
            for time in time_horizon_list:
                df, aggregated_df = data_downloader(exch, sym, time)
                
                
                results = []

                for t in range(1, 31):
                    pnl_dict, pnl_df, params, trained_model = run_trial(t, df, aggregated_df)
                    results.append({
                        "trial": t,
                        "params": params,
                        "pnl_sum": pnl_dict["pnl_sum"],
                        "pnl_df": pnl_df,
                        "model": trained_model
                    })

                top_trials = sorted(results, key=lambda x: x["pnl_sum"], reverse=True)[:10]

                for rank, trial in enumerate(top_trials, start=1):
                    metadata = {
                        'exchange': exch,
                        'symbol': sym,
                        'time_horizon': time,
                        'model_name': 'lstm_model',
                        'training_params': trial['params'],
                        'notes': f'Rank {rank} trial with pnl_sum={trial["pnl_sum"]}',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Build folder path manually (before calling setup_trainer_directory)
                    folder_path = os.path.join(
                        'D:/cryptodata/ml/trainer',
                        sym,
                        time,
                        'lstm_model'
                    )

                    # Expected weight file path
                    weight_file_path = os.path.join(
                        folder_path,
                        f"model_rank{rank}_trial{trial['trial']}.weights.h5"
                    )

                    # Check if weight file exists
                    if os.path.exists(weight_file_path):
                        model_object_to_pass = weight_file_path  # pass existing weights
                    else:
                        model_object_to_pass = None  # no previous weights

                    # Call setup_trainer_directory with model_object
                    paths = setup_trainer_directory(
                        base_path='D:/cryptodata/ml/trainer',
                        symbol=sym,
                        time_horizon=time,
                        model_name='lstm_model',
                        metadata=metadata,
                        model_object=model_object_to_pass
                    )

                    # Save current trial weights (overwrite or first time)
                    trial['model'].save_weights(weight_file_path)


                    if os.path.exists(paths['metadata']):
                        with open(paths['metadata'], 'r') as f:
                            existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                        existing_data.append(metadata)
                        with open(paths['metadata'], 'w') as f:
                            json.dump(existing_data, f, indent=4)
                    else:
                        with open(paths['metadata'], 'w') as f:
                            json.dump([metadata], f, indent=4)

                    schema_name = "ml_ledger"
                    table_name = f"{exch}_{sym}_{time}_lstm_trial{trial['trial']}".lower()
                    store_signal_stegnth_data(schema_name, table_name, trial["pnl_df"])

                print("\n✅ Top LSTM trials saved with weights, metadata, and DB records!")
