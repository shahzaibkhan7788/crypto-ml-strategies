import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.binance.data_downloader import data_downloader
from backtest.backtest import BackTest
from base import setup_trainer_directory
from utils import store_signal_stegnth_data  # adjust path to where your function is

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


class RandomForestSignal:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.X = None
        self.y = None
        self.scaler = MinMaxScaler()

    def preprocess_data(self):
        features = ['open', 'high', 'low', 'close', 'volume']
        self.X = self.df[features].values
        self.y = self.df['signal'].values
        # Scaling not strictly needed for RF
        # self.X = self.scaler.fit_transform(self.X)

    def build_model(self, n_estimators=100, max_depth=None, min_samples_leaf=1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42
        )

    def train(self):
        self.model.fit(self.X, self.y)

    def predict(self):
        preds = self.model.predict(self.X)
        print("here are the preds:",preds)
        result_df = self.df.copy()
        if 'timestamp' not in result_df.columns:
            result_df = result_df.reset_index()
        result_df['aggregated_signal'] = preds
        return result_df[['timestamp', 'aggregated_signal']]


def run_trial(trial_number, df, aggregated_df):
    # Random hyperparameters
    n_estimators = int(np.random.choice([50, 100, 150]))
    max_depth = int(np.random.choice(range(3, 11)))
    min_samples_leaf = int(np.random.choice(range(1, 6)))
    lookahead = int(np.random.choice(range(1, 4)))
    threshold = float(round(np.random.uniform(0.001, 0.01), 4))

    print(f"\n=== Trial {trial_number} params ===")
    print(f"n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_samples_leaf={min_samples_leaf}, lookahead={lookahead}, threshold={threshold}")

    # Label data
    df_labeled = generate_labels(aggregated_df, lookahead, threshold)
    print("Signal labels sample:")
    print(df_labeled[['timestamp', 'signal']].head())

    rf_model = RandomForestSignal(df_labeled)
    rf_model.preprocess_data()
    rf_model.build_model(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    rf_model.train()

    signals_df = rf_model.predict().dropna()
    print("Backtest signals sample:")
    print(signals_df.head())

    backtest = BackTest(signals_df, df, fee=0.005, tp_threshold=5, sl_threshold=3)
    pnl_dict, pnl_df = backtest.calculate_signal_strength()

    print(f"Trial {trial_number} pnl_sum: {pnl_dict['pnl_sum']}")
    return pnl_dict, pnl_df, {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "lookahead": lookahead,
        "threshold": threshold
    }, rf_model.model


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
                num_trials = 41

                for t in range(1, num_trials + 1):
                    pnl_dict, pnl_df, params, trained_model = run_trial(t, df, aggregated_df)
                    results.append({
                        "trial": t,
                        "params": params,
                        "pnl_sum": pnl_dict["pnl_sum"],
                        "pnl_df": pnl_df,
                        "model": trained_model
                    })

                # Top 10 by PnL
                top_trials = sorted(results, key=lambda x: x["pnl_sum"], reverse=True)[:10]

                for rank, trial in enumerate(top_trials, start=1):
                    print(f"\n=== Top {rank} Trial ===")
                    print(f"Trial: {trial['trial']}, pnl_sum: {trial['pnl_sum']}")
                    print(f"Params: {trial['params']}")

                    # Metadata for RF
                    metadata = {
                        'exchange': exch,
                        'symbol': sym,
                        'time_horizon': time,
                        'model_name': 'random_forest',
                        'training_params': trial['params'],
                        'notes': f'Rank {rank} trial with pnl_sum={trial["pnl_sum"]}',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Create directory
                    paths = setup_trainer_directory(
                        base_path='D:/cryptodata/ml/trainer',
                        symbol=sym,
                        time_horizon=time,
                        model_name='random_forest',
                        metadata=metadata,
                        model_object=trial['model']
                    )

                
                    # Append metadata to JSON
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

                    # Save signal strength to DB
                    schema_name = "ml_ledger"
                    table_name = f"{exch}_{sym}_{time}_random_forest_trial{trial['trial']}".lower()
                    store_signal_stegnth_data(schema_name, table_name, trial["pnl_df"])

                print("\n✅ Top Random Forest trials saved with metadata, model, backtest results, and DB records!")
