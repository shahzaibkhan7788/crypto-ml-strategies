import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backtest.backtest import BackTest
from base import setup_trainer_directory
from utils import store_signal_stegnth_data  # make sure this exists
from data.binance.data_downloader import data_downloader

def generate_labels(df, lookahead, threshold):
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


class DecisionTreeSignal:
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
        self.X = self.scaler.fit_transform(self.X)

    def build_model(self, max_depth=None, min_samples_leaf=1):
        self.model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    def train(self):
        self.model.fit(self.X, self.y)

    def predict(self):
        preds = self.model.predict(self.X)
        result_df = self.df.copy()
        if 'timestamp' not in result_df.columns:
            result_df = result_df.reset_index()
        result_df['aggregated_signal'] = preds
        return result_df[['timestamp', 'aggregated_signal']]


def run_trial(trial_number, df, aggregated_df):
    max_depth = int(np.random.choice(range(3, 11)))
    min_samples_leaf = int(np.random.choice(range(1, 6)))
    lookahead = int(np.random.choice(range(1, 4)))
    threshold = float(round(np.random.uniform(0.001, 0.01), 4))

    print(f"\n=== Trial {trial_number} params ===")
    print(f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, lookahead={lookahead}, threshold={threshold}")

    df_labeled = generate_labels(aggregated_df, lookahead, threshold)
    print(df_labeled[['timestamp', 'signal']].head())

    dt_model = DecisionTreeSignal(df_labeled)
    dt_model.preprocess_data()
    dt_model.build_model(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    dt_model.train()

    signals_df = dt_model.predict().dropna()
    print("Signals sample:", signals_df.head())

    backtest = BackTest(signals_df, df, fee=0.005, tp_threshold=5, sl_threshold=3)
    pnl_dict, pnl_df = backtest.calculate_signal_strength()

    return {
        "trial": trial_number,
        "params": {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "lookahead": lookahead,
            "threshold": threshold
        },
        "pnl_sum": pnl_dict["pnl_sum"],
        "pnl_df": pnl_df,
        "model": dt_model.model
    }


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

                for t in range(1, 41):  # 20 trials
                    results.append(run_trial(t, df, aggregated_df))

                # Select top 10
                top_trials = sorted(results, key=lambda x: x["pnl_sum"], reverse=True)[:10]

                for rank, trial in enumerate(top_trials, start=1):
                    print(f"\n=== Top {rank} Trial ===")
                    print(f"Trial {trial['trial']} pnl_sum={trial['pnl_sum']}")

                    metadata = {
                        "symbol": sym,
                        "time_horizon": time,
                        "model_name": "decision_tree_classifier",
                        "training_params": {},
                        "optuna_params": trial['params'],
                        "notes": f"Rank {rank} trial with pnl_sum={trial['pnl_sum']}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    paths = setup_trainer_directory(
                        base_path='D:/cryptodata/ml/trainer',
                        symbol=sym,
                        time_horizon=time,
                        model_name="decision_tree_classifier",
                        metadata=metadata,
                        model_object=trial["model"]
                    )

                    # Save model separately with unique name
                    model_pkl_path = os.path.join(paths['folder'], f"model_rank{rank}_trial{trial['trial']}.pkl")
                    with open(model_pkl_path, 'wb') as f:
                        pickle.dump(trial["model"], f)


                    # Append metadata
                    if os.path.exists(paths['metadata']):
                        with open(paths['metadata'], 'r') as f:
                            existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                        existing_data.append(metadata)
                    else:
                        existing_data = [metadata]
                    with open(paths['metadata'], 'w') as f:
                        json.dump(existing_data, f, indent=4)

                    # Save to DB
                    schema_name = "ml_ledger"
                    table_name = f"{exch}_{sym}_{time}_decision_tree_trial{trial['trial']}".lower()
                    store_signal_stegnth_data(schema_name, table_name, trial["pnl_df"])

                print("\n✅ Top Decision Tree trials saved & stored in DB.")
