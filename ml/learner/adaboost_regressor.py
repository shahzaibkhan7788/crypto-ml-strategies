import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backtest.backtest import BackTest
from base import setup_trainer_directory
from utils import store_signal_stegnth_data
from data.binance.data_downloader import data_downloader


# =========================
# Helper Functions
# =========================
def generate_labels(df, lookahead, threshold):
    """
    Classification labels for regression output:
    1 if return > threshold,
    -1 if return < -threshold,
    0 otherwise.
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
# AdaBoost Model Class
# =========================
class AdaBoostSignal:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.X = None
        self.y = None
        self.scaler = MinMaxScaler()

    def preprocess_data(self):
        features = ['open', 'high', 'low', 'close', 'volume']
        scaled = self.scaler.fit_transform(self.df[features].values)
        self.X = scaled
        self.y = self.df['signal'].values  # already thresholded labels

    def build_model(self, n_estimators=50, learning_rate=1.0, max_depth=3):
        base_estimator = DecisionTreeRegressor(max_depth=max_depth)
        self.model = AdaBoostRegressor(
            estimator=base_estimator,  # changed from base_estimator
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )

    def train(self):
        self.model.fit(self.X, self.y)

    def predict(self, threshold=0.0):
        raw_preds = self.model.predict(self.X)
        preds = np.where(raw_preds > threshold, 1,
                         np.where(raw_preds < -threshold, -1, 0))

        result_df = self.df.copy()
        if 'timestamp' not in result_df.columns:
            result_df = result_df.reset_index()

        result_df['aggregated_signal'] = preds
        result_df['predicted_value'] = raw_preds
        return result_df[['timestamp', 'aggregated_signal', 'predicted_value']]


# =========================
# Trial Runner
# =========================
def run_trial(trial_number, df, aggregated_df):
    n_estimators = int(np.random.choice([50, 100, 150]))
    learning_rate = round(np.random.uniform(0.1, 1.0), 2)
    max_depth = int(np.random.choice(range(2, 6)))
    lookahead = int(np.random.choice(range(1, 4)))
    threshold = round(np.random.uniform(0.001, 0.01), 4)

    print(f"\n=== Trial {trial_number} ===")
    print(f"n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, "
          f"lookahead={lookahead}, threshold={threshold}")

    df_labeled = generate_labels(aggregated_df, lookahead, threshold)

    model = AdaBoostSignal(df_labeled)
    model.preprocess_data()
    model.build_model(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.train()

    signals_df = model.predict(threshold=0.0).dropna()

    backtest = BackTest(signals_df[['timestamp', 'aggregated_signal']], df, fee=0.005, tp_threshold=5, sl_threshold=3)
    pnl_dict, pnl_df = backtest.calculate_signal_strength()

    return {
        "trial": trial_number,
        "params": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "lookahead": lookahead,
            "threshold": threshold
        },
        "pnl_sum": pnl_dict["pnl_sum"],
        "pnl_df": pnl_df,
        "model": model.model
    }


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

                for t in range(1, 41):
                    trial_result = run_trial(t, df, aggregated_df)
                    results.append(trial_result)

                top_trials = sorted(results, key=lambda x: x["pnl_sum"], reverse=True)[:10]

                for rank, trial in enumerate(top_trials, start=1):
                    metadata = {
                        'exchange': exch,
                        'symbol': sym,
                        'time_horizon': time,
                        'model_name': 'adaboost_regressor',
                        'training_params': trial['params'],
                        'notes': f'Rank {rank} trial with pnl_sum={trial["pnl_sum"]}',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    paths = setup_trainer_directory(
                        base_path='D:/cryptodata/ml/trainer',
                        symbol=sym,
                        time_horizon=time,
                        model_name='adaboost_regressor',
                        metadata=metadata,
                        model_object=trial['model']
                    )

                    model_pkl_path = os.path.join(paths['folder'], f"model_rank{rank}_trial{trial['trial']}.pkl")
                    with open(model_pkl_path, 'wb') as f:
                        pickle.dump(trial['model'], f)

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
                    table_name = f"{exch}_{sym}_{time}_adaboost_trial{trial['trial']}".lower()
                    store_signal_stegnth_data(schema_name, table_name, trial["pnl_df"])

                print("\n✅ Top AdaBoost trials saved successfully and DB updated!")
