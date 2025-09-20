import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

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
# Gradient Boosting Regression Model Class
# =========================
class GradientBoostRegOHLCV:
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.features = ['open', 'high', 'low', 'close', 'volume']
        self.scaler = StandardScaler()
        self.X, self.y = None, None

    def preprocess_data(self):
        X = self.df[self.features].values
        y = self.df['signal'].values
        self.X = self.scaler.fit_transform(X)
        self.y = y

    def build_model(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, min_samples_split=2):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_split=min_samples_split
        )

    def train(self):
        self.model.fit(self.X, self.y)

    def predict(self):
        raw_preds = self.model.predict(self.X)
        preds = np.where(raw_preds > 0, 1, np.where(raw_preds < 0, -1, 0))

        result_df = self.df.copy()
        result_df['aggregated_signal'] = preds

        if 'timestamp' not in result_df.columns:
            result_df = result_df.reset_index()

        return result_df[['timestamp', 'aggregated_signal']]


# =========================
# Trial Runner
# =========================
def run_trial(trial_number, df, aggregated_df):
    n_estimators = int(np.random.choice(range(50, 501, 50)))
    learning_rate = float(round(np.random.uniform(0.01, 0.3), 2))
    max_depth = int(np.random.choice(range(2, 11)))
    subsample = float(round(np.random.uniform(0.5, 1.0), 2))
    min_samples_split = int(np.random.choice(range(2, 11)))
    lookahead = int(np.random.choice(range(1, 4)))
    threshold = float(round(np.random.uniform(0.001, 0.01), 4))

    print(f"\n=== Trial {trial_number} ===")
    print(f"Params: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, "
          f"subsample={subsample}, min_samples_split={min_samples_split}, lookahead={lookahead}, threshold={threshold}")

    df_labeled = generate_labels(aggregated_df, lookahead, threshold)

    model = GradientBoostRegOHLCV(df_labeled)
    model.preprocess_data()
    model.build_model(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split
    )
    model.train()

    signals_df = model.predict().dropna()

    backtest = BackTest(signals_df, df, fee=0.005, tp_threshold=5, sl_threshold=3)
    pnl_dict, pnl_df = backtest.calculate_signal_strength()

    print(f"PnL Sum: {pnl_dict['pnl_sum']}")

    return pnl_dict, pnl_df, {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "min_samples_split": min_samples_split,
        "lookahead": lookahead,
        "threshold": threshold
    }, model.model


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
                        'model_name': 'gradient_boost_model',
                        'training_params': trial['params'],
                        'notes': f'Rank {rank} trial with pnl_sum={trial["pnl_sum"]}',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    paths = setup_trainer_directory(
                        base_path='D:/cryptodata/ml/trainer',
                        symbol=sym,
                        time_horizon=time,
                        model_name='gradient_boost_model',
                        metadata=metadata,
                        model_object=None
                    )

                    # Save model as pickle
                    weight_file_path = os.path.join(paths['folder'], f"model_rank{rank}_trial{trial['trial']}.pkl")
                    with open(weight_file_path, 'wb') as f:
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
                    table_name = f"{exch}_{sym}_{time}_gradient_boosting_trial{trial['trial']}".lower()
                    store_signal_stegnth_data(schema_name, table_name, trial["pnl_df"])

                print("\n✅ Top Gradient Boosting Regression trials saved with pickle weights, metadata, and DB records!")
