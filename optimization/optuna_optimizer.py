import optuna
import sys
import os
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from objective_function import OptunaObjective
from utils import indicator_range_config, convert_to_minutes, convert_from_minutes
from data.binance.data_downloader import data_downloader
from strategies.strategy_pipeline.generator import StrategyGenerator, config_reader


class OptunaOptimizer:
    def __init__(self, objective_class, true_indicators, study_name="strategy_optimization", direction="maximize", n_trials=100, save_dir="results"):
        self.objective_class = objective_class
        self.true_indicators = true_indicators
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.save_dir = save_dir
     
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def optimize_strategies(self):
        
        self.study = optuna.create_study(direction=self.direction, study_name=self.study_name)
        self.study.optimize(self.objective_class, n_trials=self.n_trials)
        
        filtered_params, config_paramters = indicator_range_config(self.study.best_trial, self.true_indicators)
        
        return filtered_params,self.study.best_trial

    def save_best_strategy_params(self, best_params):
        best_params_path = os.path.join(self.save_dir, f"{self.study_name}_best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=4)
       
    def save_all_trials(self):
        trial_data = [{
            **trial.params,
            "value": trial.value
        } for trial in self.study.trials]

        df = pd.DataFrame(trial_data)
        all_trials_path = os.path.join(self.save_dir, f"{self.study_name}_all_trials.csv")
        df.to_csv(all_trials_path, index=False)
        print(f"✅ All trial results saved to {all_trials_path}")


if __name__ == "__main__":
    strategy_names = []
    data,limit,interpolation=  config_reader()
    reader= StrategyGenerator(data,limit,interpolation)
    exchange_list,symbol_list,time_horizon_list= reader.parse_strategy_input_data()
    
    for i in range(int(limit)):
        for exch in exchange_list:
            for sym in symbol_list:
                for time in time_horizon_list:
                    standard_time= convert_to_minutes(time, exch)
                    best_time= convert_from_minutes(standard_time)
                   
                    strategy_name = f"strategy_{len(strategy_names):02d}"
                    strategy_names.append(strategy_name)
                    print("strategy executed right now:",strategy_name)
                    
                    df, resample_df = data_downloader(exch, sym, best_time)
                    
                    objective = OptunaObjective(base_data=resample_df,exchange=exch,symbol=sym,
                                time_horizon=best_time,strategy_name=strategy_name)
                    optimizer = OptunaOptimizer(objective, objective.true_indicators)
    
                    best_params,best_trial = optimizer.optimize_strategies()
                    objective.save_to_db(best_params,best_trial)
                    
                    optimizer.save_best_strategy_params(best_params)

                    optimizer.save_all_trials()

