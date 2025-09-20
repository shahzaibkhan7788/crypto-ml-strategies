import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indicator.indicator_calculator import technical_indicator
from utils import extract_true_indicators_by_category,indicator_range_config
from backtest.backtest import BackTest

class OptunaObjective:
    def __init__(self,base_data, fee=0.05, tp_threshold=5.0, sl_threshold=3.0, target_metric="pnl_sum",exchange=None,symbol=None,time_horizon=None, strategy_name=None):
        from strategies.strategy_pipeline.utils.indicator_utils import technical_indicators_dict_random, flatten_indicators
        self.base_data= base_data
        self.fee = fee
        self.tp_threshold = tp_threshold
        self.sl_threshold = sl_threshold
        self.target_metric = target_metric
        self.indicators_categories = technical_indicators_dict_random()
        self.flattened_indicators = flatten_indicators(self.indicators_categories)
        self.true_indicators, self.updated_dict= extract_true_indicators_by_category(self.flattened_indicators)
        self.exchange= exchange
        self.symbol= symbol
        self.time_horizon= time_horizon
        self.strategy_name= strategy_name
        self.pnl_sum_threshold=100
        self.trial_aggregated_map = {}
        self.trial_backtest_map = {}
        self.trial_pnl_map= {}

    def build_strategy_from_trial(self, trial):
        print("build strategy called for trial:",trial.number)
        from signalgenerator.technical_indicator_signal.signal_generator import SignalGenerator
        from strategies.strategy_pipeline.utils.indicator_utils import aggregated_dataframe
        
        self.filtered_indicators,self.indicators_config= indicator_range_config(trial,self.true_indicators)
        indicator = technical_indicator(self.base_data, self.updated_dict, self.true_indicators, 
                                        self.filtered_indicators)
        indicators_dict = indicator.calculate_indicators()
        signalgenerator = SignalGenerator(indicators_dict)
        signal_generator_df = signalgenerator.generate_singal()
        print("signal generated well for trial",trial.number,signal_generator_df.columns)
        self.aggregated_df= aggregated_dataframe(signal_generator_df)
        print("aggregated df last five rows:",self.aggregated_df.tail(5))
   
    
    def run_backtest(self, signals_df):
        self.backtester = BackTest(signals_df, self.base_data, self.fee, self.tp_threshold, self.sl_threshold)
        results,self.backtest_df = self.backtester.calculate_signal_strength()
        print("backtest last five rows:",self.backtest_df.tail(5))
        print("here is the column of result:",results)
        return results
    
    
    
    def save_to_db(self,best_params,best_trail):
        from data.binance.crypto_dataloader import StrategyHandling
        
        
        best_trial_number = best_trail.number
        best_pnl_sum = self.trial_pnl_map.get(best_trial_number, -999)
        best_aggregated_df = self.trial_aggregated_map[best_trial_number]
        best_backtest_df= self.trial_backtest_map[best_trial_number]

        if best_pnl_sum >= self.pnl_sum_threshold:
            print("✅ PNL passed threshold. Saving to DB.")
            backtest_table_name= f"backtest_{self.strategy_name}".lower()
            self.backtester.schema(best_backtest_df,backtest_table_name)
        
            print("symbol name passed to strategy handling is..is there any digit or not:",self.symbol)
            strategy_handling= StrategyHandling(self.strategy_name,self.exchange,self.symbol,
                                                self.time_horizon,best_aggregated_df)

            print("here is my flatten indicators:",self.flattened_indicators)
            print("here is my best params indicators:",best_params)
            print("here is my indicaot config::",self.indicators_config)
            strategy_handling.store_strategy_metadata_and_signal(self.flattened_indicators,best_params,self.indicators_config)
        else:
            print("❌ PNL below threshold. Strategy not saved to DB.")

        
    def __call__(self, trial):
        print(f"Trial number call {trial.number} for:{self.exchange}_{self.symbol}_{self.time_horizon}")
        try:
            self.build_strategy_from_trial(trial)
            self.trial_aggregated_map[trial.number] = self.aggregated_df.copy()
            
            results = self.run_backtest(self.aggregated_df)
            self.trial_backtest_map[trial.number] = self.backtest_df.copy()
            
            # Store pnl_sum directly with the trial number
            pnl_sum = results.get("pnl_sum", -999)
            print("PNL Sum for trial", trial.number, ":", pnl_sum)
            self.trial_pnl_map = getattr(self, 'trial_pnl_map', {})
            self.trial_pnl_map[trial.number] = pnl_sum

            return pnl_sum  # optimize pnl_sum directly
        except Exception as e:
            print(f"⚠️ Trial failed due to: {e}")
            return -999
