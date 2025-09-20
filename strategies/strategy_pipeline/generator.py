import os
from configparser import ConfigParser
from datetime import datetime
from utility.config_loader import load_config
config = load_config()

from utils import convert_to_minutes, convert_from_minutes

def config_reader():
    data={}
    limit={}
    general={}

    for key in config['Data']:
        value= config['Data'][key]
        value_list= [v.strip() for v in value.split(',')]
        data[key]=value_list
      
    limit= config['Limits'].get("max_strategies")
    interpolation = config['Interpolation'].getboolean('interpolation')

    return data,limit,interpolation


class StrategyGenerator:
    def __init__(self, data=None, limit=None,interpolation=True):
        self.data = data  
        self.limit = limit  
        self.interpolation= interpolation
        print("Strategy generator being called",self.data,self.data['exchange'])
        
    
    def parse_strategy_input_data(self):
        exchange = self.data['exchange']
        print("the exchange is:",exchange)      

        symbols = self.data['symbol']     
        time_horizon = self.data['time_horizon']    
        end_time = self.data['end_time'][0] 

        if end_time == 'now':
            end_time = datetime.now().strftime('%Y-%m-%d')

        self.symbol_list = [s.strip() for s in symbols]
        self.exchange_list = [s.strip().lower() for s in exchange]
        self.time_horizon_list= [s.strip().lower() for s in time_horizon]
        
        return self.exchange_list,self.symbol_list,self.time_horizon_list
       
    def generate_random_strategies(self):
        strategy_names = ['strategy_00','strategy_01','strategy_02','strategy_03','strategy_04','strategy_05',
                          'strategy_06','strategy_07','strategy_08','strategy_09','strategy_10','strategy_11']
        strategy_objects = {}
 
        for i in range(int(self.limit)):
            for exch in self.exchange_list:
                for sym in self.symbol_list:
                    for time in self.time_horizon_list:
                        standard_time= convert_to_minutes(time, exch)
                        best_time= convert_from_minutes(standard_time)
                        strategy_name = f"strategy_{len(strategy_names):02d}"
                        strategy_names.append(strategy_name)
                        print("strategy executed right now:",strategy_name)
                        print("time that are exeuted:",best_time)
                        if exch == "binance":
                            crypto_data = StrategyHandling(strategy_name,
                                                        exchange=exch,
                                                        symbol=sym,
                                                        time_horizon=best_time)
                            print("symbol name in generator.py:",sym)
                            
                        if exch == "bybit":
                            crypto_data = StrategyHandling(strategy_name,
                            exchange=exch,
                            symbol=sym,
                            time_horizon=best_time)
                            print("symbol name in generator.py:",sym)
                        else:
                            print(f"No data returned for {sym} at {time}")

                        strategy_objects[strategy_name] = crypto_data
                        crypto_data.store_strategy_signal()
        return strategy_names, strategy_objects
    
    

        
if __name__ == "__main__":
    from data.binance.crypto_dataloader import StrategyHandling
    data,limit,interpolation= config_reader()
    print("now init function will call")
    generator= StrategyGenerator(data,limit,interpolation)
    print("now random strategc calll")
    exchange_list,symbol_list,time_horizon_list=generator.parse_strategy_input_data()
    strategy_names, strategy_objects= generator.generate_random_strategies()






