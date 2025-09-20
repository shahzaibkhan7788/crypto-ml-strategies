import talib
import pandas as pd 
import numpy as np 
import configparser
import re
import os

from utility.config_loader import load_config
config = load_config()

from data.binance.data_downloader import data_downloader
from indicator.indicator_calculator import technical_indicator


def config_reader():
    data=[]
    indicators_categories={}
    true_indicators={}
    
    for section in config.sections():
        if section.lower()=='data':
            for item in config['Data']:
                data.append(config['Data'][item])
        else:
            for key in config[section]:
                indicator_name = key.lower()
                indicators_categories[indicator_name] = section.lower()
                true_indicators[indicator_name]= config[section][key]
    return data,indicators_categories, true_indicators


data, indicators_categories, true_indicators= config_reader()


class SignalGenerator:
    def __init__(self,indicators_dict):
        self.cols={}
        self.df = indicators_dict['result_df']
        self.trend = indicators_dict['trend']
        self.volume = indicators_dict['volume']
        self.volatility = indicators_dict['volatility']
        self.momentum = indicators_dict['momentum']
        self.cycle = indicators_dict['cycle']
        self.patterns = indicators_dict['patterns']
        self.price = indicators_dict['price']
        
        self.df.dropna(inplace=True)
        self.df.columns = [col.lower() for col in self.df.columns]
        if 'datetime' in self.df.columns and 'timestamp' not in self.df.columns:
            self.df.rename(columns={'datetime': 'timestamp'}, inplace=True)

 
    
    def trend_signal_generator(self):
        if 'sma' in self.trend:
            sma_signal = pd.Series(0, index=self.df.index)
            bullish_sma = (self.df['sma_50'] > self.df['sma_100']) & (self.df['sma_50'].shift(1) > self.df['sma_100'].shift(1))
            bearish_sma = (self.df['sma_50'] < self.df['sma_100']) & (self.df['sma_50'].shift(1) < self.df['sma_100'].shift(1))
            sma_signal[bullish_sma] = 1
            sma_signal[bearish_sma] = -1
            self.cols['SMA_signal'] = sma_signal

        if 'ema' in self.trend:
            ema_signal = pd.Series(0, index=self.df.index)
            bullish_ema = (self.df['ema_20'] > self.df['ema_80']) & (self.df['ema_20'].shift(1) > self.df['ema_80'].shift(1))
            bearish_ema = (self.df['ema_20'] < self.df['ema_80']) & (self.df['ema_20'].shift(1) < self.df['ema_80'].shift(1))
            ema_signal[bullish_ema] = 1
            ema_signal[bearish_ema] = -1
            self.cols['EMA_signal'] = ema_signal

        if 'macd' in self.trend:
            macd_signal = pd.Series(0, index=self.df.index)
            bullish_macd = (self.df['macd'] > self.df['signal_macd']) & (self.df['macd'].shift(1) <= self.df['signal_macd'].shift(1))
            bearish_macd = (self.df['macd'] < self.df['signal_macd']) & (self.df['macd'].shift(1) >= self.df['signal_macd'].shift(1))
            macd_signal[bullish_macd] = 1
            macd_signal[bearish_macd] = -1
            self.cols['MACD_signal'] = macd_signal

        if 'adx' in self.trend:
            adx_signal = pd.Series(0, index=self.df.index)
            adx_threshold = 25
            bullish_adx = (self.df['di+'] > self.df['di-']) & (self.df['adx'] > adx_threshold)
            bearish_adx = (self.df['di-'] > self.df['di+']) & (self.df['adx'] > adx_threshold)
            adx_signal[bullish_adx] = 1
            adx_signal[bearish_adx] = -1
            self.cols['ADX_signal'] = adx_signal

        if 'sar' in self.trend:
            sar_signal = pd.Series(0, index=self.df.index)
            sar_signal[self.df['close'] > self.df['sar']] = 1
            sar_signal[self.df['close'] < self.df['sar']] = -1
            self.cols['SAR_signal'] = sar_signal

  
    def volume_signal_generator(self):
        if 'obv' in self.volume:
            self.cols['OBV_signal'] = np.where(self.df['obv'] > self.df['obv_ma_10'], 1,
                                            np.where(self.df['obv'] < self.df['obv_ma_10'], -1, 0))

        if 'ad' in self.volume:
            self.cols['AD_signal'] = np.where(self.df['ad'] > self.df['ad_ma_10'], 1,
                                            np.where(self.df['ad'] < self.df['ad_ma_10'], -1, 0))

        if 'adosc' in self.volume:
            self.cols['ADOSC_signal'] = np.where(self.df['adosc'] > 0, 1,
                                                np.where(self.df['adosc'] < 0, -1, 0))

        if 'mfi' in self.volume:
            self.cols['MFI_signal'] = np.where(self.df['mfi_14'] > 80, -1,
                                            np.where(self.df['mfi_14'] < 20, 1, 0))

        if 'custom_volume' in self.volume:
            self.cols['VOL_signal'] = np.where(self.df['volume'] > 1.5 * self.df['vol_ma_20'], 1, 0)

       
    def momentum_signal_generator(self):
        if 'adx' in self.momentum:
            self.cols['ADX__signal'] = np.where(self.df['adx'] > 25, 1,
                                                np.where(self.df['adx'] < 20, -1, 0))

        if 'adxr' in self.momentum:
            self.cols['ADXR__signal'] = np.where(self.df['adxr'] > 25, 1,
                                                np.where(self.df['adxr'] < 20, -1, 0))

        if 'apo' in self.momentum:
            self.cols['APO__signal'] = np.where(self.df['apo'] > 0, 1,
                                                np.where(self.df['apo'] < 0, -1, 0))

        if 'aroon' in self.momentum:
            self.cols['AROON__signal'] = np.where(self.df['aroon'] > 50, 1, -1)

        if 'aroonosc' in self.momentum:
            self.cols['AROONOSC__signal'] = np.where(self.df['aroonosc'] > 0, 1,
                                                    np.where(self.df['aroonosc'] < 0, -1, 0))

        if 'bop' in self.momentum:
            self.cols['BOP__signal'] = np.where(self.df['bop'] > 0, 1,
                                                np.where(self.df['bop'] < 0, -1, 0))

        if 'cci' in self.momentum:
            self.cols['CCI__signal'] = np.where(self.df['cci_20'] > 100, -1,
                                                np.where(self.df['cci_20'] < -100, 1, 0))

        if 'cmo' in self.momentum:
            self.cols['CMO__signal'] = np.where(self.df['cmo_14'] > 50, 1,
                                                np.where(self.df['cmo_14'] < -50, -1, 0))

        if 'dx' in self.momentum:
            self.cols['DX__signal'] = np.where(self.df['dx_14'] > 25, 1, 0)

        if 'macd' in self.momentum:
            self.cols['MACD__signal'] = np.where(self.df['macd'] > 0, 1,
                                                np.where(self.df['macd'] < 0, -1, 0))

        if 'macdext' in self.momentum:
            self.cols['MACDEXT__signal'] = np.where(self.df['macdext'] > 0, 1,
                                                    np.where(self.df['macdext'] < 0, -1, 0))

        if 'macdfix' in self.momentum:
            self.cols['MACDFIX__signal'] = np.where(self.df['macdfix'] > 0, 1,
                                                    np.where(self.df['macdfix'] < 0, -1, 0))

        if 'mfi' in self.momentum:
            self.cols['MFI__signal'] = np.where(self.df['mfi'] > 80, -1,
                                                np.where(self.df['mfi'] < 20, 1, 0))

        if 'minus_di' in self.momentum:
            self.cols['MINUS_DI__signal'] = np.where(self.df['minus_di_14'] > 20, -1, 0)

        if 'minus_dm' in self.momentum:
            self.cols['MINUS_DM__signal'] = np.where(self.df['minus_dm_14'] > 0, -1, 0)

        if 'mom' in self.momentum:
            self.cols['MOM__signal'] = np.where(self.df['mom_10'] > 0, 1,
                                                np.where(self.df['mom_10'] < 0, -1, 0))

        if 'plus_di' in self.momentum:
            self.cols['PLUS_DI__signal'] = np.where(self.df['plus_di_14'] > 20, 1, 0)

        if 'plus_dm' in self.momentum:
            self.cols['PLUS_DM__signal'] = np.where(self.df['plus_dm_14'] > 0, 1, 0)

        if 'ppo' in self.momentum:
            self.cols['PPO__signal'] = np.where(self.df['ppo'] > 0, 1,
                                                np.where(self.df['ppo'] < 0, -1, 0))

        if 'roc' in self.momentum:
            self.cols['ROC__signal'] = np.where(self.df['roc_10'] > 0, 1,
                                                np.where(self.df['roc_10'] < 0, -1, 0))

        if 'rocp' in self.momentum:
            self.cols['ROCP__signal'] = np.where(self.df['rocp_10'] > 0, 1,
                                                np.where(self.df['rocp_10'] < 0, -1, 0))

        if 'rocr' in self.momentum:
            self.cols['ROCR__signal'] = np.where(self.df['rocr_10'] > 1, 1,
                                                np.where(self.df['rocr_10'] < 1, -1, 0))

        if 'rocr100' in self.momentum:
            self.cols['ROCR100__signal'] = np.where(self.df['rocr100_10'] > 100, 1,
                                                    np.where(self.df['rocr100_10'] < 100, -1, 0))

        if 'rsi' in self.momentum:
            self.cols['RSI__signal'] = np.where(self.df['rsi_14'] > 70, -1,
                                                np.where(self.df['rsi_14'] < 30, 1, 0))

        if 'stoch' in self.momentum:
            self.cols['STOCH__signal'] = np.where(self.df['stoch_slowk_14'] > self.df['stoch_slowd_3'], 1,
                                                np.where(self.df['stoch_slowk_14'] < self.df['stoch_slowd_3'], -1, 0))

        if 'stochf' in self.momentum:
            self.cols['STOCHF__signal'] = np.where(self.df['stochf_fastk_14'] > self.df['stochf_fastd_3'], 1,
                                                np.where(self.df['stochf_fastk_14'] < self.df['stochf_fastd_3'], -1, 0))

        if 'stochrsi' in self.momentum:
            self.cols['STOCHRSI__signal'] = np.where(self.df['stochrsi_fastk_14'] > 0.8, -1,
                                                    np.where(self.df['stochrsi_fastk_14'] < 0.2, 1, 0))

        if 'trix' in self.momentum:
            self.cols['TRIX__signal'] = np.where(self.df['trix_14'] > 0, 1,
                                                np.where(self.df['trix_14'] < 0, -1, 0))

        if 'ultosc' in self.momentum:
            self.cols['ULTOSC__signal'] = np.where(self.df['ultosc'] > 70, -1,
                                                np.where(self.df['ultosc'] < 30, 1, 0))

        if 'willr' in self.momentum:
            self.cols['WILLR__signal'] = np.where(self.df['willr_14'] > -20, -1,
                                                np.where(self.df['willr_14'] < -80, 1, 0))
    
    def pattern_signal_generator(self):
        for pattern in self.patterns:
            pattern_upper = pattern.upper()
            if pattern in self.df.columns:
                signal_col = f"{pattern_upper}_signal"
                self.cols[signal_col] = np.where(self.df[pattern] > 0, 1,
                                                np.where(self.df[pattern] < 0, -1, 0))

    def price_signal_generator(self):
        if 'avgprice' in self.price:
            self.cols['AVGPRICE__signal'] = self.df['avgprice'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
        if 'medprice' in self.price:
            self.cols['MEDPRICE__signal'] = self.df['medprice'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
        if 'typprice' in self.price:
            self.cols['TYPPRICE__signal'] = self.df['typprice'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
        if 'wclprice' in self.price:
            self.cols['WCLPRICE__signal'] = self.df['wclprice'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


    def cycle_signal_generator(self):
        if 'ht_dcperiod' in self.cycle:
            self.cols['HT_DCPERIOD__signal'] = self.df['ht_dcperiod'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
        if 'ht_dcphase' in self.cycle:
            self.cols['HT_DCPHASE__signal'] = self.df['ht_dcphase'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
        if 'ht_phasor' in self.cycle:
            self.cols['HT_PHASOR__signal'] = self.df['ht_phasor_inphase'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        if 'ht_sine' in self.cycle:
            self.cols['HT_SINE__signal'] = np.where(self.df['ht_sine'] > 0, 1,
                                                    np.where(self.df['ht_sine'] < 0, -1, 0))
            
        if 'ht_trendmode' in self.cycle:
            self.cols['HT_TRENDMODE__signal'] = np.where(self.df['ht_trendmode'] == 1, 1,
                                                        np.where(self.df['ht_trendmode'] == 0, -1, 0))

    
    def volatility_signal_generator(self):
        if 'atr' in self.volatility:
            self.cols['ATR_signal'] = np.where(self.df['atr_14'] > self.df['atr_ma_14'], 1,
                                            np.where(self.df['atr_14'] < self.df['atr_ma_14'], -1, 0))

        if 'natr' in self.volatility:
            self.cols['NATR_signal'] = np.where(self.df['natr_14'] > self.df['natr_ma_14'], 1,
                                                np.where(self.df['natr_14'] < self.df['natr_ma_14'], -1, 0))

        if 'cci' in self.volatility:
            self.df['trange_ma_14'] = self.df['trange'].rolling(14).mean()
            self.cols['TRANGE_signal'] = np.where(self.df['trange'] > self.df['trange_ma_14'], 1,
                                                np.where(self.df['trange'] < self.df['trange_ma_14'], -1, 0))

        if 'bbands' in self.volatility:
            self.cols['BBANDS_signal'] = np.where(self.df['close'] > self.df['bbands_upper_20'], -1,
                                                np.where(self.df['close'] < self.df['bbands_lower_20'], 1, 0))

        if 'stddev' in self.volatility:
            self.cols['STDDEV_signal'] = np.where(self.df['stddev_20'] > self.df['stddev_ma_14'], 1,
                                                np.where(self.df['stddev_20'] < self.df['stddev_ma_14'], -1, 0))
            
            
    def generate_singal(self):
        self.trend_signal_generator()
        self.volatility_signal_generator()
        self.momentum_signal_generator()
        self.volume_signal_generator()
        self.price_signal_generator()
        self.cycle_signal_generator()
        self.pattern_signal_generator()
        self.df = pd.concat([self.df, pd.DataFrame(self.cols, index=self.df.index)], axis=1)
        print("Number of columns in the dataset",self.df.shape[1])
        
        return self.df
                    
        




