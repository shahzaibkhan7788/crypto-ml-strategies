import talib
import pandas as pd


class technical_indicator:
    def __init__(self, df,indicators_categories,true_indicators,indicators_config=None):
        self.cols = {}
        self.df = df
        self.indicators_categories= indicators_categories
        self.indicator_config= indicators_config
        self.true_indicators= true_indicators
        self.trend= []
        self.volume= []
        self.momentum= []
        self.volatility= []
        self.price= []
        self.cycle=[]
        self.patterns= []
        self.result_df= self.df.copy()
        print("here is the indicators config:",indicators_config)
        
        
    def categorize_indicators(self):
        self.trend = [ind for ind in self.true_indicators if self.indicators_categories.get(ind) == 'trend']
        self.volume = [ind for ind in self.true_indicators if self.indicators_categories.get(ind) == 'volume']
        self.cycle = [ind for ind in self.true_indicators if self.indicators_categories.get(ind) == 'cycle']
        self.price = [ind for ind in self.true_indicators if self.indicators_categories.get(ind) == 'price']
        self.momentum = [ind for ind in self.true_indicators if self.indicators_categories.get(ind) == 'momentum']
        self.volatility = [ind for ind in self.true_indicators if self.indicators_categories.get(ind) == 'volatility']
        self.patterns = [ind for ind in self.true_indicators if self.indicators_categories.get(ind) == 'pattern']
        
    def trend_indicator(self):
        for ind in self.trend:
        
            if ind == 'sma':
                    window1, window2 = (
                        (
                            self.indicator_config.get(ind, {}).get("window1", 20),
                            self.indicator_config.get(ind, {}).get("window2", 100)
                        )
                        if self.indicator_config is not None else (20, 100))
                    self.cols[f'sma_50'] = talib.SMA(self.df['close'], timeperiod=window1)
                    self.cols[f'sma_100'] = talib.SMA(self.df['close'], timeperiod=window2)

            elif ind == 'ema':
                    window1, window2 = (
                        (
                            self.indicator_config.get(ind, {}).get("window1", 20),
                            self.indicator_config.get(ind, {}).get("window2", 80)
                        )
                        if self.indicator_config is not None else (20, 80))
                    self.cols[f'ema_20'] = talib.EMA(self.df['close'], timeperiod=window1)
                    self.cols[f'ema_80'] = talib.EMA(self.df['close'], timeperiod=window2)


            elif ind == 'macd': #three values
                fast, slow, signal = (
                    (
                        self.indicator_config.get(ind, {}).get("fastperiod", 12),
                        self.indicator_config.get(ind, {}).get("slowperiod", 26),
                        self.indicator_config.get(ind, {}).get("signalperiod", 9)
                    )
                    if self.indicator_config is not None else (12, 26, 9))
            
                macd, signal_line, hist = talib.MACD(self.df['close'], fast, slow, signal)
                self.cols['macd'] = macd
                self.cols['signal_macd'] = signal_line
                self.cols['macd_hist'] = hist


            elif ind == 'adx': 
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config is not None else 14)
                
                self.cols['DI+'] = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)
                self.cols['DI-'] = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)
                self.cols['ADX'] = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)

            elif ind == 'sar': #three values
                 acc, maxv = (
                    (
                        self.indicator_config.get(ind, {}).get("acceleration", 0.02),
                        self.indicator_config.get(ind, {}).get("maximum", 0.2))
                    if self.indicator_config is not None else (0.02, 0.2)
                    )
            
                 self.cols['SAR'] = talib.SAR(self.df['high'], self.df['low'], acceleration=acc, maximum=maxv)

    
    
    def volume_indicator(self):
        
        for ind in self.volume:
  
            if ind=='obv':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['OBV'] = talib.OBV(self.df['close'], self.df['volume'])
                self.cols['OBV_MA_10'] = self.cols['OBV'].rolling(window).mean()  # Use self.cols here

            elif ind=='ad':
                window = ( self.indicator_config.get(ind, {}).get("window", 10)
                          if self.indicator_config else 10)
                self.cols['AD'] = talib.AD(self.df['high'], self.df['low'], self.df['close'], self.df['volume'])
                self.cols['AD_MA_10'] = self.cols['AD'].rolling(window).mean()  # Use self.cols here

            elif ind=='adosc': #three vlaues
                fast, slow = (
                    (
                        self.indicator_config.get(ind, {}).get("fastperiod", 12),
                        self.indicator_config.get(ind, {}).get("slowperiod", 26),
                    )
                    if self.indicator_config else (3, 10))
                 
                self.cols['ADOSC'] = talib.ADOSC(self.df['high'], self.df['low'], self.df['close'], self.df['volume'],
                                                fastperiod=fast, slowperiod=slow)

            elif ind=='mfi':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['MFI_14'] = talib.MFI(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], timeperiod=window)

            elif ind=='custom_volume':
                window = ( self.indicator_config.get(ind, {}).get("window", 20)
                          if self.indicator_config else 20)
                self.cols['VOL_MA_20'] = self.df['volume'].rolling(window).mean()
            
    def momentum_indicator(self):
        for ind in self.momentum:

            if ind=='rsi':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['RSI_14'] = talib.RSI(self.df['close'], timeperiod=window)
            elif ind=='stoch':
                fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype = (
                        (
                            self.indicator_config.get(ind, {}).get("fastk_period", 14),
                            self.indicator_config.get(ind, {}).get("slowk_period", 3),
                            self.indicator_config.get(ind, {}).get("slowk_matype", 0),
                            self.indicator_config.get(ind, {}).get("slowd_period", 3),
                            self.indicator_config.get(ind, {}).get("slowd_matype", 0)
                        )
                        if self.indicator_config else (14, 3, 0, 3, 0)
                    )
                slowk, slowd = talib.STOCH(
                    self.df['high'], self.df['low'], self.df['close'],
                    fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype,
                    slowd_period=slowd_period, slowd_matype=slowd_matype)
                self.cols['STOCH_slowk_14'] = slowk
                self.cols['STOCH_slowd_3'] = slowd
                
            elif ind=='cci':
                window = ( self.indicator_config.get(ind, {}).get("window", 20)
                          if self.indicator_config else 20)
                self.cols['CCI_20'] = talib.CCI(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)
            elif ind=='willr':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['WILLR_14'] = talib.WILLR(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)
            elif ind=='mom':
                window = ( self.indicator_config.get(ind, {}).get("window", 10)
                          if self.indicator_config else 10)
                self.cols['MOM_10'] = talib.MOM(self.df['close'], timeperiod=window)


            elif ind=='bop':
                self.cols['BOP'] = talib.BOP(self.df['open'], self.df['high'], self.df['low'], self.df['close'])

            elif ind=='cmo':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['CMO_14'] = talib.CMO(self.df['close'], timeperiod=window)

            elif ind=='dx':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['DX_14'] = talib.DX(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)

            elif ind=='macd': #three values
                fast, slow, signal = (
                    (
                        self.indicator_config.get(ind, {}).get("fastperiod", 9),
                        self.indicator_config.get(ind, {}).get("slowperiod", 30),
                        self.indicator_config.get(ind, {}).get("signalperiod", 10)
                    )
                    if self.indicator_config else (9, 30, 10))
                macd, macdsignal, macdhist = talib.MACD(self.df['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
                self.cols['MACD'] = macd
                self.cols['signal_MACD'] = macdsignal
                self.cols['MACD_hist'] = macdhist

            elif ind=='macdext': #threee values
                fastperiod,fastmatype,slowperiod,slowmatype,singalperiod,signalmatype = (
                        (
                            self.indicator_config.get(ind, {}).get("fastperiod", 12),
                            self.indicator_config.get(ind, {}).get("fastmatype", 0),
                            self.indicator_config.get(ind, {}).get("slowperiod", 26),
                            self.indicator_config.get(ind, {}).get("slowmatype", 0),
                            self.indicator_config.get(ind, {}).get("singalperiod", 9),
                            self.indicator_config.get(ind, {}).get("singalmatype", 0)
                        )
                        if self.indicator_config else (12, 0, 26, 0, 9,0)
                    )
                macd_ext, macd_signal_ext, macd_hist_ext = talib.MACDEXT(
                    self.df['close'], fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=singalperiod, signalmatype=signalmatype)
                self.cols['MACDEXT'] = macd_ext
                self.cols['signal_MACDEXT'] = macd_signal_ext
                self.cols['MACDEXT_hist'] = macd_hist_ext

            elif ind=='macdfix':
                window = ( self.indicator_config.get(ind, {}).get("window", 9)
                          if self.indicator_config else 9)
                macd_fix, macd_fix_signal, macd_fix_hist = talib.MACDFIX(self.df['close'], signalperiod=window)
                self.cols['MACDFIX'] = macd_fix
                self.cols['signal_MACDFIX'] = macd_fix_signal
                self.cols['MACDFIX_hist'] = macd_fix_hist

            elif ind=='mfi':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['MFI_14'] = talib.MFI(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], timeperiod=window)

            elif ind=='minus_di':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['MINUS_DI_14'] = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)

            elif ind=='minus_dm':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['MINUS_DM_14'] = talib.MINUS_DM(self.df['high'], self.df['low'], timeperiod=window)

            elif ind=='plus_di':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['PLUS_DI_14'] = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=window)

            elif ind=='plus_dm':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['PLUS_DM_14'] = talib.PLUS_DM(self.df['high'], self.df['low'], timeperiod=window)

            elif ind=='ppo': #threee values
                fast, slow, matype = (
                    (
                        self.indicator_config.get(ind, {}).get("fastperiod", 12),
                        self.indicator_config.get(ind, {}).get("slowperiod", 26),
                        self.indicator_config.get(ind, {}).get("matype", 0)
                    )
                    if self.indicator_config else (12, 26, 0))
                self.cols['PPO'] = talib.PPO(self.df['close'], fastperiod=fast, slowperiod=slow, matype=matype)
                #one great issue in your code that you cannot select window for the ppo..


            elif ind=='roc':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['ROC_10'] = talib.ROC(self.df['close'], timeperiod=window)

            elif ind=='rocp':
                window = ( self.indicator_config.get(ind, {}).get("window", 10)
                          if self.indicator_config else 10)
                self.cols['ROCP_10'] = talib.ROCP(self.df['close'], timeperiod=window)

            elif ind=='rocr':
                window = ( self.indicator_config.get(ind, {}).get("window", 10)
                          if self.indicator_config else 10)
                self.cols['ROCR_10'] = talib.ROCR(self.df['close'], timeperiod=window)

            elif ind=='rocr100':
                window = ( self.indicator_config.get(ind, {}).get("window", 10)
                          if self.indicator_config else 10)
                self.cols['ROCR100_10'] = talib.ROCR100(self.df['close'], timeperiod=window)

            elif ind=='stochf':  #three values
                fastk_period, fastd_period, fastd_matype = (
                    (
                        self.indicator_config.get(ind, {}).get("fastk_period", 14),
                        self.indicator_config.get(ind, {}).get("fastd_period", 3),
                        self.indicator_config.get(ind, {}).get("fastd_matype", 0)
                    )
                    if self.indicator_config else (14, 3, 0))
                
                fastk, fastd = talib.STOCHF(self.df['high'], self.df['low'], self.df['close'], fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
                self.cols['STOCHF_fastk_14'] = fastk
                self.cols['STOCHF_fastd_3'] = fastd

            elif ind=='stochrsi': #three values
                timeperiod, fastk, fastd_period, fastd_matype = (
                    (
                        self.indicator_config.get(ind, {}).get("timeperiod", 14),
                        self.indicator_config.get(ind, {}).get("fastk_period", 3),
                        self.indicator_config.get(ind, {}).get("fastd_period", 3),
                        self.indicator_config.get(ind, {}).get("fastd_matype", 0)
                    )
                    if self.indicator_config else (14, 3, 3,0))
                self.cols['STOCHRSI_fastk_14'] = talib.STOCHRSI(self.df['close'], timeperiod=timeperiod, fastk_period=fastk, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]
                self.cols['STOCHRSI_fastd_3'] = talib.STOCHRSI(self.df['close'], timeperiod=timeperiod, fastk_period=fastk, fastd_period=fastd_period, fastd_matype=fastd_matype)[1]

            elif ind=='trix':
                window = ( self.indicator_config.get(ind, {}).get("window", 10)
                          if self.indicator_config else 10)
                print("window in trix:",window)
                self.cols['TRIX_14'] = talib.TRIX(self.df['close'], timeperiod=window)
                print("self. trix column:",self.cols['TRIX_14'])

            elif ind=='ultosc': # three values
                timeperiod1, timeperiod2, timeperiod3 = (
                    (
                        self.indicator_config.get(ind, {}).get("timperiod1", 7),
                        self.indicator_config.get(ind, {}).get("timeperiod2", 14),
                        self.indicator_config.get(ind, {}).get("timeperiod3", 28)
                    )
                    if self.indicator_config else (7, 14, 28))
                self.cols['ULTOSC'] = talib.ULTOSC(self.df['high'], self.df['low'], self.df['close'], timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)

    
    def volatility_indicator(self):
        for ind in self.volatility:
            if ind=='atr':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['ATR_14'] = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
                self.cols['ATR_MA_14'] = self.cols['ATR_14'].rolling(window).mean()

            elif ind=='natr':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['NATR_14'] = talib.NATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
                self.cols['NATR_MA_14'] = self.cols['NATR_14'].rolling(window).mean()

            elif ind=='trange':
                window = ( self.indicator_config.get(ind, {}).get("window", 14)
                          if self.indicator_config else 14)
                self.cols['TRANGE'] = talib.TRANGE(self.df['high'], self.df['low'], self.df['close'])
                self.cols['TRANGE_MA_14'] = self.cols['TRANGE'].rolling(window).mean()

            elif ind=='bbands': #threee values
                timeperiod, nbdevup, nbdevevdn,matype = (
                    (
                        self.indicator_config.get(ind, {}).get("timeperiod", 20),
                        self.indicator_config.get(ind, {}).get("nbdevup", 2),
                        self.indicator_config.get(ind, {}).get("nbdevdn", 2),
                        self.indicator_config.get(ind, {}).get("matype", 0)
                    )
                    if self.indicator_config else (20,2, 2, 0))
                upperband, middleband, lowerband = talib.BBANDS(
                    self.df['close'],
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2,
                    matype=0
                )
                self.cols['BBANDS_upper_20'] = upperband
                self.cols['BBANDS_middle_20'] = middleband
                self.cols['BBANDS_lower_20'] = lowerband

            elif ind=='stddev':
                window = ( self.indicator_config.get(ind, {}).get("window", 20)
                          if self.indicator_config else 20)
                self.cols['STDDEV_20'] = talib.STDDEV(self.df['close'], timeperiod=window, nbdev=1)
                self.cols['STDDEV_MA_14'] = self.cols['STDDEV_20'].rolling(14).mean()

    def cycle_indicator(self):
        for ind in self.cycle:
            if ind=='ht_dcperiod':
                self.cols['HT_DCPERIOD'] = talib.HT_DCPERIOD(self.df['close'])
        
            elif ind=='ht_dcphase':
                self.cols['HT_DCPHASE'] = talib.HT_DCPHASE(self.df['close'])
            
            elif ind=='ht_phasor':
                inphase, quadrature = talib.HT_PHASOR(self.df['close'])
                self.cols['HT_PHASOR_inphase'] = inphase
                self.cols['HT_PHASOR_quadrature'] = quadrature
            
            elif ind=='ht_sine':
                sine, leadsine = talib.HT_SINE(self.df['close'])
                self.cols['HT_SINE'] = sine
                self.cols['HT_SINE_leadsine'] = leadsine
            
            elif ind=='ht_trendmode':
                self.cols['HT_TRENDMODE'] = talib.HT_TRENDMODE(self.df['close'])
                
            

    def pattern_recognition(self):
        pattern_funcs = {func.lower(): func for func in dir(talib) if func.startswith('CDL')}
        for pattern in self.patterns:
            pattern_func_name = pattern.upper()
            if pattern_func_name in pattern_funcs.values():
                func = getattr(talib, pattern_func_name)
                self.cols[pattern_func_name] = func(
                    self.df['open'], self.df['high'], self.df['low'], self.df['close']
                )

    def price_indicator(self):
        for ind in self.price:
            if ind=='avgprice':
                self.cols['AVGPRICE'] = talib.AVGPRICE(self.df['open'], self.df['high'], self.df['low'], self.df['close'])

            elif ind=='medprice':
                self.cols['MEDPRICE'] = talib.MEDPRICE(self.df['high'], self.df['low'])

            elif ind=='typprice':
                self.cols['TYPPRICE'] = talib.TYPPRICE(self.df['high'], self.df['low'], self.df['close'])

            elif ind=='wclprice':
                self.cols['WCLPRICE'] = talib.WCLPRICE(self.df['high'], self.df['low'], self.df['close'])


       
    # will be change as you free for the project   
    def calculate_indicators(self, max_value=None):
        # Clear previous cols
        self.cols = {}

        # Categorize and calculate all indicators with current config
        self.categorize_indicators()
        self.trend_indicator()
        self.volume_indicator()
        self.momentum_indicator()
        self.volatility_indicator()
        self.price_indicator()
        self.cycle_indicator()
        self.pattern_recognition()

        self.result_df = pd.concat(
            [self.df, pd.DataFrame(self.cols, index=self.df.index)],
            axis=1
        )
        print("Indicators calculated successfully", self.result_df.columns)

        nan_counts = self.result_df.isna().sum().sort_values(ascending=False)
        print(nan_counts)

        if max_value is not None:
            print("The max NaN threshold is:", max_value)
            nan_counts = self.result_df.isna().sum()

            for col_name, nan_count in nan_counts.items():
                if nan_count > max_value:
                    base_name = col_name.split("_")[0].lower()
                    if base_name in self.indicator_config:
                        params = self.indicator_config[base_name]
                        print(f"\n⚠ {col_name} has {nan_count} NaNs, reducing parameters: {params}")

                        while nan_count > max_value:
                            updated = False
                            for key in params:
                                if "window" in key or "period" in key:
                                    if params[key] > 1:
                                        old_val = params[key]
                                        params[key] = max(1, params[key] // 2)
                                        updated = True
                                        print(f"🔄 Reduced {base_name} {key} from {old_val} → {params[key]}")

                            if not updated:
                                print(f"❌ Cannot reduce {base_name} parameters further.")
                                break

                            # Update config
                            self.indicator_config[base_name] = params

                            # Recalculate ALL indicators with new config
                            self.calculate_indicators(max_value=None)  # No nested recursion here!

                            # Refresh nan count for this column
                            nan_count = self.result_df[col_name].isna().sum()

                        print(f"✅ {col_name} now has {nan_count} NaNs")

            # Print final NaN counts
            final_nan_counts = self.result_df.isna().sum().sort_values(ascending=False)
            print("\nFinal NaN counts after parameter adjustment:")
            print(final_nan_counts)

        return {
            'result_df': self.result_df,
            'trend': self.trend,
            'volume': self.volume,
            'volatility': self.volatility,
            'momentum': self.momentum,
            'cycle': self.cycle,
            'patterns': self.patterns,
            'price': self.price
        }
