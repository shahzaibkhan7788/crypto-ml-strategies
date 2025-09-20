import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

class LedgerAnalyzer:
    def __init__(self, filepath_or_buffer):
        if isinstance(filepath_or_buffer, StringIO):
            self.df = pd.read_csv(filepath_or_buffer, parse_dates=['datetime'])
        else:
            self.df = pd.read_csv(filepath_or_buffer, parse_dates=['datetime'])
        self.stats = {}
        self.initial_capital = 1000  # Default, can be set via method

    def preprocess(self):
        """Prepare data for analysis"""
        self.df.sort_values('datetime', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Calculate daily returns if needed
        if 'daily_return' not in self.df.columns:
            self.df['daily_return'] = self.df['pnl'].pct_change()
            
    
    def _sanitize_stats(self):
        for k, v in self.stats.items():
            if isinstance(v, float):
                if np.isinf(v):
                    self.stats[k] = 1e6 if v > 0 else -1e6  # Replace inf with large finite value
                elif np.isnan(v):
                    self.stats[k] = 0.0  # Replace NaN with 0


    def set_initial_capital(self, capital):
        self.initial_capital = capital

    def calculate_all_stats(self):
        """Calculate all metrics organized by category"""
        self.preprocess()
        df = self.df
        
        # Identify trades
        trades = self._identify_trades()
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] < 0]
        
        # Calculate durations
        durations = self._calculate_trade_durations()
        daily_returns = self._get_daily_returns()
        
        # ====================== PROFITABILITY METRICS ======================
        """Calculate all metrics organized by category"""
        self.preprocess()
        df = self.df
        
        trades = self._identify_trades()
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] < 0]
        
        durations = self._calculate_trade_durations()
        daily_returns = self._get_daily_returns()

        # PROFITABILITY METRICS
        self.stats.update({
            "profitability_net_profit": round(trades['pnl'].sum(), 4),
            "profitability_gross_profit": round(wins['pnl'].sum(), 4),
            "profitability_gross_loss": round(losses['pnl'].sum(), 4),
            "profitability_return_on_account": round((trades['pnl'].sum() / self.initial_capital) * 100, 2),
            "profitability_annualized_return": self._calculate_annualized_return(),
            "profitability_profit_factor": round(abs(wins['pnl'].sum()/losses['pnl'].sum()), 2) if losses['pnl'].sum() != 0 else float('inf'),
            "profitability_expectancy": self._calculate_expectancy(wins, losses, trades),
            "profitability_avg_profit_per_trade": round(trades['pnl'].mean(), 4) if len(trades) > 0 else 0,
            "profitability_avg_win": round(wins['pnl'].mean(), 4) if len(wins) > 0 else 0,
            "profitability_avg_loss": round(losses['pnl'].mean(), 4) if len(losses) > 0 else 0,
            "profitability_max_win": round(wins['pnl'].max(), 4) if len(wins) > 0 else 0,
            "profitability_max_loss": round(losses['pnl'].min(), 4) if len(losses) > 0 else 0,
            "profitability_median_win": round(wins['pnl'].median(), 4) if len(wins) > 0 else 0,
            "profitability_median_loss": round(losses['pnl'].median(), 4) if len(losses) > 0 else 0,
            "profitability_win_std_dev": round(wins['pnl'].std(), 4) if len(wins) > 1 else 0,
            "profitability_loss_std_dev": round(losses['pnl'].std(), 4) if len(losses) > 1 else 0,
        })
        
        print("profitable status calcaulted")

        # TRADE ACTIVITY METRICS
        self.stats.update({
            "activity_total_trades": len(trades),
            "activity_long_trades": len(trades[trades['predicted_direction'] == 'long']),
            "activity_short_trades": len(trades[trades['predicted_direction'] == 'short']),
            "activity_winning_trades": len(wins),
            "activity_losing_trades": len(losses),
            "activity_avg_trade_duration": round(durations.mean(), 2) if not durations.empty else 0,
            "activity_trades_per_day": round(len(trades)/df['datetime'].dt.date.nunique(), 2) if df['datetime'].dt.date.nunique() > 0 else 0,
            "activity_long_win_rate": self._calculate_directional_win_rate(trades, 'long'),
            "activity_short_win_rate": self._calculate_directional_win_rate(trades, 'short'),
        })
        
        
        print("trade activity calcaulted")
        drawdown_stats = self._calculate_drawdown_stats()
        print("here one more function called call calculate draw down status")
        self.stats.update({
            "risk_max_drawdown": round(drawdown_stats['max_drawdown'], 4),
            "risk_avg_drawdown": round(drawdown_stats['avg_drawdown'], 4),
            "risk_drawdown_duration": drawdown_stats['max_duration_days'],
            "risk_volatility": round(daily_returns.std(), 4) if len(daily_returns) > 1 else 0,
            "risk_var_95": self._calculate_var(daily_returns),
            "risk_cvar_95": self._calculate_cvar(daily_returns),
            "risk_ulcer_index": self._calculate_ulcer_index(),
            "risk_risk_of_ruin": self._estimate_risk_of_ruin(),
        })
        
        print("risk and drawdown occur calcaulted")

        # RISK-REWARD EFFICIENCY METRICS
        self.stats.update({
            "efficiency_risk_reward_ratio": round(abs(wins['pnl'].mean()/losses['pnl'].mean()), 2) if len(losses) > 0 else float('inf'),
            "efficiency_payoff_ratio": self._calculate_payoff_ratio(wins, losses),
            "efficiency_k_ratio": self._calculate_k_ratio(),
            "efficiency_profit_per_dollar_risked": round(trades['pnl'].sum() / (self.initial_capital * len(trades)), 4) if len(trades) > 0 else 0,
            "efficiency_kelly_criterion": self._calculate_kelly_criterion(),
        })

        # STATISTICAL CONSISTENCY METRICS
        self.stats.update({
            "consistency_sharpe_ratio": self._calculate_sharpe_ratio(),
            "consistency_sortino_ratio": self._calculate_sortino_ratio(),
            "consistency_calmar_ratio": self._calculate_calmar_ratio(),
            "consistency_z_score": self._calculate_z_score(),
            "consistency_skewness": round(daily_returns.skew(), 4) if len(daily_returns) > 2 else 0,
            "consistency_kurtosis": round(daily_returns.kurtosis(), 4) if len(daily_returns) > 3 else 0,
            "consistency_information_ratio": self._calculate_information_ratio(),
            "consistency_r_squared": self._calculate_r_squared(),
        })
        
        print("statistical calcaulted")

        # PORTFOLIO MANAGEMENT METRICS
        self.stats.update({
            "portfolio_alpha": self._calculate_alpha(),
            "portfolio_beta": self._calculate_beta(),
            "portfolio_turnover": self._calculate_turnover(),
            "portfolio_position_concentration": self._calculate_concentration(),
            "portfolio_leverage_ratio": self._estimate_leverage(),
        })
        
        print("portfolio status calcaulted")

        # EXECUTION & COST METRICS
        self.stats.update({
            "execution_slippage_per_trade": self._estimate_slippage(),
            "execution_commission_per_trade": self._estimate_commissions(),
            "execution_spread_cost": self._estimate_spread_cost(),
            "execution_fill_rate": self._estimate_fill_rate(),
        })
        
        print("now pass to senitistat calcaulted")
        self._add_derived_metrics(wins, losses, trades)

        self._sanitize_stats()
        return self.stats


    # ====================== HELPER METHODS ======================
    
    def _identify_trades(self):
        """Identify trade entries in the ledger"""
        return self.df[
            self.df['action'].str.contains('buy|sell', case=False, na=False) |
            self.df['predicted_direction'].isin(['long', 'short'])
        ]
    
    def _calculate_trade_durations(self):
        """Calculate durations between trade entry and exit in hours"""
        trades = self._get_completed_trades()
        durations = []
        entry_time = None
        
        for _, row in trades.iterrows():
            if row['predicted_direction'] in ['long', 'short'] and 'buy' in str(row['action']).lower():
                entry_time = row['datetime']
            elif entry_time is not None:
                exit_time = row['datetime']
                duration = (exit_time - entry_time).total_seconds() / 3600
                durations.append(duration)
                entry_time = None
        
        return pd.Series(durations)
    
    def _get_completed_trades(self):
        """Filter dataframe to only include trade entries and exits"""
        return self.df[
            self.df['action'].str.contains('buy|sell', case=False, na=False) |
            self.df['predicted_direction'].isin(['long', 'short'])
        ].sort_values('datetime')
    
    def _get_daily_returns(self):
        """Calculate daily returns from PnL"""
        daily_pnl = self.df.groupby(self.df['datetime'].dt.date)['pnl'].sum()
        return daily_pnl.pct_change().dropna()
    
    def _calculate_drawdown_stats(self):
        """Calculate drawdown statistics"""
        cumulative = self.df['pnl'].cumsum()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak)
        
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
        
        # Calculate duration of max drawdown
        max_dd_period = (drawdown == max_drawdown).idxmax()
        recovery_idx = (cumulative[max_dd_period:] >= peak[max_dd_period]).idxmax()
        duration_days = (self.df.loc[recovery_idx, 'datetime'] - self.df.loc[max_dd_period, 'datetime']).days
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_duration_days': duration_days
        }
    
    def _calculate_annualized_return(self):
        """Calculate annualized return percentage"""
        if len(self.df) < 2:
            return 0
            
        days = (self.df['datetime'].iloc[-1] - self.df['datetime'].iloc[0]).days
        if days == 0:
            return 0
            
        total_return = self.df['pnl'].sum() / self.initial_capital
        annualized = ((1 + total_return) ** (365/days)) - 1
        return round(annualized * 100, 2)
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate Sharpe ratio"""
        daily_returns = self._get_daily_returns()
        if len(daily_returns) < 2:
            return 0
            
        excess_returns = daily_returns - risk_free_rate/252
        return round(excess_returns.mean() / excess_returns.std() * np.sqrt(252), 2)
    
    def _calculate_sortino_ratio(self, risk_free_rate=0.0):
        """Calculate Sortino ratio"""
        daily_returns = self._get_daily_returns()
        if len(daily_returns) < 2:
            return 0
            
        excess_returns = daily_returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 1:
            return float('inf')
            
        downside_std = downside_returns.std()
        return round(excess_returns.mean() / downside_std * np.sqrt(252), 2)
    
    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio (return vs max drawdown)"""
        max_drawdown = self.stats.get('max_drawdown', 0)
        if max_drawdown >= 0:  # No drawdown or positive "drawdown"
            return float('inf')
            
        annual_return = self.stats.get('annualized_return', 0) / 100
        return round(annual_return / abs(max_drawdown), 2)
    
    def _calculate_expectancy(self, wins, losses, trades):
        """Calculate trading expectancy"""
        if len(trades) == 0:
            return 0
            
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        win_rate = len(wins)/len(trades)
        loss_rate = len(losses)/len(trades)
        
        return round((win_rate * avg_win) - (loss_rate * abs(avg_loss)), 4)
    
    def _calculate_directional_win_rate(self, trades, direction):
        """Calculate win rate for long or short trades"""
        dir_trades = trades[trades['predicted_direction'] == direction]
        dir_wins = dir_trades[dir_trades['pnl'] > 0]
        return round(len(dir_wins)/len(dir_trades)*100, 2) if len(dir_trades) > 0 else 0
    
    def _calculate_var(self, returns, confidence_level=0.95):
        print("here is the var")
        """Calculate Value at Risk"""
        if len(returns) < 2:
            return 0
        print("return var")
        return round(returns.quantile(1 - confidence_level), 4)
    
    def _calculate_cvar(self, returns, confidence_level=0.95):
        print("here is the svar")
        """Calculate Conditional Value at Risk"""
        if len(returns) < 2:
            return 0
        var = returns.quantile(1 - confidence_level)
        cvar = returns[returns <= var].mean()
        print("return cvar")
        return round(cvar, 4)
    
    def _calculate_ulcer_index(self):
        """Calculate Ulcer Index"""
        print("here is the uclr")
        cumulative = self.df['pnl'].cumsum()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = ((cumulative - peak) / peak) * 100
        squared_dd = drawdown ** 2
        print("return ucvar")
        return round(np.sqrt(squared_dd.mean()), 2) if len(squared_dd) > 0 else 0
    
    def _estimate_risk_of_ruin(self, simulations=1000, max_trades_per_sim=1000):
        print("here is the estimate")
        """Estimate probability of losing entire capital"""
        win_rate = self.stats.get('win_rate', 50) / 100
        avg_win = self.stats.get('avg_profit_per_trade', 1)
        avg_loss = abs(self.stats.get('avg_loss_per_trade', 1))

        if avg_loss == 0:
            return 0

        ruin_count = 0
        for _ in range(simulations):
            capital = self.initial_capital
            trades = 0
            while capital > 0 and capital < 2 * self.initial_capital and trades < max_trades_per_sim:
                if np.random.random() < win_rate:
                    capital += avg_win
                else:
                    capital -= avg_loss
                trades += 1
            if capital <= 0:
                ruin_count += 1

        print("return estimate")
        return round(ruin_count / simulations * 100, 2)

    
    def _calculate_payoff_ratio(self, wins, losses):
        """Calculate payoff ratio (average win vs average loss)"""
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return float('inf')
            
        return round(avg_win / abs(avg_loss), 2)
    
    def _calculate_k_ratio(self):
        """Calculate K-Ratio (performance consistency over time)"""
        if len(self.df) < 3:
            return 0
            
        cumulative = self.df['pnl'].cumsum()
        x = np.arange(len(cumulative))
        slope, _, _, _, _ = np.polyfit(x, cumulative, 1, full=True)
        return round(slope[0] * 100, 2)
    
    def _calculate_alpha_beta(self):
        """Calculate alpha and beta against a benchmark (simplified)"""
        # In a real implementation, you'd compare to an actual benchmark
        daily_returns = self._get_daily_returns()
        if len(daily_returns) < 2:
            return 0, 0
            
        # Mock benchmark returns (S&P 500 typically has ~8% annual return)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.01, len(daily_returns)))
        
        covariance = np.cov(daily_returns, benchmark_returns)[0, 1]
        benchmark_var = np.var(benchmark_returns)
        
        beta = covariance / benchmark_var
        alpha = daily_returns.mean() - (beta * benchmark_returns.mean())
        
        return alpha * 252, beta  # Annualized alpha
    
    def _calculate_alpha(self):
        alpha, _ = self._calculate_alpha_beta()
        return round(alpha, 4)
    
    def _calculate_beta(self):
        _, beta = self._calculate_alpha_beta()
        return round(beta, 4)
    
    def _calculate_r_squared(self):
        """Calculate R-squared against benchmark"""
        daily_returns = self._get_daily_returns()
        if len(daily_returns) < 2:
            return 0
            
        # Mock benchmark
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.01, len(daily_returns)))
        correlation = daily_returns.corr(benchmark_returns)
        return round(correlation ** 2, 4)
    
    def _estimate_slippage(self):
        """Estimate average slippage per trade"""
        # This assumes your data has execution price vs intended price
        if 'slippage' in self.df.columns:
            return round(self.df['slippage'].mean(), 4)
        return 0.0  # Default if no slippage data
    
    def _estimate_commissions(self):
        """Estimate average commission per trade"""
        # Default commission estimate
        return 0.001  # 0.1% per trade
    
    def _estimate_spread_cost(self):
        """Estimate spread cost impact"""
        # This would require bid/ask data
        return 0.0005  # 0.05% spread estimate
    
    def _estimate_fill_rate(self):
        """Estimate order fill rate"""
        if 'fill_status' in self.df.columns:
            fills = self.df['fill_status'].str.contains('filled', case=False).sum()
            return round(fills / len(self.df) * 100, 2)
        return 100.0  # Assume perfect fills if no data
    
    def _calculate_turnover(self):
        """Calculate portfolio turnover rate"""
        trades = self._identify_trades()
        return round(len(trades) / (self.initial_capital / 10000), 2)  # Normalized
    
    def _calculate_concentration(self):
        """Calculate position concentration"""
        # This would require position size data
        return 0.0  # Default if no position data
    
    def _calculate_kelly_criterion(self):
        """Calculate optimal Kelly position size"""
        win_rate = self.stats.get('win_rate', 50) / 100
        win_loss_ratio = self.stats.get('risk_reward_ratio', 1)
        
        if win_loss_ratio == 0:
            return 0
            
        return round(win_rate - ((1 - win_rate) / win_loss_ratio), 4)
    
    def _estimate_leverage(self):
        """Estimate average leverage used"""
        # This would require position size vs capital data
        return 1.0  # Default no leverage
    
    def _calculate_z_score(self):
        """Calculate Z-score for strategy consistency"""
        wins = self.stats.get('winning_trades', 0)
        losses = self.stats.get('losing_trades', 0)
        total = wins + losses
        
        if total == 0:
            return 0
            
        win_rate = wins / total
        expected_wins = total * 0.5  # Null hypothesis: 50% win rate
        variance = total * 0.5 * 0.5
        std_dev = np.sqrt(variance)
        
        if std_dev == 0:
            return 0
            
        return round((wins - expected_wins) / std_dev, 2)
    
    def _calculate_information_ratio(self):
        """Calculate Information Ratio (active return vs tracking error)"""
        # Simplified version without benchmark
        daily_returns = self._get_daily_returns()
        if len(daily_returns) < 2:
            return 0
            
        return round(daily_returns.mean() / daily_returns.std() * np.sqrt(252), 2)
    
    def _add_derived_metrics(self, wins, losses, trades):
        """Add additional metrics that depend on previously calculated stats"""
        self.stats.update({
            "avg_win": round(wins['pnl'].mean(), 4) if len(wins) > 0 else 0,
            "avg_loss": round(losses['pnl'].mean(), 4) if len(losses) > 0 else 0,
            "max_win": round(wins['pnl'].max(), 4) if len(wins) > 0 else 0,
            "max_loss": round(losses['pnl'].min(), 4) if len(losses) > 0 else 0,
            "median_win": round(wins['pnl'].median(), 4) if len(wins) > 0 else 0,
            "median_loss": round(losses['pnl'].median(), 4) if len(losses) > 0 else 0,
            "win_std_dev": round(wins['pnl'].std(), 4) if len(wins) > 1 else 0,
            "loss_std_dev": round(losses['pnl'].std(), 4) if len(losses) > 1 else 0,
        })

    def get_stats(self):
        """Return calculated statistics"""
        return self.stats