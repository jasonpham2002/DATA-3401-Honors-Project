import pandas as pd
import numpy as np
from strategy import Strategy, MACrossover, RSIMeanReversion, MLHybridStrategy
import matplotlib.pyplot as plt

class Backtest:
    """
    BacktestingEngine simulating daily trading execution, portfolio cash tracking,
    transaction logging, risk-adjusted performance metric calculations, and visualizations.
    """
    def __init__(self, asset_data, strategy, starting_cash=100000):
        self.current_cash = starting_cash
        self.starting_cash = starting_cash
        self.trade_history = {}
        self.cur_own_stocks = 0
        self.money_invested = 0
        self.strat = strategy
        ticker = getattr(strategy, 'ticker', None)
        self.data = Strategy._extract_single_ticker(asset_data, ticker=ticker)
        self.tot_net_worth = {}
        self.trade_profits = []

    def run(self):
        """Executes simulation day-by-day across the historical price series."""
        self.current_cash = self.starting_cash
        self.trade_history = {}
        self.cur_own_stocks = 0
        self.money_invested = 0
        self.tot_net_worth = {}
        self.trade_profits = []

        self.strat.generate_signal()
        self.strat.execute_trade()
        self.strat.calculate_return()

        if self.data is not self.strat.data:
            for col in self.strat.data.columns:
                self.data[col] = self.strat.data[col]

        for index, row in self.data.iterrows():
            pos = row.get('position', 0)
            close_price = row['Close']

            # Buy signal (entry or flip from flat)
            if pos == 1 and self.current_cash >= close_price:
                new_shares = int(self.current_cash // close_price)
                if new_shares > 0:
                    cost = new_shares * close_price
                    self.cur_own_stocks += new_shares
                    self.current_cash -= cost
                    self.money_invested = cost
                    self.trade_history[index] = {
                        "Action": "Buy",
                        "Shares": new_shares,
                        "Price": close_price,
                        "Money spent": cost
                    }

            # Sell signal (exit to cash)
            elif pos == -1 and self.cur_own_stocks > 0:
                proceeds = self.cur_own_stocks * close_price
                profit = proceeds - self.money_invested
                self.current_cash += proceeds
                self.trade_profits.append(profit)
                self.trade_history[index] = {
                    "Action": "Sell",
                    "Shares": self.cur_own_stocks,
                    "Price": close_price,
                    "Money gained": proceeds,
                    "Profit": profit
                }
                self.cur_own_stocks = 0
                self.money_invested = 0

            # Calculate total mark-to-market net worth each day
            self.tot_net_worth[index] = self.cur_own_stocks * close_price + self.current_cash

        # If a position remains open at the end of the simulation, record its mark-to-market profit
        if self.cur_own_stocks > 0 and len(self.data) > 0:
            final_close = self.data['Close'].iloc[-1]
            unrealized_profit = (self.cur_own_stocks * final_close) - self.money_invested
            self.trade_profits.append(unrealized_profit)

        # Maintain proper DatetimeIndex
        self.tot_net_worth = pd.Series(self.tot_net_worth)

    def cal_metrics(self):
        """
        Calculates core portfolio performance metrics:
        - Total Return (%)
        - Sharpe Ratio (annualized, 252 trading days)
        - Max Drawdown (%)
        - Win Rate (%)
        - Profit Factor
        - Buy & Hold Baseline Return (%)
        """
        profits = pd.Series(self.trade_profits) if len(self.trade_profits) > 0 else pd.Series([0])

        # Total return
        tot_returns = (self.tot_net_worth.iloc[-1] - self.starting_cash) / self.starting_cash

        # Sharpe ratio
        daily_returns = self.tot_net_worth.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        running_max = self.tot_net_worth.cummax()
        drawdown = (self.tot_net_worth - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        num_win = (profits > 0).sum()
        tot_trades = len(self.trade_profits)
        win_rate = (num_win / tot_trades) if tot_trades > 0 else 0.0

        # Profit factor
        tot_win = profits[profits > 0].sum()
        tot_lose = abs(profits[profits < 0].sum())
        if tot_lose > 0:
            profit_factor = tot_win / tot_lose
        elif tot_win > 0:
            profit_factor = np.inf
        else:
            profit_factor = 0.0

        # Buy & Hold baseline
        buy_hold = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0]

        return {
            "Total returns": tot_returns,
            "Sharpe Ratio": sharpe_ratio,
            "Max drawdown": max_drawdown,
            "Win rate": win_rate,
            "Profit factor": profit_factor,
            "Baseline score (buy-hold)": buy_hold
        }

    def plot_result(self, strategy_name="Strategy"):
        """Plots equity curve and price chart with active signal shading."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # 1. Equity Curve
        axes[0].plot(self.tot_net_worth.index, self.tot_net_worth.values, label=f"{strategy_name} Net Worth", color="blue", linewidth=1.5)
        axes[0].axhline(self.starting_cash, color="gray", linestyle="--", alpha=0.7, label="Initial Capital")
        axes[0].set_title(f"{strategy_name} - Equity Curve", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # 2. Price Chart with Trade Signals
        axes[1].plot(self.data.index, self.data['Close'].values, label="Asset Close Price", color="black", linewidth=1.2)
        axes[1].fill_between(
            self.data.index,
            self.data['Close'].min(),
            self.data['Close'].max(),
            where=(self.data['signal'] == 1),
            facecolor='green',
            alpha=0.25,
            label="In Trade (Long)"
        )
        axes[1].set_title("Asset Price with Strategy Positions", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Price ($)")
        axes[1].set_xlabel("Date")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_drawdown(self, strategy_name="Strategy"):
        """Plots drawdown severity curve over time."""
        running_max = self.tot_net_worth.cummax()
        drawdown = (self.tot_net_worth - running_max) / running_max

        plt.figure(figsize=(12, 4))
        plt.plot(drawdown.index, drawdown * 100, label="Drawdown", color="crimson", linewidth=1.2)
        plt.fill_between(drawdown.index, drawdown * 100, 0, facecolor="crimson", alpha=0.3)
        plt.title(f"{strategy_name} - Drawdown Severity Curve", fontsize=14, fontweight="bold")
        plt.ylabel("Drawdown (%)")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()
