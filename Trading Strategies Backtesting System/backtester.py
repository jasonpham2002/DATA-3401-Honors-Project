import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from pipeline import DataPipeline
import numpy as np
from strategy import Strategy, MACrossover, RSIMeanReversion
import matplotlib.pyplot as plt

class Backtest:
    def __init__(self, asset_data,strategy,starting_cash=100000):
        self.current_cash = starting_cash
        self.starting_cash = starting_cash
        self.trade_history = {}
        self.cur_own_stocks = 0
        self.money_invested = 0
        self.data = asset_data
        self.strat = strategy
        self.tot_net_worth = {}
        self.trade_profits = []

    def run(self):
        self.strat.generate_signal()
        self.strat.execute_trade()

        for index, row in self.data.iterrows():
            if row['position'] == 1:
                self.cur_own_stocks = self.current_cash // row['Close']
                self.current_cash -= self.cur_own_stocks * row['Close']
                self.money_invested = self.cur_own_stocks * row['Close']
                self.trade_history[f"{index}"] = {"Bought":self.cur_own_stocks,
                                                  "Money spent": row['Close']*self.cur_own_stocks}

            elif row['position'] == -1:
                self.current_cash += self.cur_own_stocks * row['Close']
                self.trade_history[f"{index}"] = {"Sold": self.cur_own_stocks,
                                                  "Money gained": row['Close'] * self.cur_own_stocks}
                profit = (self.cur_own_stocks * row['Close']) - (self.money_invested)
                self.trade_profits.append(profit)
                self.cur_own_stocks = 0


            #Calculate total net worth each day
            self.tot_net_worth[f"{index}"] = self.cur_own_stocks * row['Close'] + self.current_cash

        self.tot_net_worth = pd.Series(self.tot_net_worth)

    def cal_metrics(self):
        profits = pd.Series(self.trade_profits)

        #total returns
        tot_returns = (self.tot_net_worth.iloc[-1] - self.starting_cash)/self.starting_cash

        #sharpe ratio
        daily_returns = self.tot_net_worth.pct_change()
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())

        #max drawdown
        running_max = self.tot_net_worth.cummax()
        drawdown = (self.tot_net_worth - running_max)/ running_max
        max_drawndown = drawdown.min()

        #win rate
        num_win = (profits>0).sum()
        tot_trades = len(profits)
        win_rate = num_win/tot_trades

        #profit factor
        tot_win = profits[profits > 0].sum()
        tot_lose = profits[profits < 0].sum()
        profit_factor = (tot_win / tot_lose) * -1

        #baseline
        buy_hold = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[1]) / self.data['Close'].iloc[1]

        return {"Total returns": tot_returns, "Sharpe  Ratio": sharpe_ratio,
                "Max drawdown": max_drawndown, "Win rate": win_rate,
                "Profit factor": profit_factor, "Baseline score (buy-hold)": buy_hold}

    def plot_result(self):

        self.tot_net_worth.plot(title="Equity Curve", figsize=(10,6), ylabel="Portfolio Value")
        plt.show()

        self.data['Close'].plot(title="Asset Price with Trades", figsize=(10, 6), ylabel="Price")
        plt.fill_between(self.data.index,
                         self.data['Close'].min(),  # The bottom of the shading
                         self.data['Close'].max(),  # The top of the shading
                         where=(self.data['signal'] == 1),  # Condition
                         facecolor='green',
                         alpha=0.3)

        plt.show()


