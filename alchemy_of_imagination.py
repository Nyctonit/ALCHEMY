# ============================
# alchemy_of_imagination_rl_autonomous.py
# Autonomous RL-Based Trading AI with Continuous Live Learning
# ============================

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# -----------------------------
# Helper Functions
# -----------------------------
def sanitize_yf_data(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].capitalize() for c in df.columns]
    else:
        df.columns = [str(c).capitalize() for c in df.columns]
    needed = ["Open", "High", "Low", "Close"]
    df = df[[c for c in needed if c in df.columns]].dropna()
    if df.empty:
        raise ValueError("Downloaded data is empty or missing required columns.")
    return df

def compute_features(df):
    returns = df["Close"].pct_change().fillna(0)
    vol = returns.rolling(20, min_periods=1).std()
    trend = df["Close"].pct_change(5).fillna(0)
    momentum = df["Close"].diff().fillna(0)
    return pd.DataFrame({
        "returns": returns,
        "vol": vol,
        "trend": trend,
        "momentum": momentum,
        "close": df["Close"]
    })

# -----------------------------
# RL Agent with Adaptive Learning
# -----------------------------
class RLTrader:
    def __init__(self, state_size, gamma=0.95, alpha=0.01,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 min_alpha=0.001, max_alpha=0.05):
        self.state_size = state_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.q_table = {}
        self.pnl_history = []

    def get_state_key(self, state):
        return tuple(np.round(state, 5))

    def act(self, state, recent_vol=0.02, equity=10000.0):
        key = self.get_state_key(state)
        if np.random.rand() < self.epsilon or key not in self.q_table:
            pos_size = np.clip(np.random.normal(0.05, recent_vol), 0.01, 0.2)
            action = {
                "pos_size": pos_size,
                "direction": np.random.choice([1, -1]),
                "stop_mult": np.random.uniform(0.97, 1.02),
                "take_mult": np.random.uniform(1.01, 1.08)
            }
        else:
            action = self.q_table[key].copy()
        return action

    def learn(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        old_value = np.array(list(action.values()))
        next_max = np.max(self.q_table.get(next_key, old_value))
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[key] = {k: v for k, v in zip(action.keys(), new_value)}
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Adaptive learning rate based on recent performance
        self.pnl_history.append(reward)
        if len(self.pnl_history) > 20:
            recent_mean = np.mean(self.pnl_history[-20:])
            self.alpha = np.clip(0.01 + (recent_mean / 1000), self.min_alpha, self.max_alpha)

# -----------------------------
# Trading Simulator RL (Live)
# -----------------------------
class TradingSimulatorRL:
    def __init__(self, df, equity=10000.0, agent=None):
        self.df = sanitize_yf_data(df)
        self.equity = equity
        self.equity_curve = []
        self.trades = []
        self.features = compute_features(self.df)
        self.agent = agent or RLTrader(state_size=self.features.shape[1])

    def run(self):
        self.equity_curve = [self.equity]
        self.trades = []

        for i in range(1, len(self.df)):
            state = self.features.iloc[i-1].values
            next_state = self.features.iloc[i].values
            recent_vol = self.features["vol"].iloc[max(0, i-20):i].mean()
            action = self.agent.act(state, recent_vol=recent_vol, equity=self.equity)

            pos_size = action["pos_size"]
            direction = action["direction"]
            stop_mult = action["stop_mult"]
            take_mult = action["take_mult"]

            position_value = self.equity * pos_size
            price_change = self.df["Close"].iloc[i] - self.df["Close"].iloc[i-1]
            pnl = direction * price_change * (position_value / self.df["Close"].iloc[i-1])
            self.equity += pnl

            reward = pnl - 0.01 * max(0, self.equity_curve[-1] - self.equity)
            self.agent.learn(state, action, reward, next_state)

            self.trades.append({
                "pnl": pnl,
                "equity": self.equity,
                "position_value": position_value,
                "direction": direction,
                "stop_mult": stop_mult,
                "take_mult": take_mult
            })
            self.equity_curve.append(self.equity)

        return self.equity

    def get_results(self):
        return pd.DataFrame(self.trades), pd.DataFrame({"equity": self.equity_curve})

# -----------------------------
# WalkForward RL Live Autonomous
# -----------------------------
class WalkForwardRLLive:
    def __init__(self, pair="EURUSD=X", meta_save_path="rl_autonomous_agent.pkl"):
        self.pair = pair
        self.meta_save_path = meta_save_path
        self.agent = RLTrader(state_size=5)
        if os.path.exists(meta_save_path):
            with open(meta_save_path, "rb") as f:
                self.agent = pickle.load(f)

    def run_live(self, start_days_ago=30):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=start_days_ago)

        raw = yf.download(self.pair, start=start_date.strftime("%Y-%m-%d"),
                          end=end_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        raw = sanitize_yf_data(raw)
        if raw.empty:
            print("No new market data.")
            return None, None

        sim = TradingSimulatorRL(raw, agent=self.agent)
        sim.run()
        trades_df, equity_df = sim.get_results()

        with open(self.meta_save_path, "wb") as f:
            pickle.dump(self.agent, f)

        return trades_df, equity_df

# -----------------------------
# Diagnostics
# -----------------------------
def diagnostics(trades_df, equity_df):
    print("\n🔍 Diagnostics")
    if trades_df is not None and not trades_df.empty:
        print("\nSample trades:")
        print(trades_df.head())
    if equity_df is not None and not equity_df.empty:
        plt.figure(figsize=(10,4))
        plt.plot(equity_df["equity"].values)
        plt.title("Equity Curve")
        plt.grid(True)
        plt.show()
    if trades_df is not None and not trades_df.empty:
        plt.figure(figsize=(10,4))
        plt.hist(trades_df["pnl"].dropna(), bins=30)
        plt.title("Trade PnL Distribution")
        plt.show()

# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":
    print("🔹 Starting Autonomous RL Continuous Learning Test...")
    live_ai = WalkForwardRLLive(pair="EURUSD=X")

    trades_df, equity_df = live_ai.run_live(start_days_ago=30)
    diagnostics(trades_df, equity_df)
    print("✅ Continuous RL Autonomous AI test completed.")
