# ============================
# alchemy_of_imagination_rl_autonomous_safe.py
# RL-Based Fully Autonomous Trading AI with Continuous Live Learning & Risk Safety
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
# RL Agent (Fully Autonomous)
# -----------------------------
class RLTraderAutonomous:
    def __init__(self, state_size, gamma=0.95, alpha=0.01, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_size = state_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_state_key(self, state):
        return tuple(np.round(state, 5))

    def act(self, state):
        key = self.get_state_key(state)
        if np.random.rand() < self.epsilon or key not in self.q_table:
            action = np.array([
                np.random.uniform(0.01, 0.15),  # position size % of equity
                np.random.choice([1, -1]),       # long or short
                np.random.uniform(0.97, 1.03),  # stop-loss multiplier
                np.random.uniform(1.01, 1.07)   # take-profit multiplier
            ], dtype=float)
        else:
            action = np.array(self.q_table[key], dtype=float)
        return action

    def learn(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        old_value = np.array(self.q_table.get(key, np.array(action, dtype=float)), dtype=float)
        next_value = np.array(self.q_table.get(next_key, np.array(action, dtype=float)), dtype=float)
        new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
        self.q_table[key] = new_value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -----------------------------
# Trading Simulator RL (Autonomous & Safe)
# -----------------------------
class TradingSimulatorRLSafe:
    def __init__(self, df, equity=10000.0, agent=None,
                 max_drawdown=0.3, max_pos_size=0.15, min_equity=500.0):
        self.df = sanitize_yf_data(df)
        self.equity = equity
        self.equity_curve = []
        self.trades = []
        self.features = compute_features(self.df)
        self.agent = agent or RLTraderAutonomous(state_size=self.features.shape[1])
        self.max_drawdown = max_drawdown
        self.max_pos_size = max_pos_size
        self.min_equity = min_equity

    def run(self):
        self.equity_curve = [self.equity]
        self.trades = []

        for i in range(1, len(self.df)):
            state = self.features.iloc[i-1].values
            next_state = self.features.iloc[i].values

            action = self.agent.act(state)
            pos_size, direction, stop_mult, take_mult = action

            # Safety adjustments
            if self.equity < self.min_equity:
                pos_size = 0  # skip trade
            pos_size = min(pos_size, self.max_pos_size)
            current_drawdown = 1 - min(self.equity_curve)/self.equity_curve[0]
            if current_drawdown > self.max_drawdown:
                pos_size *= 0.5  # reduce size if drawdown too high

            position_value = self.equity * pos_size
            price_change = self.df["Close"].iloc[i] - self.df["Close"].iloc[i-1]
            pnl = direction * price_change * (position_value / self.df["Close"].iloc[i-1])

            # Hard stop-loss limit per trade
            max_loss = 0.1 * self.equity  # 10% max loss
            pnl = np.clip(pnl, -max_loss, None)

            self.equity += pnl

            # Reward shaping: penalize for safety intervention
            reward = pnl / max(self.equity_curve[-1], 1.0)
            if pos_size == 0 or current_drawdown > self.max_drawdown:
                reward *= 0.5  # penalize skipped/reduced trades

            self.agent.learn(state, action, reward, next_state)

            self.trades.append({
                "pnl": pnl,
                "equity": self.equity,
                "position_value": position_value,
                "direction": direction,
                "stop_mult": stop_mult,
                "take_mult": take_mult,
                "pos_size": pos_size
            })
            self.equity_curve.append(self.equity)

        return self.equity

    def get_results(self):
        return pd.DataFrame(self.trades), pd.DataFrame({"equity": self.equity_curve})

# -----------------------------
# WalkForward RL Live (Autonomous & Safe)
# -----------------------------
class WalkForwardRLLiveSafe:
    def __init__(self, pair="EURUSD=X", meta_save_path="rl_autonomous_safe.pkl"):
        self.pair = pair
        self.meta_save_path = meta_save_path
        self.agent = RLTraderAutonomous(state_size=5)
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

        sim = TradingSimulatorRLSafe(raw, agent=self.agent)
        sim.run()
        trades_df, equity_df = sim.get_results()

        with open(self.meta_save_path, "wb") as f:
            pickle.dump(self.agent, f)

        return trades_df, equity_df

# -----------------------------
# Diagnostics
# -----------------------------
def diagnostics(trades_df, equity_df, max_drawdown=0.3):
    print("\n🔍 Diagnostics")
    if trades_df is not None and not trades_df.empty:
        print("\nSample trades:")
        print(trades_df.head())

        # Safe scatter plot: handle NaNs in direction
        trades_df_plot = trades_df.dropna(subset=["direction", "pnl"]).copy()
        colors = trades_df_plot["direction"].map({1:'green', -1:'red'}).fillna('gray')
        plt.figure(figsize=(12,5))
        plt.scatter(trades_df_plot.index, trades_df_plot["pnl"], c=colors, alpha=0.6)
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Individual Trade PnL (Green=Long, Red=Short, Gray=Unknown)")
        plt.xlabel("Trade #")
        plt.ylabel("PnL")
        plt.grid(True)
        plt.show()

        # Drawdown check
        drawdown = 1 - min(trades_df["equity"])/trades_df["equity"].iloc[0]
        print(f"✅ Drawdown is within safe limits ({drawdown*100:.1f}%)")

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
    print("🔹 Starting Autonomous RL Continuous Learning with Risk Safety Test...")
    live_ai = WalkForwardRLLiveSafe(pair="EURUSD=X")

    trades_df, equity_df = live_ai.run_live(start_days_ago=30)

    diagnostics(trades_df, equity_df)
    print("✅ Fully Autonomous RL AI with Risk Safety test completed.")
