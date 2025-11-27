# ==============================
# alchemy_of_imagination_rl_capital_multi_scan_watchlist.py
# Multi-Pair RL Paper Trading with Market Scanning & Live Watchlist
# ==============================

!pip install dash plotly requests pandas numpy

import time, threading
import numpy as np
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
import requests

# -----------------------------
# Configuration
# -----------------------------
API_KEY = "MtVOF7ynTKpKYkKD"  # Replace with your Capital.com API key
ACCOUNT_TYPE = "demo"
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "GBP/JPY", "AUD/USD"]
EQUITY_START = 10000.0
SCAN_WINDOW = 10
INTERVAL_SEC = 60

# -----------------------------
# Capital.com REST API Wrapper
# -----------------------------
class CapitalClient:
    BASE_URL = "https://demo-api.capital.com"

    def __init__(self, api_key):
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def get_candles(self, symbol, interval="H1", count=50):
        endpoint = f"{self.BASE_URL}/v1/prices/{symbol}/history"
        params = {"interval": interval, "count": count}
        r = requests.get(endpoint, headers=self.headers, params=params)
        if r.status_code != 200:
            raise Exception(f"Failed fetching candles: {r.text}")
        data = r.json()
        return [float(c["close"]) for c in data.get("candles", [])]

# -----------------------------
# RL Helper Functions
# -----------------------------
def compute_features(prices, window=10):
    prices = np.array(prices)
    if len(prices) < window + 1:
        window = len(prices) - 1
    returns = (prices[-window:] - prices[-window-1:-1]) / prices[-window-1:-1]
    returns = np.nan_to_num(returns)
    vol = np.std(returns)
    momentum = prices[-1] - prices[-window]
    trend = (prices[-1] - np.mean(prices[-window:])) / np.mean(prices[-window:])
    price_norm = prices[-1] / prices[0] - 1
    state = np.array([price_norm, momentum, vol, trend])
    return state

def compute_opportunity_score(prices, window=SCAN_WINDOW):
    state = compute_features(prices, window)
    score = abs(state[2]) * abs(state[1]) * abs(state[3])
    return score

# -----------------------------
# RL Trader
# -----------------------------
class RLTrader:
    def __init__(self, state_size=4, gamma=0.95, alpha=0.01, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
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
                np.random.uniform(0.01, 0.15),
                np.random.choice([1, -1]),
                np.random.rand()
            ], dtype=float)
        else:
            action = np.array(self.q_table[key], dtype=float)
        return action

    def learn(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        old_value = np.array(self.q_table.get(key, np.array(action)), dtype=float)
        next_value = np.array(self.q_table.get(next_key, np.array(action)), dtype=float)
        new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
        self.q_table[key] = new_value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -----------------------------
# Trading Simulator
# -----------------------------
class TradingSimulator:
    def __init__(self, client, pair, equity=EQUITY_START):
        self.client = client
        self.pair = pair
        self.equity = equity
        self.equity_curve = [equity]
        self.trades = []
        self.agent = RLTrader(state_size=4)
        self.open_position = None
        self.prices = []

    def step(self, price):
        self.prices.append(price)
        if len(self.prices) < 2:
            return

        state = compute_features(self.prices)
        action = self.agent.act(state)
        pos_size, direction, exit_prob = action
        pos_size = min(pos_size, 0.15)

        pnl = 0
        if self.open_position:
            unrealized = (price - self.open_position['entry_price']) * self.open_position['direction'] * \
                         (self.open_position['size'] * self.equity / self.open_position['entry_price'])
            if exit_prob > 0.5:
                pnl = unrealized
                self.equity += pnl
                self.trades.append({"pnl": pnl, "equity": self.equity,
                                    "direction": self.open_position['direction'],
                                    "size": self.open_position['size']})
                self.open_position = None
        else:
            self.open_position = {"direction": int(direction), "size": pos_size, "entry_price": price}

        self.equity_curve.append(self.equity)
        reward = self.equity_curve[-1] - self.equity_curve[-2]
        self.agent.learn(state, action, reward, state)

# -----------------------------
# Continuous Paper Trader
# -----------------------------
class ContinuousPaperTrader:
    def __init__(self, client, pairs):
        self.client = client
        self.pairs = pairs
        self.active_pair = pairs[0]
        self.simulators = {p: TradingSimulator(client, p) for p in pairs}
        self.lock = threading.Lock()
        self.manual_switch = None
        self.opportunity_scores = {p: 0 for p in pairs}

    def switch_pair(self, pair_name):
        with self.lock:
            if pair_name in self.pairs:
                self.manual_switch = pair_name
                print(f"Switched manually to {pair_name}")

    def run(self):
        while True:
            with self.lock:
                if self.manual_switch:
                    self.active_pair = self.manual_switch
                    self.manual_switch = None
                else:
                    # Auto: scan all pairs for opportunity
                    scores = []
                    for p, sim in self.simulators.items():
                        try:
                            prices = self.client.get_candles(p)
                            score = compute_opportunity_score(prices)
                            self.opportunity_scores[p] = score
                            scores.append((p, score))
                        except:
                            self.opportunity_scores[p] = 0
                            scores.append((p, 0))
                    self.active_pair = max(scores, key=lambda x: x[1])[0]

            # Trade step
            try:
                prices = self.client.get_candles(self.active_pair)
                sim = self.simulators[self.active_pair]
                for price in prices[-5:]:
                    sim.step(price)
            except Exception as e:
                print(f"Error trading {self.active_pair}: {e}")
            time.sleep(INTERVAL_SEC)

# -----------------------------
# Dashboard
# -----------------------------
def run_dashboard(trader):
    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("Multi-Pair RL Paper Trading with Market Scanning & Watchlist"),
        html.Div(id="current-pair"),
        html.Div([html.Button(p, id=f"btn-{p}", n_clicks=0) for p in PAIRS]),
        dcc.Graph(id="equity-graph"),
        html.H4("Pair Opportunity Scores"),
        html.Div(id="watchlist"),
        dcc.Interval(id="interval-update", interval=2000, n_intervals=0)
    ])

    # Manual switching callbacks
    for p in PAIRS:
        @app.callback(
            Output("current-pair", "children"),
            Input(f"btn-{p}", "n_clicks"),
            State("current-pair", "children"),
            prevent_initial_call=True
        )
        def manual_switch(n_clicks, current, pair=p):
            if n_clicks > 0:
                trader.switch_pair(pair)
            return f"Active Pair: {trader.active_pair}"

    # Update equity graph
    @app.callback(
        Output("equity-graph", "figure"),
        Input("interval-update", "n_intervals")
    )
    def update_graph(n):
        sim = trader.simulators[trader.active_pair]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=sim.equity_curve, mode="lines+markers",
                                 name=f"Equity - {trader.active_pair}", line=dict(color="blue")))
        return fig

    # Update watchlist with opportunity scores
    @app.callback(
        Output("watchlist", "children"),
        Input("interval-update", "n_intervals")
    )
    def update_watchlist(n):
        items = [html.Div(f"{p}: {trader.opportunity_scores[p]:.6f}") for p in PAIRS]
        return items

    app.run(host="0.0.0.0", port=8050, debug=False)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    client = CapitalClient(API_KEY)
    trader = ContinuousPaperTrader(client, PAIRS)

    t = threading.Thread(target=trader.run)
    t.start()

    run_dashboard(trader)
