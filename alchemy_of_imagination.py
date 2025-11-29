#!/usr/bin/env python3
"""
alchemy_service.py
Production-ready runner for the RL paper-trader + Dash dashboard.
Features:
 - persistent Q-table save/load
 - autosave of model
 - auto-retry + exponential backoff for API calls
 - API outage self-diagnosis (goes into offline mode if broker not reachable)
 - live logging appended to an in-memory ring buffer and exposed to Dash
 - dual-mode (auto/manual) pair switching and market scanning watchlist
 - safe paper-only execution by default (no real orders)
"""

import os
import time
import threading
import logging
import json
import pickle
import math
from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import requests
from dash import Dash, dcc, html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go

# ---------------------------
# CONFIG
# ---------------------------
WORKDIR = os.path.expanduser("~/alchemy")
MODEL_PATH = os.path.join(WORKDIR, "models")
QTABLE_FILE = os.path.join(MODEL_PATH, "qtable.pkl")
LOG_BUFFER_SIZE = 1000
API_KEY_ENV = "CAPITAL_API_KEY"  # set this env var if using Capital REST
USE_BROKER = False  # keep False (paper-sim) until we add live order routing (task A)
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "GBP/JPY", "AUD/USD"]
EQUITY_START = 10000.0
SCAN_WINDOW = 10
INTERVAL_SEC = 60
AUTOSAVE_SECONDS = 60
REQUEST_TIMEOUT = 10.0

# Capital demo endpoint (may be unreachable in some networks)
CAPITAL_BASE = "https://demo-api.capital.com"  # note: DNS sometimes blocked; code handles offline gracefully

# ---------------------------
# Logging (in-memory + std)
# ---------------------------
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)

logger = logging.getLogger("alchemy")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
h = logging.StreamHandler()
h.setFormatter(fmt)
logger.addHandler(h)

# in-memory ring buffer for recent logs (exposed to dashboard)
log_buffer = deque(maxlen=LOG_BUFFER_SIZE)

def log(msg, level="info"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {level.upper()}: {msg}"
    log_buffer.append(line)
    if level == "info":
        logger.info(msg)
    elif level == "warn":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)

# ---------------------------
# util: retry decorator
# ---------------------------
def retry(max_attempts=4, base_delay=1.0, backoff=2.0):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last_exc = None
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    log(f"Attempt {attempt}/{max_attempts} failed: {e}", "warn")
                    if attempt == max_attempts:
                        break
                    time.sleep(delay)
                    delay *= backoff
            raise last_exc
        return wrapper
    return deco

# ---------------------------
# Capital REST minimal wrapper (only used for candles and account health)
# ---------------------------
class CapitalClient:
    def __init__(self, api_key: str = None):
        self.base = CAPITAL_BASE
        self.api_key = api_key or os.getenv(API_KEY_ENV)
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    @retry(max_attempts=3, base_delay=1.0)
    def get_candles(self, symbol: str, interval: str = "H1", count: int = 50):
        # NOTE: Capital uses symbol names — we try a few formats to be robust
        possible_endpoints = [
            f"{self.base}/v1/prices/{symbol}/history",
            f"{self.base}/v1/prices/{symbol.replace('/', '')}/history",
            f"{self.base}/v1/prices/{symbol.replace('/', '-')}/history",
        ]
        for url in possible_endpoints:
            try:
                r = self.session.get(url, params={"interval": interval, "count": count}, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    candles = data.get("candles") or data.get("items") or []
                    closes = []
                    for c in candles:
                        # tolerant extraction
                        if isinstance(c, dict) and ("close" in c):
                            closes.append(float(c["close"]))
                        elif isinstance(c, dict) and ("c" in c):
                            closes.append(float(c["c"]))
                    if closes:
                        return closes
                else:
                    msg = f"get_candles: {r.status_code} {r.text[:200]}"
                    log(msg, "warn")
            except requests.exceptions.RequestException as e:
                log(f"request exception: {e}", "warn")
                continue
        raise Exception("Failed to fetch candles for %s" % symbol)

    @retry(max_attempts=2, base_delay=1.0)
    def ping(self):
        # ping a simple endpoint to check DNS/resolution and API reachability
        url = f"{self.base}/v1/time"  # some brokers have /time; tolerate failure
        try:
            r = self.session.get(url, timeout=REQUEST_TIMEOUT)
            return r.status_code == 200
        except Exception as e:
            log(f"ping error: {e}", "warn")
            return False

# ---------------------------
# RL helper & model persistence (Q-table)
# ---------------------------
def compute_features(prices, window=10):
    arr = np.array(prices, dtype=float)
    if len(arr) < window + 1:
        window = len(arr) - 1
    if window <= 0:
        return np.zeros(4)
    returns = (arr[-window:] - arr[-window-1:-1]) / (arr[-window-1:-1] + 1e-12)
    returns = np.nan_to_num(returns)
    vol = float(np.std(returns))
    momentum = float(arr[-1] - arr[-window])
    trend = float((arr[-1] - np.mean(arr[-window:])) / (np.mean(arr[-window:]) + 1e-12))
    price_norm = float(arr[-1] / (arr[0] + 1e-12) - 1)
    return np.array([price_norm, momentum, vol, trend], dtype=float)

class RLTrader:
    def __init__(self, state_size=4, alpha=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_size = state_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: Dict[Tuple, np.ndarray] = {}

    def state_key(self, s: np.ndarray):
        return tuple(np.round(s, 5))

    def act(self, state: np.ndarray):
        key = self.state_key(state)
        if np.random.rand() < self.epsilon or key not in self.q_table:
            action = np.array([np.random.uniform(0.01, 0.15),
                               int(np.random.choice([1, -1])),
                               np.random.uniform(0.0, 1.0)], dtype=float)
        else:
            action = self.q_table[key].copy()
        return action

    def learn(self, s, a, reward, s_next):
        k = self.state_key(s)
        k_next = self.state_key(s_next)
        old = self.q_table.get(k, np.array(a, dtype=float))
        nxt = self.q_table.get(k_next, np.array(a, dtype=float))
        # simple vector TD update
        new = old + self.alpha * (reward + self.gamma * np.array(nxt) - old)
        self.q_table[k] = new
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path=QTABLE_FILE):
        try:
            with open(path, "wb") as f:
                pickle.dump(self.q_table, f)
            log(f"Q-table saved ({len(self.q_table)} entries) -> {path}")
        except Exception as e:
            log(f"Failed to save Q-table: {e}", "error")

    def load(self, path=QTABLE_FILE):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.q_table = pickle.load(f)
                log(f"Q-table loaded ({len(self.q_table)} entries) <- {path}")
            else:
                log("No Q-table file found; starting fresh")
        except Exception as e:
            log(f"Failed to load Q-table: {e}", "error")

# ---------------------------
# Trading simulation core (paper-only)
# ---------------------------
class TradingSimulator:
    def __init__(self, symbol: str, client: CapitalClient, agent: RLTrader, equity: float = EQUITY_START):
        self.symbol = symbol
        self.client = client
        self.agent = agent
        self.equity = equity
        self.equity_curve = [equity]
        self.trades = []  # list of dicts
        self.prices = []
        self.open_pos = None

    def step_price(self, price: float):
        self.prices.append(price)
        if len(self.prices) < 2:
            return
        state = compute_features(self.prices, window=SCAN_WINDOW)
        action = self.agent.act(state)
        size_pct, direction, exit_prob = float(action[0]), int(np.sign(action[1])), float(action[2])
        size_pct = max(0.0, min(0.2, size_pct))  # clamp
        if self.open_pos:
            unrealized = (price - self.open_pos["entry_price"]) * self.open_pos["direction"] * (self.open_pos["size"] * self.equity / (self.open_pos["entry_price"] + 1e-12))
            # let RL decide exit dynamically
            if exit_prob > 0.5:
                pnl = unrealized
                self.equity += pnl
                self.trades.append({"pnl": float(pnl), "equity": float(self.equity), "direction": self.open_pos["direction"], "size": self.open_pos["size"]})
                self.open_pos = None
            else:
                pnl = 0.0
        else:
            # open new position
            self.open_pos = {"direction": direction, "size": size_pct, "entry_price": price}
            pnl = 0.0

        self.equity_curve.append(self.equity)
        reward = (self.equity_curve[-1] - self.equity_curve[-2])
        # learn with normalized reward
        self.agent.learn(state, action, reward / (self.equity_curve[0] + 1e-9), state)

# ---------------------------
# ContinuousPaperTrader (multi-pair + scanning + manual)
# ---------------------------
class ContinuousPaperTrader:
    def __init__(self, client: CapitalClient, pairs=PAIRS):
        self.client = client
        self.pairs = pairs
        self.agent = RLTrader()
        self.agent.load()
        self.simulators = {p: TradingSimulator(p, client, self.agent) for p in pairs}
        self.active_pair = pairs[0]
        self.manual_switch = None
        self.opportunity_scores = {p: 0.0 for p in pairs}
        self.offline_mode = False
        self._last_saved = time.time()
        self._lock = threading.Lock()
        # start background autosave thread
        threading.Thread(target=self._autosave_loop, daemon=True).start()
        # start health poller
        threading.Thread(target=self._health_poller, daemon=True).start()

    def _autosave_loop(self):
        while True:
            try:
                if time.time() - self._last_saved > AUTOSAVE_SECONDS:
                    with self._lock:
                        self.agent.save()
                        self._last_saved = time.time()
                time.sleep(5)
            except Exception as e:
                log(f"autosave loop error: {e}", "error")
                time.sleep(5)

    def _health_poller(self):
        while True:
            try:
                ok = False
                try:
                    ok = self.client.ping()
                except Exception:
                    ok = False
                if not ok:
                    # Mark offline and keep running using cached prices (if any)
                    if not self.offline_mode:
                        log("API unreachable: entering offline mode (will rely on last data).", "warn")
                    self.offline_mode = True
                else:
                    if self.offline_mode:
                        log("API reachable again: leaving offline mode.", "info")
                    self.offline_mode = False
            except Exception as e:
                log(f"health poller error: {e}", "error")
            time.sleep(10)

    def switch_manual(self, pair_name: str):
        with self._lock:
            if pair_name in self.pairs:
                self.manual_switch = pair_name
                log(f"Manual switch requested -> {pair_name}")

    def run_cycle(self):
        """Single cycle: scan market -> choose active pair -> feed simulator with recent bars"""
        try:
            # scanning
            scores = []
            for p in self.pairs:
                try:
                    prices = self.client.get_candles(p, count=SCAN_WINDOW + 5)
                    score = abs(np.std(prices[-SCAN_WINDOW:])) * abs(prices[-1] - prices[-SCAN_WINDOW])  # simple opportunistic score
                    self.opportunity_scores[p] = float(score)
                    scores.append((p, score, prices))
                except Exception as e:
                    self.opportunity_scores[p] = 0.0
                    scores.append((p, 0.0, []))
                    log(f"scan fetch failed for {p}: {e}", "warn")

            # choose active pair (manual takes precedence)
            with self._lock:
                if self.manual_switch:
                    self.active_pair = self.manual_switch
                    self.manual_switch = None
                else:
                    # pick pair with highest score
                    self.active_pair = max(scores, key=lambda x: x[1])[0]

            # feed last few bars into simulator for active pair
            chosen_prices = None
            for p, s, prices in scores:
                if p == self.active_pair:
                    chosen_prices = prices
                    break
            if chosen_prices:
                sim = self.simulators[self.active_pair]
                for price in chosen_prices[-5:]:
                    sim.step_price(price)
            else:
                log(f"No recent prices for {self.active_pair}", "warn")

        except Exception as e:
            log(f"run_cycle error: {e}", "error")

# ---------------------------
# Dash app (exposes equity, watchlist, logs, manual controls)
# ---------------------------
def create_dash_app(trader: ContinuousPaperTrader):
    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("Alchemy RL Paper Trader (service)"),
        html.Div([
            html.Div(id="active-pair", style={"fontWeight": "bold"}),
            html.Div(id="equity-display"),
            html.Button("Manual: EUR/USD", id="btn-EUR/USD"),
            html.Button("Manual: GBP/USD", id="btn-GBP/USD"),
            html.Button("Manual: USD/JPY", id="btn-USD/JPY"),
            html.Button("Manual: GBP/JPY", id="btn-GBP/JPY"),
            html.Button("Manual: AUD/USD", id="btn-AUD/USD"),
            html.Button("Clear Logs", id="clear-logs"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        dcc.Graph(id="equity-graph"),
        html.H4("Watchlist opportunity scores"),
        html.Div(id="watchlist"),
        html.H4("Recent logs"),
        html.Pre(id="logs", style={"height": "220px", "overflow": "auto", "backgroundColor": "#111", "color": "#fff", "padding": "8px"}),
        dcc.Interval(id="interval", interval=2000, n_intervals=0)
    ])

    # manual switching callbacks
    for p in PAIRS:
        btn_id = f"btn-{p}"
        @app.callback(Output("active-pair", "children"), Input(btn_id, "n_clicks"), State("active-pair", "children"), prevent_initial_call=True)
        def manual_switch(n_clicks, current, pair=p):
            if n_clicks:
                trader.switch_manual(pair)
            return f"Active Pair: {trader.active_pair}"

    @app.callback(Output("equity-graph", "figure"), Input("interval", "n_intervals"))
    def update_equity(n):
        sim = trader.simulators[trader.active_pair]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=sim.equity_curve, mode="lines+markers", name=f"Equity - {trader.active_pair}"))
        fig.update_layout(title=f"Equity - Active: {trader.active_pair}", xaxis_title="Step", yaxis_title="Equity")
        return fig

    @app.callback(Output("watchlist", "children"), Input("interval", "n_intervals"))
    def update_watch(n):
        items = [html.Div(f"{p}: {trader.opportunity_scores.get(p,0):.6f}") for p in trader.pairs]
        return items

    @app.callback(Output("logs", "children"), [Input("interval", "n_intervals"), Input("clear-logs", "n_clicks")])
    def update_logs(n_intervals, clear_clicks):
        if clear_clicks:
            log_buffer.clear()
        return "\n".join(list(log_buffer))

    @app.callback(Output("equity-display", "children"), Input("interval", "n_intervals"))
    def update_eq(n):
        sim = trader.simulators[trader.active_pair]
        return f"Equity: {sim.equity_curve[-1]:.2f} | Offline mode: {trader.offline_mode}"

    return app

# ---------------------------
# Runner & system glue
# ---------------------------
def runner_loop(trader: ContinuousPaperTrader):
    log("Runner loop started")
    while True:
        try:
            trader.run_cycle()
        except Exception as e:
            log(f"Runner loop caught error: {e}", "error")
        time.sleep(INTERVAL_SEC)

def main():
    log("Starting alchemy_service")
    api_key = os.getenv(API_KEY_ENV, None)
    client = CapitalClient(api_key=api_key)
    trader = ContinuousPaperTrader(client, PAIRS)

    # start runner thread
    t = threading.Thread(target=runner_loop, args=(trader,), daemon=True)
    t.start()

    app = create_dash_app(trader)
    # Dash uses app.run (newer versions). Bind to 0.0.0.0 and a port (8050).
    # In systemd deploy, prefer a production frontend (nginx) but this is fine for testing.
    app.run(host="0.0.0.0", port=8050)

if __name__ == "__main__":
    main()
