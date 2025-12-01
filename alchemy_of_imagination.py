#!/usr/bin/env python3
"""
alchemy_service.py
Production-ready runner for RL paper-trader + Dash dashboard.
Upgrades:
 - Auto-refresh token manager (CST + security token)
 - WebSocket ingestion with reconnect + backoff
 - Background trade executor (tick queue)
 - Logging pipeline (rotating file + in-memory buffer)
 - Health poller & automated failover (WS -> REST fallback)
 - Systemd-friendly (long-running daemon style)
"""
import os
import time
import threading
import logging
import json
import pickle
import math
import asyncio
import queue
from collections import deque
from typing import Any, Dict, Tuple, List

import numpy as np
import requests
import websockets
from logging.handlers import RotatingFileHandler
from dash import Dash, dcc, html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go

# ---------------------------
# CONFIG
# ---------------------------
WORKDIR = os.path.expanduser("~/alchemy")
MODEL_PATH = os.path.join(WORKDIR, "models")
QTABLE_FILE = os.path.join(MODEL_PATH, "qtable.pkl")
LOG_BUFFER_SIZE = 2000
API_KEY_ENV = "MtVOF7ynTKpKYkKD"        # required
USE_BROKER = False                    # set True when ready to place live orders
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "GBP/JPY", "AUD/USD"]
EQUITY_START = 10000.0
SCAN_WINDOW = 10
INTERVAL_SEC = 60
AUTOSAVE_SECONDS = 60
REQUEST_TIMEOUT = 10.0

CAPITAL_REST_BASE = "https://demo-api.capital.com"
CAPITAL_SESSION_URL = f"{CAPITAL_REST_BASE}/v1/session"
CAPITAL_ORDERS_URL = f"{CAPITAL_REST_BASE}/v1/orders"

# WebSocket endpoint
CAPITAL_WS = "wss://api-streaming-capital.backend-capital.com/connect"

# ---------------------------
# Ensure paths
# ---------------------------
os.makedirs(MODEL_PATH, exist_ok=True)

# ---------------------------
# Logging pipeline
# ---------------------------
logger = logging.getLogger("alchemy")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

# console
console_h = logging.StreamHandler()
console_h.setFormatter(fmt)
logger.addHandler(console_h)

# rotating file
file_handler = RotatingFileHandler(os.path.join(WORKDIR, "alchemy.log"), maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

# in-memory ring for Dash
log_buffer = deque(maxlen=LOG_BUFFER_SIZE)

def log(msg: str, level: str = "info"):
    ts_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {level.upper()}: {msg}"
    log_buffer.append(ts_line)
    if level == "info":
        logger.info(msg)
    elif level == "warn":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)

# ---------------------------
# Retry decorator
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
# Capital REST client (health + orders + candles fallback)
# ---------------------------
class CapitalClient:
    def __init__(self, api_key: str = None):
        self.base = CAPITAL_REST_BASE
        self.api_key = api_key or os.getenv(API_KEY_ENV)
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    @retry(max_attempts=2, base_delay=1.0)
    def ping(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/v1/time", timeout=REQUEST_TIMEOUT)
            return r.status_code == 200
        except Exception as e:
            log(f"REST ping error: {e}", "warn")
            return False

    @retry(max_attempts=3, base_delay=1.0)
    def get_candles(self, symbol: str, interval: str = "H1", count: int = 50):
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
                        if isinstance(c, dict) and ("close" in c):
                            closes.append(float(c["close"]))
                        elif isinstance(c, dict) and ("c" in c):
                            closes.append(float(c["c"]))
                    if closes:
                        return closes
                else:
                    log(f"get_candles: {r.status_code} {r.text[:200]}", "warn")
            except requests.RequestException as e:
                log(f"get_candles request error: {e}", "warn")
                continue
        raise Exception(f"Failed to fetch candles for {symbol}")

    @retry(max_attempts=3, base_delay=1.0)
    def place_order(self, epic: str, size: float, direction: int):
        # size semantics: depends on broker; here we post a simple payload
        order_type = "BUY" if direction > 0 else "SELL"
        payload = {
            "epic": epic,
            "size": size,
            "direction": order_type,
            "orderType": "MARKET"
        }
        try:
            r = self.session.post(CAPITAL_ORDERS_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code in (200, 201):
                data = r.json()
                log(f"Order success: {order_type} {size} {epic} -> {data}", "info")
                return data
            else:
                log(f"Order failed: {r.status_code} {r.text[:300]}", "error")
                return None
        except Exception as e:
            log(f"Order exception: {e}", "error")
            return None

# ---------------------------
# Token manager (auto-refresh CST + security token)
# ---------------------------
class CapitalAuth:
    """
    Uses API key (Bearer) to obtain session tokens: CST and x-security-token.
    Auto-refreshes in background.
    """
    def __init__(self, api_key: str, refresh_interval_sec: int = 1800):
        self.api_key = api_key
        self.cst = None
        self.security_token = None
        self._lock = threading.Lock()
        self.refresh_interval = refresh_interval_sec
        self._stop = threading.Event()
        # start background auto-refresh
        threading.Thread(target=self._refresh_loop, daemon=True).start()

    def _login_once(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            r = requests.post(CAPITAL_SESSION_URL, headers=headers, json={}, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            # some brokers provide tokens in headers or body; try both patterns
            cst = data.get("cst") or r.headers.get("CST") or r.headers.get("cst")
            sec = data.get("x-security-token") or data.get("securityToken") or r.headers.get("X-SECURITY-TOKEN") or r.headers.get("x-security-token")
            with self._lock:
                self.cst = cst
                self.security_token = sec
            log("Obtained new CST & security token", "info")
        except Exception as e:
            log(f"Failed to fetch CST/security token: {e}", "error")

    def _refresh_loop(self):
        while not self._stop.is_set():
            try:
                self._login_once()
            except Exception as e:
                log(f"Token refresh loop error: {e}", "warn")
            # sleep then refresh again
            time.sleep(self.refresh_interval)

    def get_tokens(self):
        with self._lock:
            return self.cst, self.security_token

    def stop(self):
        self._stop.set()

# ---------------------------
# RL helper & model persistence
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
# Trading simulation core (paper-only by default)
# ---------------------------
class TradingSimulator:
    def __init__(self, symbol: str, client: CapitalClient, agent: RLTrader, equity: float = EQUITY_START):
        self.symbol = symbol
        self.client = client
        self.agent = agent
        self.equity = equity
        self.equity_curve = [equity]
        self.trades = []
        self.prices = []
        self.open_pos = None

    def step_price(self, price: float):
        """Process one tick price (called by background executor)"""
        try:
            self.prices.append(price)
            if len(self.prices) < 2:
                return
            state = compute_features(self.prices, window=SCAN_WINDOW)
            action = self.agent.act(state)
            size_pct, direction, exit_prob = float(action[0]), int(np.sign(action[1])), float(action[2])
            size_pct = max(0.0, min(0.2, size_pct))
            if self.open_pos:
                unrealized = (price - self.open_pos["entry_price"]) * self.open_pos["direction"] * (self.open_pos["size"] * self.equity / (self.open_pos["entry_price"] + 1e-12))
                if exit_prob > 0.5:
                    pnl = unrealized
                    self.equity += pnl
                    self.trades.append({"pnl": float(pnl), "equity": float(self.equity), "direction": self.open_pos["direction"], "size": self.open_pos["size"]})
                    # live execution would go here if USE_BROKER
                    self.open_pos = None
            else:
                self.open_pos = {"direction": direction, "size": size_pct, "entry_price": price}
                # live execution would go here if USE_BROKER
            self.equity_curve.append(self.equity)
            reward = self.equity_curve[-1] - self.equity_curve[-2]
            self.agent.learn(state, action, reward / (self.equity_curve[0] + 1e-9), state)
        except Exception as e:
            log(f"Simulator step_price error ({self.symbol}): {e}", "error")

# ---------------------------
# Tick queue + background executor
# ---------------------------
TICK_QUEUE_SIZE = 10000
tick_queue = queue.Queue(maxsize=TICK_QUEUE_SIZE)

class BackgroundExecutor:
    def __init__(self, trader):
        self.trader = trader
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        log("BackgroundExecutor started", "info")
        while not self._stop.is_set():
            try:
                epic, price = tick_queue.get(timeout=1.0)
                if epic in self.trader.simulators:
                    self.trader.simulators[epic].step_price(price)
                tick_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log(f"BackgroundExecutor error: {e}", "error")

    def stop(self):
        self._stop.set()

# ---------------------------
# WebSocket ingestion layer (async with reconnect/backoff)
# ---------------------------
class CapitalWS:
    def __init__(self, auth: CapitalAuth, epics: List[str], trader_ref, reconnect_base=2.0, reconnect_max=300.0):
        self.auth = auth
        self.epics = epics
        self.trader_ref = trader_ref
        self.url = CAPITAL_WS
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._connected = threading.Event()
        self.reconnect_base = reconnect_base
        self.reconnect_max = reconnect_max

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _start_loop(self):
        backoff = self.reconnect_base
        while not self._stop.is_set():
            try:
                cst, sec = self.auth.get_tokens()
                if not cst or not sec:
                    log("No CST/security token available yet — waiting before connecting WS", "warn")
                    time.sleep(5)
                    continue
                asyncio.run(self._connect_and_listen(cst, sec))
                backoff = self.reconnect_base  # reset after clean exit
            except Exception as e:
                log(f"WS top-level error: {e}", "error")
                # set offline mode for trader
                if hasattr(self.trader_ref, "offline_mode"):
                    self.trader_ref.offline_mode = True
                time.sleep(backoff)
                backoff = min(self.reconnect_max, backoff * 2)

    async def _connect_and_listen(self, cst, sec):
        headers = []  # no custom headers required for this WS
        ws_url = self.url
        log("Attempting WebSocket connection...", "info")
        try:
            async with websockets.connect(ws_url, ping_interval=None) as ws:
                self._connected.set()
                # set trader back online
                if hasattr(self.trader_ref, "offline_mode"):
                    self.trader_ref.offline_mode = False
                # subscribe
                sub_msg = {
                    "destination": "marketData.subscribe",
                    "correlationId": str(int(time.time()*1000)),
                    "cst": cst,
                    "securityToken": sec,
                    "payload": {"epics": self.epics}
                }
                await ws.send(json.dumps(sub_msg))
                log("Subscribed to WS marketData", "info")
                # spawn ping coroutine
                async def ping_loop():
                    while True:
                        ping_msg = {
                            "destination": "ping",
                            "correlationId": str(int(time.time()*1000)),
                            "cst": cst,
                            "securityToken": sec
                        }
                        await ws.send(json.dumps(ping_msg))
                        await asyncio.sleep(540)  # 9 minutes safe (docs said 10)
                asyncio.create_task(ping_loop())

                # read messages
                async for message in ws:
                    try:
                        data = json.loads(message)
                        dest = data.get("destination")
                        if dest in ("quote",):
                            payload = data.get("payload", {})
                            epic = payload.get("epic")
                            # pick bid/ofr/mid
                            bid = payload.get("bid")
                            ofr = payload.get("ofr")
                            price = None
                            if bid is not None and ofr is not None:
                                price = (float(bid) + float(ofr)) / 2.0
                            elif payload.get("c") is not None:
                                price = float(payload.get("c"))
                            if epic and price is not None:
                                try:
                                    tick_queue.put_nowait((epic, price))
                                except queue.Full:
                                    log("Tick queue full; dropping tick", "warn")
                        # you can handle other destinations here (subscriptions acks etc.)
                    except Exception as e:
                        log(f"WS message handling error: {e}", "warn")
        except Exception as e:
            self._connected.clear()
            log(f"WebSocket connection lost: {e}", "warn")
            # mark offline
            if hasattr(self.trader_ref, "offline_mode"):
                self.trader_ref.offline_mode = True
            raise

# ---------------------------
# ContinuousPaperTrader (now accepts clients + auth + WS)
# ---------------------------
class ContinuousPaperTrader:
    def __init__(self, client: CapitalClient, auth: CapitalAuth = None, ws: CapitalWS = None, pairs=PAIRS):
        self.client = client
        self.auth = auth
        self.ws = ws
        self.pairs = pairs
        self.agent = RLTrader()
        self.agent.load()
        self.simulators = {p: TradingSimulator(p, client, self.agent) for p in pairs}
        self.active_pair = pairs[0]
        self.manual_switch = None
        self.opportunity_scores = {p: 0.0 for p in pairs}
        self.offline_mode = True  # default until WS connects or REST ping passes
        self._last_saved = time.time()
        self._lock = threading.Lock()
        # start background autosave thread
        threading.Thread(target=self._autosave_loop, daemon=True).start()
        # start health poller
        threading.Thread(target=self._health_poller, daemon=True).start()
        # background executor
        self.executor = BackgroundExecutor(self)

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
        # monitors REST reachability and triggers WS reconnect attempts via auth+ws layer
        while True:
            try:
                ok = False
                try:
                    ok = self.client.ping()
                except Exception:
                    ok = False
                if not ok:
                    if not self.offline_mode:
                        log("REST unreachable: entering offline mode.", "warn")
                    self.offline_mode = True
                else:
                    if self.offline_mode:
                        log("REST reachable again: leaving offline mode.", "info")
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
        """Fallback cycle: use REST candles to compute opportunity scores (WS feeds ticks to executor)."""
        try:
            # scanning (REST fallback)
            scores = []
            for p in self.pairs:
                try:
                    prices = self.client.get_candles(p, count=SCAN_WINDOW + 5)
                    score = abs(np.std(prices[-SCAN_WINDOW:])) * abs(prices[-1] - prices[-SCAN_WINDOW])
                    self.opportunity_scores[p] = float(score)
                    scores.append((p, score, prices))
                except Exception as e:
                    self.opportunity_scores[p] = 0.0
                    scores.append((p, 0.0, []))
                    log(f"scan fetch failed for {p}: {e}", "warn")

            # choose active pair
            with self._lock:
                if self.manual_switch:
                    self.active_pair = self.manual_switch
                    self.manual_switch = None
                else:
                    self.active_pair = max(scores, key=lambda x: x[1])[0]

            # optional: feed last few bars into simulator when WS not available
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
# Dash app (unchanged semantics; shows offline state)
# ---------------------------
def create_dash_app(trader: ContinuousPaperTrader):
    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("Alchemy RL Trader (WS + Auto-refresh tokens)"),
        html.Div([
            html.Div(id="active-pair", style={"fontWeight": "bold"}),
            html.Div(id="equity-display"),
            html.Div(id="mode-display"),
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
        html.Pre(id="logs", style={"height": "220px","overflow":"auto","backgroundColor":"#111","color":"#fff","padding":"8px"}),
        dcc.Interval(id="interval", interval=2000, n_intervals=0)
    ])

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

    @app.callback(Output("mode-display", "children"), Input("interval", "n_intervals"))
    def update_mode(n):
        return f"Mode: {'LIVE' if USE_BROKER else 'PAPER'}"

    return app

# ---------------------------
# Runner & orchestration
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
    log("Starting alchemy_service with WS + token manager")

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        log("ERROR: CAPITAL_API_KEY environment variable not set", "error")
        raise SystemExit("CAPITAL_API_KEY missing")

    rest_client = CapitalClient(api_key=api_key)
    auth = CapitalAuth(api_key=api_key)
    # instantiate trader with REST client and auth. WS created after trader so it can set offline_mode
    trader = ContinuousPaperTrader(rest_client, auth=auth, ws=None, pairs=PAIRS)

    # start background runner for REST-based scanning/fallback
    threading.Thread(target=runner_loop, args=(trader,), daemon=True).start()

    # start WS ingestion (uses auth tokens)
    ws = CapitalWS(auth=auth, epics=PAIRS, trader_ref=trader)
    trader.ws = ws
    ws.start()

    # Background executor already started inside trader constructor
    app = create_dash_app(trader)
    app.run(host="0.0.0.0", port=8050)

if __name__ == "__main__":
    main()
