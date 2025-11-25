# ============================
# alchemy_of_imagination.py
# ============================
import numpy as np
import pandas as pd
import yfinance as yf
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
import warnings
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

def features_from_fold(params, df):
    """Feature generator for MetaBrain."""
    em, sm, ap = params
    returns = df["Close"].pct_change().dropna()
    vol = returns.rolling(20).std().mean()
    trend = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
    kurt = returns.kurtosis()
    skew = returns.skew()
    return [em, sm, ap, vol, trend, kurt, skew]

# -----------------------------
# Trading Simulator (Advanced)
# -----------------------------
class TradingSimulator:
    def __init__(self, df, entry_mult=0.5, stop_mult=1.0, atr_period=14,
                 regime_stop_map=None, rng_seed=None):
        self.df = sanitize_yf_data(df)
        self.entry_mult = entry_mult
        self.stop_mult = stop_mult
        self.atr_period = atr_period
        self.regime_stop_map = regime_stop_map or {"calm_bull": 1.0, "volatile_bear": 1.3}
        self.rng = np.random.default_rng(rng_seed)
        self.equity_curve = []
        self.trades = []

    def calc_atr(self, df):
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift(1))
        low_close = np.abs(df["Low"] - df["Close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(self.atr_period, min_periods=1).mean()

    def run(self):
        df = self.df.copy()
        df["atr"] = self.calc_atr(df)
        equity = 10000.0
        self.equity_curve = [equity]
        trades = []

        for i in range(1, len(df)):
            atr = df["atr"].iloc[i]
            regime = df["regime_name"].iloc[i] if "regime_name" in df.columns else "calm_bull"
            close_prev = float(df["Close"].iloc[i - 1])
            close_curr = float(df["Close"].iloc[i])
            if np.isnan(close_prev) or np.isnan(close_curr): continue
            entry_level = close_prev * (1 + self.entry_mult * 0.01)
            stop_level = close_prev * (1 - self.stop_mult * 0.01 * self.regime_stop_map.get(regime, 1.0))
            signal = self.rng.choice([-1, 1])
            pnl = signal * (close_curr - close_prev)
            pnl -= 0.0001 * close_curr  # friction cost
            equity += pnl
            trades.append(pnl)
            self.equity_curve.append(equity)

        self.trades = pd.DataFrame({"pnl": trades})
        self.equity_df = pd.DataFrame({"equity": self.equity_curve})
        return equity

    def get_results(self):
        return self.equity_df, self.trades

# -----------------------------
# Evolutionary Optimizer (Adaptive)
# -----------------------------
class EvolutionaryOptimizer:
    def __init__(self, param_bounds, base_intensity=0.3, decay_rate=0.95, min_intensity=0.05):
        self.param_bounds = param_bounds
        self.intensity = base_intensity
        self.decay_rate = decay_rate
        self.min_intensity = min_intensity
        self.generation = 0

    def mutate(self, parent):
        if isinstance(parent, dict):
            child = parent.copy()
            for k, (low, high) in self.param_bounds.items():
                val = parent[k]
                mutation = np.random.normal(0, self.intensity * (high - low) * 0.1)
                child[k] = np.clip(val + mutation, low, high)
        else:
            child = []
            for i, (low, high) in enumerate(self.param_bounds.values()):
                val = parent[i]
                mutation = np.random.normal(0, self.intensity * (high - low) * 0.1)
                child.append(np.clip(val + mutation, low, high))
            child = tuple(child)
        return child

    def update_intensity(self, feedback=None):
        if feedback is not None and feedback < 0:
            self.intensity = min(1.0, self.intensity * 1.1)
        else:
            self.intensity = max(self.min_intensity, self.intensity * self.decay_rate)
        self.generation += 1

# -----------------------------
# MetaBrain (Hybrid Learner)
# -----------------------------
class MetaBrain:
    def __init__(self):
        self.X, self.y = [], []
        self.rf = None
        self.sgd = SGDRegressor(max_iter=500, tol=1e-3)

    def append(self, features, score):
        self.X.append(features)
        self.y.append(score)

    def batch_train(self):
        if len(self.X) < 5: return
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf.fit(self.X, self.y)

    def online_update_sgd(self, X_new, y_new, n_iter=5):
        if len(X_new) == 0: return
        for _ in range(n_iter):
            self.sgd.partial_fit(X_new, y_new)

    def predict(self, X):
        if self.rf is None: return [0]*len(X)
        preds = self.rf.predict(X)
        if hasattr(self.sgd, "predict"):
            preds += self.sgd.predict(X)
        return preds

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

# -----------------------------
# WalkForward Orchestrator
# -----------------------------
class WalkForwardOrchestrator:
    def __init__(self, pair="EURUSD=X", start="2018-01-01", end="2024-06-30",
                 entry_mults=[0.3,0.5,0.7], stop_mults=[1.0,1.5,2.0], atr_periods=[7,14,21],
                 train_months=9, test_months=3, n_candidates=12, elite_keep=3, evolve_offspring=6,
                 use_meta_after=3, verbose=True, meta_save_path="meta_brain.pkl"):
        self.pair = pair
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.param_bounds = {
            "entry_mult": (min(entry_mults), max(entry_mults)),
            "stop_mult": (min(stop_mults), max(stop_mults)),
            "atr_period": (min(atr_periods), max(atr_periods))
        }
        self.ep = EvolutionaryOptimizer(self.param_bounds)
        self.meta = MetaBrain()
        self.train_months = train_months
        self.test_months = test_months
        self.n_candidates = n_candidates
        self.elite_keep = elite_keep
        self.evolve_offspring = evolve_offspring
        self.use_meta_after = use_meta_after
        self.verbose = verbose
        self.meta_save_path = meta_save_path

    def _sample_param_tuple(self):
        pb = self.param_bounds
        return (float(np.random.uniform(*pb["entry_mult"])),
                float(np.random.uniform(*pb["stop_mult"])),
                int(round(np.random.uniform(*pb["atr_period"]))))

    def run(self):
        raw = yf.download(self.pair, start=self.start, end=self.end, progress=False, auto_adjust=True)
        raw = sanitize_yf_data(raw)[["Open","High","Low","Close"]]
        if raw.empty: raise ValueError("No market data downloaded.")
        results, combined_equities, all_trades = [], [], []

        train_start = raw.index.min()
        fold = 0
        while True:
            train_end = train_start + pd.DateOffset(months=self.train_months) - pd.DateOffset(days=1)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=self.test_months) - pd.DateOffset(days=1)
            if test_end > raw.index.max(): break
            fold += 1
            df_train = raw.loc[(raw.index >= train_start) & (raw.index <= train_end)].copy()
            df_test = raw.loc[(raw.index >= test_start) & (raw.index <= test_end)].copy()
            if df_train.empty or df_test.empty:
                train_start += pd.DateOffset(months=self.test_months)
                continue

            # regime tagging
            vol30 = df_train["Close"].pct_change().rolling(20, min_periods=1).std().fillna(0)
            median_vol = float(vol30.median()) if len(vol30) > 0 else 0.0
            df_train["regime_name"] = np.where(df_train["Close"].pct_change().rolling(20).std() < median_vol, "calm_bull", "volatile_bear")
            df_test["regime_name"]  = np.where(df_test["Close"].pct_change().rolling(20).std() < median_vol, "calm_bull", "volatile_bear")

            # candidate selection
            param_grid = [self._sample_param_tuple() for _ in range(self.n_candidates)]
            evaluated = []
            for params in param_grid:
                sim = TradingSimulator(df_train, entry_mult=params[0], stop_mult=params[1], atr_period=params[2])
                sim.run()
                _, trades = sim.get_results()
                pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
                dd = float((trades["pnl"].cumsum().cummax() - trades["pnl"].cumsum()).max()) if not trades.empty else 0.0
                score = pnl / (dd + 1e-9) if dd > 0 else pnl
                evaluated.append((score, params, trades))

            best_score, best_params, best_trades = max(evaluated, key=lambda x: x[0])
            feats = features_from_fold(best_params, df_train)
            self.meta.append(feats, best_score)
            self.meta.batch_train()
            self.meta.online_update_sgd([feats], [best_score], n_iter=4)

            results.append({"fold": fold, "params": best_params, "train_score": best_score})
            all_trades.extend([{"fold": fold, **t} for t in best_trades.to_dict("records")])
            combined_equities.append(best_trades["pnl"].cumsum())

            # evolve param grid
            elites = sorted([p for _,p,_ in evaluated], key=lambda x: x[0])[:self.elite_keep]
            for _ in range(self.evolve_offspring):
                child = self.ep.mutate(random.choice(elites))
                param_grid.append(child)

            train_start += pd.DateOffset(months=self.test_months)

        # save meta
        if self.meta_save_path:
            self.meta.save(self.meta_save_path)

        return pd.DataFrame(results), pd.concat(combined_equities, ignore_index=True), pd.DataFrame(all_trades), self.meta

# -----------------------------
# Diagnostics
# -----------------------------
def diagnostics(results_df, combined_eq, trades_df):
    print("\n🔍 Diagnostics")
    if not results_df.empty:
        print("\nFold summary:")
        display(results_df.head())
        print("\nFold score stats:")
        print(results_df[["fold","train_score"]].describe())
    if not combined_eq.empty:
        plt.figure(figsize=(10,4)); plt.plot(combined_eq.values); plt.title("Combined OOS Equity"); plt.grid(True); plt.show()
    if not trades_df.empty and "pnl" in trades_df.columns:
        plt.figure(figsize=(10,4)); plt.hist(trades_df["pnl"].dropna(), bins=30); plt.title("Trade PnL Distribution"); plt.show()
