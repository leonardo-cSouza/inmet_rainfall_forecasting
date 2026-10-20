"""
The seven forecasting models and their hyperparameter searches.

Every search follows the paper's protocol: a 5-fold expanding-window split of
the TRAINING period (sklearn TimeSeriesSplit), the configuration with the
lowest mean validation RMSE (on the scaled log target) is kept, refitted on
the whole training period, and only then used on the test period. The test
set is never used for selection.

Each `run_*` function takes a StationData and returns (y_pred_mm, best_params).
"""
import itertools
import os
import time

import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

import config

# --------------------------------------------------------------------------
# Search spaces (Table 2 of the paper). QUICK grids are for smoke tests only.
# --------------------------------------------------------------------------
TREE_SPACES = {
    "Random Forest": {
        "n_estimators": randint(200, 1001), "max_depth": randint(3, 101),
        "min_samples_split": randint(2, 41), "min_samples_leaf": randint(1, 21),
        "max_features": ["sqrt", "log2", None],
    },
    "XGBoost": {
        "n_estimators": randint(200, 1001), "learning_rate": uniform(0.01, 0.29),
        "max_depth": randint(3, 13), "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 11),
    },
    "LightGBM": {
        "n_estimators": randint(200, 1001), "learning_rate": uniform(0.01, 0.29),
        "max_depth": randint(3, 16), "num_leaves": randint(10, 101),
        "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(10, 51),
    },
}

RC_GRID = {"units": [100, 300, 500, 1000], "sr": [0.5, 0.7, 0.9, 1.1, 1.2],
           "lr": [0.1, 0.3, 0.5], "ridge": [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]}
MLP_GRID = {"units": [32, 64, 128, 256], "dropout": [0.0, 0.1, 0.2],
            "layers": [1, 2, 3], "batch_size": [32, 64, 128]}
LSTM_GRID = {"lookback": [7, 14, 30], "layers": [1, 2, 3], "units": [32, 64, 128],
             "learning_rate": [1e-2, 1e-3, 1e-4], "dropout": [0.0, 0.2]}

# MEDIUM: reduced MLP/LSTM grids that fit all 46 stations in a few hours on a laptop
# (other models keep their full search spaces)
MEDIUM = {
    "MLP": {"units": [32, 64, 128], "dropout": [0.0, 0.2], "layers": [1, 2], "batch_size": [32, 64]},
    "LSTM": {"lookback": [7, 14, 30], "layers": [1], "units": [32, 64], "learning_rate": [1e-3],
             "dropout": [0.0, 0.2]},
}

QUICK = {
    "tree_iter": 5,
    "RC": {"units": [100, 300], "sr": [0.9], "lr": [0.3], "ridge": [1e-3, 1e-1]},
    "MLP": {"units": [64], "dropout": [0.0], "layers": [1, 2], "batch_size": [64]},
    "LSTM": {"lookback": [7], "layers": [1], "units": [32], "learning_rate": [1e-3],
             "dropout": [0.0]},
}


# CPU workers for the tree ensembles; lower it when stations run in parallel
N_JOBS = int(os.environ.get("INMET_N_JOBS", -1))


def search_space(name, size, full):
    """size: "full" (Table 2), "medium" (MLP/LSTM only) or "quick" (smoke tests)."""
    if size == "quick":
        return QUICK[name]
    if size == "medium" and name in MEDIUM:
        return MEDIUM[name]
    return full


def grid(space):
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*space.values())]


def folds(n_samples, n_folds=config.N_FOLDS):
    return list(TimeSeriesSplit(n_splits=n_folds).split(np.arange(n_samples)))


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a).ravel() - np.asarray(b).ravel()) ** 2)))


# --------------------------------------------------------------------------
# Tree ensembles: randomized search
# --------------------------------------------------------------------------
def run_tree(name, data, size="full"):
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(random_state=config.SEED, n_jobs=N_JOBS)
    elif name == "XGBoost":
        import xgboost as xgb
        estimator = xgb.XGBRegressor(objective="reg:squarederror", random_state=config.SEED,
                                     n_jobs=N_JOBS)
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(objective="regression", random_state=config.SEED,
                                  n_jobs=N_JOBS, verbose=-1)
    else:
        raise ValueError(name)

    search = RandomizedSearchCV(
        estimator, TREE_SPACES[name],
        n_iter=QUICK["tree_iter"] if size == "quick" else config.N_RANDOM_ITER,
        cv=TimeSeriesSplit(n_splits=config.N_FOLDS),
        scoring="neg_root_mean_squared_error",
        random_state=config.SEED, refit=True, error_score="raise")
    search.fit(data.X_train, data.y_train_scaled)
    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    return data.to_mm(search.best_estimator_.predict(data.X_test)), params


# --------------------------------------------------------------------------
# Reservoir Computing (Echo State Network, ReservoirPy)
# --------------------------------------------------------------------------
def _esn(p):
    from reservoirpy.nodes import Reservoir, Ridge
    return Reservoir(units=p["units"], sr=p["sr"], lr=p["lr"], seed=config.SEED) >> \
        Ridge(ridge=p["ridge"])


def run_rc(data, size="full"):
    X, y = data.X_train, data.y_train_scaled.reshape(-1, 1)
    best, best_score = None, np.inf
    for p in grid(search_space("RC", size, RC_GRID)):
        scores = []
        for tr, va in folds(len(X)):
            esn = _esn(p)
            esn.fit(X[tr], y[tr], warmup=min(config.RC_WARMUP, len(tr) // 4))
            # validation days directly follow the fold's training days, so the
            # reservoir state carries over as it would in deployment
            scores.append(_rmse(y[va], esn.run(X[va])))
        if np.mean(scores) < best_score:
            best, best_score = p, float(np.mean(scores))

    esn = _esn(best)
    esn.fit(X, y, warmup=config.RC_WARMUP)
    return data.to_mm(esn.run(data.X_test)), {**best, "cv_rmse_scaled": best_score, "grid": size}


# --------------------------------------------------------------------------
# Keras helpers (MLP and LSTM)
# --------------------------------------------------------------------------
def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.random.set_seed(config.SEED)
    # INMET_TF_THREADS limits TensorFlow's CPU threads when several stations
    # are trained in parallel processes
    threads = int(os.environ.get("INMET_TF_THREADS", 0))
    if threads:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:              # already initialised in this process
            pass
    return tf


def _fit_keras(model, X, y, batch_size, epochs=300, patience=30):
    """Fit with early stopping on the last 10% (chronological) of the given data."""
    from tensorflow.keras.callbacks import EarlyStopping
    cut = int(len(X) * 0.9)
    model.fit(X[:cut], y[:cut], validation_data=(X[cut:], y[cut:]),
              epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True)])
    return model


def _mlp(n_features, p):
    from tensorflow.keras import Input, Sequential
    from tensorflow.keras.layers import Dense, Dropout
    model = Sequential([Input(shape=(n_features,))])
    for _ in range(p["layers"]):
        model.add(Dense(p["units"], activation="relu"))
        if p["dropout"] > 0:
            model.add(Dropout(p["dropout"]))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def run_mlp(data, size="full"):
    tf = _keras()
    X, y = data.X_train, data.y_train_scaled
    best, best_score = None, np.inf
    for p in grid(search_space("MLP", size, MLP_GRID)):
        scores = []
        for tr, va in folds(len(X)):
            tf.keras.backend.clear_session()
            model = _fit_keras(_mlp(X.shape[1], p), X[tr], y[tr], p["batch_size"])
            scores.append(_rmse(y[va], model.predict(X[va], verbose=0)))
        if np.mean(scores) < best_score:
            best, best_score = p, float(np.mean(scores))

    tf.keras.backend.clear_session()
    model = _fit_keras(_mlp(X.shape[1], best), X, y, best["batch_size"])
    return data.to_mm(model.predict(data.X_test, verbose=0)), {**best, "cv_rmse_scaled": best_score, "grid": size}


def make_sequences(X, lookback):
    """Window ending at day i (inclusive) for each target day i >= lookback - 1."""
    return np.stack([X[i - lookback + 1:i + 1] for i in range(lookback - 1, len(X))]).astype("float32")


def _lstm(n_features, p):
    from tensorflow.keras import Input, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    model = Sequential([Input(shape=(p["lookback"], n_features))])
    for i in range(p["layers"]):
        model.add(LSTM(p["units"], return_sequences=i < p["layers"] - 1))
        if p["dropout"] > 0:
            model.add(Dropout(p["dropout"]))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=p["learning_rate"]), loss="mse")
    return model


def run_lstm(data, size="full", batch_size=32):
    """
    Each sample is the window of the last `lookback` days of the 20 features,
    ending on the target day (same information as the other models, plus history).
    """
    tf = _keras()
    X, y = data.X_train, data.y_train_scaled
    best, best_score = None, np.inf
    configs = grid(search_space("LSTM", size, LSTM_GRID))
    t0 = time.time()
    for k, p in enumerate(configs, 1):
        if k % 10 == 0:
            print(f"      LSTM config {k}/{len(configs)}, {time.time() - t0:.0f}s, "
                  f"best CV RMSE {best_score:.4f}", flush=True)
        L, scores = p["lookback"], []
        for tr, va in folds(len(X)):
            if len(tr) <= 2 * L or len(va) <= L:
                continue
            X_tr, y_tr = make_sequences(X[tr], L), y[tr][L - 1:]
            # validation windows may look back into the fold's training days
            X_va = make_sequences(np.concatenate([X[tr][-(L - 1):], X[va]]) if L > 1 else X[va], L)
            tf.keras.backend.clear_session()
            model = _fit_keras(_lstm(X.shape[1], p), X_tr, y_tr, batch_size, epochs=150, patience=15)
            scores.append(_rmse(y[va], model.predict(X_va, verbose=0)))
        if scores and np.mean(scores) < best_score:
            best, best_score = p, float(np.mean(scores))

    L = best["lookback"]
    X_seq, y_seq = make_sequences(X, L), y[L - 1:]
    X_te = make_sequences(np.concatenate([X[-(L - 1):], data.X_test]) if L > 1 else data.X_test, L)
    tf.keras.backend.clear_session()
    model = _fit_keras(_lstm(X.shape[1], best), X_seq, y_seq, batch_size)
    return data.to_mm(model.predict(X_te, verbose=0)), {**best, "cv_rmse_scaled": best_score, "grid": size}


# --------------------------------------------------------------------------
# RC+ : improved Echo State Network
# --------------------------------------------------------------------------
RC_PLUS_GRID = {"units": [300, 1000], "sr": [0.7, 1.0], "lr": [0.1, 0.3, 0.6],
                "input_scaling": [0.05, 0.2, 1.0]}
RC_PLUS_RIDGE = [1e-2, 1.0, 10.0, 100.0]
RC_PLUS_SEEDS = [0, 1, 2, 3, 4]
RC_PLUS_WARMUP = 100       # transient of the zero initial state; leak rates >= 0.1 forget it fast


def _ridge_fit(Z, y, alpha):
    """Ridge with an unpenalised intercept, solved in closed form."""
    z_mean, y_mean = Z.mean(axis=0), y.mean()
    Zc = Z - z_mean
    w = np.linalg.solve(Zc.T @ Zc + alpha * np.eye(Z.shape[1]), Zc.T @ (y - y_mean))
    return w, y_mean - z_mean @ w


def run_rc_plus(data, size="full"):
    """
    Three changes to the plain ESN (Reservoir >> Ridge):
      1. the readout sees the reservoir states AND the inputs ([x(t), u(t)]),
         so RC is never worse than a linear model on the features;
      2. the input scaling is tuned (the default 1.0 saturates tanh units
         with 20 standardized inputs);
      3. predictions are averaged over reservoirs drawn with 5 random seeds.
    Reservoir states depend only on past inputs, so each reservoir is run once
    over train + test and the CV folds simply slice the state sequence.
    """
    from reservoirpy.nodes import Reservoir

    X_all = np.vstack([data.X_train, data.X_test])
    y = data.y_train_scaled
    n_tr = len(y)
    space = QUICK.get("RC+", RC_PLUS_GRID) if size == "quick" else RC_PLUS_GRID
    seeds = RC_PLUS_SEEDS[:2] if size == "quick" else RC_PLUS_SEEDS
    cv = [(tr[tr >= RC_PLUS_WARMUP], va) for tr, va in folds(n_tr)]

    best, best_score, best_states = None, np.inf, None
    for p in grid(space):
        states = []
        for s in seeds:
            res = Reservoir(units=p["units"], sr=p["sr"], lr=p["lr"],
                            input_scaling=p["input_scaling"], seed=config.SEED + s)
            states.append(np.hstack([np.asarray(res.run(X_all)), X_all]))
        for alpha in RC_PLUS_RIDGE:
            scores = []
            for tr, va in cv:
                pred = np.mean([Z[va] @ w + b for Z in states
                                for w, b in [_ridge_fit(Z[tr], y[tr], alpha)]], axis=0)
                scores.append(_rmse(y[va], pred))
            if np.mean(scores) < best_score:
                best, best_score = {**p, "ridge": alpha}, float(np.mean(scores))
                best_states = states

    # refit the readouts on the whole training period and average over seeds
    tr = np.arange(RC_PLUS_WARMUP, n_tr)
    pred = np.mean([Z[n_tr:] @ w + b for Z in best_states
                    for w, b in [_ridge_fit(Z[tr], y[tr], best["ridge"])]], axis=0)
    return data.to_mm(pred), {**best, "seeds": len(seeds), "cv_rmse_scaled": best_score}


# --------------------------------------------------------------------------
# ARIMAX baseline
# --------------------------------------------------------------------------
def run_arimax(data, mode="multi-step"):
    """
    Orders (p, d, q) chosen by auto_arima on the training period; no seasonal
    terms (the sin/cos inputs carry the season) and no rain lags (the AR part
    models them).

    mode="multi-step": the whole test period is forecast from the end of
        training using only the exogenous inputs (what the paper's code did).
    mode="one-step": each test day is predicted from the observed rainfall up
        to the previous day, like the other models; parameters stay fixed.
    """
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cols = data.lag_free_columns()
    ex_tr, ex_te = data.X_train[:, cols], data.X_test[:, cols]
    y_log = data.y_train_log

    order = auto_arima(y_log, X=ex_tr, seasonal=False, stepwise=True,
                       error_action="ignore", suppress_warnings=True).order
    res = SARIMAX(y_log, exog=ex_tr, order=order, enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)

    n_tr, n_te = len(y_log), len(ex_te)
    if mode == "one-step":
        res = res.append(np.log1p(data.y_test), exog=ex_te, refit=False)
        y_pred_log = res.predict(start=n_tr, end=n_tr + n_te - 1)
    else:
        y_pred_log = res.predict(start=n_tr, end=n_tr + n_te - 1, exog=ex_te)
    return data.clip_mm(np.expm1(np.asarray(y_pred_log))), {"order": list(order), "mode": mode}


MODEL_RUNNERS = {
    "ARIMAX": lambda d, size, arimax_mode: run_arimax(d, arimax_mode),
    "Random Forest": lambda d, size, _: run_tree("Random Forest", d, size),
    "XGBoost": lambda d, size, _: run_tree("XGBoost", d, size),
    "LightGBM": lambda d, size, _: run_tree("LightGBM", d, size),
    "MLP": lambda d, size, _: run_mlp(d, size),
    "LSTM": lambda d, size, _: run_lstm(d, size),
    "RC": lambda d, size, _: run_rc(d, size),
    "RC+": lambda d, size, _: run_rc_plus(d, size),
}
