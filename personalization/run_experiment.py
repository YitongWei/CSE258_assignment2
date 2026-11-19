"""Train personalized models A/B, tune on validation, score test.

Environment (created on F: because C: was full):
  F:\\venvs\\cse258_personalization\\Scripts\\python.exe personalization\\run_experiment.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import (
    ContentEncoder,
    build_eval_feature_table,
    build_prefix_feature_table,
    build_user_profiles,
    find_root,
    load_stage,
)
from models import ModelA, ModelB, mae, mse, tune_residual_shift

ROOT = find_root(Path(__file__).resolve().parent)
SHARED = ROOT / "data_prepared" / "shared"
TEST_LABELS = ROOT / "data_prepared" / "wyt" / "test_labels.csv"
OUT = Path(__file__).resolve().parent / "outputs"
PRED_DIR, METRIC_DIR, TUNING_DIR = OUT / "predictions", OUT / "metrics", OUT / "tuning"
for d in (PRED_DIR, METRIC_DIR, TUNING_DIR):
    d.mkdir(parents=True, exist_ok=True)

USER_MEAN_K = 5.0
USER_MEDIAN_K = 2.0
SVD_DIM = 128
SIM_K_GRID = [0.0, 5.0, 20.0]
ALPHA_GRID = [1.0, 10.0, 100.0, 1000.0]
RIDGE_FLAGS = [False, True]
SHIFT_GRID = np.round(np.arange(-0.5, 0.501, 0.05), 2)
BASES_B = ["user_median", "user_mean"]
TEST_FIRST_YEAR_END = "2013-09-10"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, str):
        return obj
    return obj


def apply_sim_shrink(feats: pd.DataFrame, sim_k: float) -> pd.DataFrame:
    out = feats.copy()
    n = out["n_hist"].to_numpy(dtype=float)
    raw = out["sim"].to_numpy(dtype=float)
    if sim_k <= 0:
        out["sim"] = raw
    else:
        out["sim"] = np.where(n > 0, raw * (n / (n + sim_k)), 0.0)
    return out


def bootstrap_ci(abs_err: np.ndarray, sq_err: np.ndarray, users: np.ndarray, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = pd.unique(users)
    user_to_idx = {u: i for i, u in enumerate(uniq)}
    buckets = [[] for _ in uniq]
    for i, u in enumerate(users):
        buckets[user_to_idx[u]].append(i)
    buckets = [np.asarray(ix, dtype=np.int64) for ix in buckets]
    mae_s, mse_s = [], []
    n_u = len(buckets)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_u, size=n_u)
        idx = np.concatenate([buckets[c] for c in chosen])
        mae_s.append(float(abs_err[idx].mean()))
        mse_s.append(float(sq_err[idx].mean()))
    return (
        (float(np.quantile(mae_s, 0.025)), float(np.quantile(mae_s, 0.975))),
        (float(np.quantile(mse_s, 0.025)), float(np.quantile(mse_s, 0.975))),
    )


def summarize(name: str, y: np.ndarray, p: np.ndarray, users: np.ndarray) -> dict:
    abs_err = np.abs(y - p)
    sq_err = (y - p) ** 2
    (mae_lo, mae_hi), (mse_lo, mse_hi) = bootstrap_ci(abs_err, sq_err, users)
    return {
        "model": name,
        "MAE": float(abs_err.mean()),
        "MAE_95CI": f"[{mae_lo:.4f}, {mae_hi:.4f}]",
        "MSE": float(sq_err.mean()),
        "MSE_95CI": f"[{mse_lo:.4f}, {mse_hi:.4f}]",
    }


def write_preds(eval_df: pd.DataFrame, preds: np.ndarray, name: str, stage: str) -> None:
    path = PRED_DIR / f"{stage}_{name}.csv"
    pd.DataFrame({"row_id": eval_df["row_id"], "prediction": preds}).to_csv(path, index=False)


def prepare_stage(stage: str):
    print(f"[{stage}] loading data...", flush=True)
    fit_df, eval_df = load_stage(SHARED, stage)
    y_fit = fit_df["rating"].to_numpy(dtype=float)
    g_mean, g_median = float(y_fit.mean()), float(np.median(y_fit))

    print(f"[{stage}] fitting content encoder on {len(fit_df):,} rows...", flush=True)
    t0 = time.time()
    encoder = ContentEncoder(n_components=SVD_DIM, min_df=5, ridge_alpha=300.0).fit(fit_df)
    print(f"[{stage}] encoder done in {time.time() - t0:.1f}s", flush=True)

    print(f"[{stage}] building leakage-safe train features...", flush=True)
    t0 = time.time()
    X_fit = encoder.recipe_matrix(fit_df)
    ridge_fit = encoder.global_ridge_predict(fit_df["ingredients"])
    train_feats = build_prefix_feature_table(
        fit_df, X_fit, g_mean, g_median, USER_MEAN_K, USER_MEDIAN_K, sim_k=0.0, global_ridge=ridge_fit
    )
    print(f"[{stage}] train features done in {time.time() - t0:.1f}s", flush=True)

    print(f"[{stage}] building frozen eval features for {len(eval_df):,} rows...", flush=True)
    t0 = time.time()
    X_eval = encoder.recipe_matrix(eval_df)
    ridge_eval = encoder.global_ridge_predict(eval_df["ingredients"])
    profiles = build_user_profiles(fit_df, X_fit, g_mean, g_median, USER_MEAN_K, USER_MEDIAN_K)
    eval_feats = build_eval_feature_table(
        eval_df, X_eval, profiles, g_mean, g_median, sim_k=0.0, global_ridge=ridge_eval
    )
    print(f"[{stage}] eval features done in {time.time() - t0:.1f}s", flush=True)

    y_eval = None
    if "rating" in eval_df.columns:
        y_eval = eval_df["rating"].to_numpy(dtype=float)
    return {
        "fit_df": fit_df,
        "eval_df": eval_df,
        "train_feats": train_feats,
        "eval_feats": eval_feats,
        "y_fit": y_fit,
        "y_eval": y_eval,
        "g_mean": g_mean,
        "g_median": g_median,
    }


def run_validation():
    data = prepare_stage("val")
    y_tr, y_va = data["y_fit"], data["y_eval"]
    rows = []

    print("[val] tuning Model B then Model A...", flush=True)
    for sim_k in SIM_K_GRID:
        tr = apply_sim_shrink(data["train_feats"], sim_k)
        va = apply_sim_shrink(data["eval_feats"], sim_k)
        for use_ridge in RIDGE_FLAGS:
            for alpha in ALPHA_GRID:
                # --- Model B ---
                for base in BASES_B:
                    model = ModelB(base=base, alpha=alpha, use_global_ridge=use_ridge, residual_shift=0.0)
                    model.fit(tr, y_tr)
                    raw_delta = model.predict_raw_residual(va)
                    base_va = va[base].to_numpy(dtype=float)
                    shift, _ = tune_residual_shift(base_va, raw_delta, y_va, SHIFT_GRID)
                    model.residual_shift = shift
                    pred = model.predict(va)
                    rows.append(
                        {
                            "family": "B",
                            "base": base,
                            "sim_k": sim_k,
                            "alpha": alpha,
                            "use_global_ridge": use_ridge,
                            "residual_shift": shift,
                            "MAE": mae(y_va, pred),
                            "MSE": mse(y_va, pred),
                        }
                    )

                # --- Model A ---
                model_a = ModelA(alpha=alpha, use_global_ridge=use_ridge).fit(tr, y_tr)
                pred_a = model_a.predict(va)
                rows.append(
                    {
                        "family": "A",
                        "base": "user_mean_feature",
                        "sim_k": sim_k,
                        "alpha": alpha,
                        "use_global_ridge": use_ridge,
                        "residual_shift": 0.0,
                        "MAE": mae(y_va, pred_a),
                        "MSE": mse(y_va, pred_a),
                    }
                )

    grid = pd.DataFrame(rows).sort_values(["MAE", "MSE"]).reset_index(drop=True)
    grid.to_csv(TUNING_DIR / "val_grid.csv", index=False)

    best_mae = _jsonable(grid.loc[grid["MAE"].idxmin()].to_dict())
    best_mse = _jsonable(grid.loc[grid["MSE"].idxmin()].to_dict())
    params = {
        "user_mean_k": USER_MEAN_K,
        "user_median_k": USER_MEDIAN_K,
        "svd_dim": SVD_DIM,
        "best_mae": best_mae,
        "best_mse": best_mse,
    }
    (TUNING_DIR / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print("[val] best MAE config:", params["best_mae"], flush=True)
    print("[val] best MSE config:", params["best_mse"], flush=True)

    # Refit chosen configs on train; write val predictions + metrics
    metrics = []
    for label, cfg in [("best_mae", best_mae), ("best_mse", best_mse)]:
        name = f"personalization_{label}"
        pred = fit_predict_from_config(data["train_feats"], data["eval_feats"], y_tr, cfg)
        write_preds(data["eval_df"], pred, name, "val")
        summary = summarize(name, y_va, pred, data["eval_df"]["user_id"].to_numpy())
        summary["tuned_on"] = "MAE" if label == "best_mae" else "MSE"
        summary["family"] = cfg["family"]
        metrics.append(summary)
        write_preds(data["eval_df"], pred, f"{str(cfg['family']).lower()}_{label}", "val")

    # Baselines for reference on same sample
    for base_name, col in [("user_mean", "user_mean"), ("user_median", "user_median")]:
        pred = data["eval_feats"][col].to_numpy(dtype=float)
        write_preds(data["eval_df"], pred, f"ref_{base_name}", "val")
        summary = summarize(f"ref_{base_name}", y_va, pred, data["eval_df"]["user_id"].to_numpy())
        summary["tuned_on"] = "baseline-ref"
        summary["family"] = "ref"
        metrics.append(summary)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRIC_DIR / "val_metrics.csv", index=False)
    print(metrics_df.to_string(index=False), flush=True)
    return params


def fit_predict_from_config(train_feats, eval_feats, y_tr, cfg) -> np.ndarray:
    sim_k = float(cfg["sim_k"])
    tr = apply_sim_shrink(train_feats, sim_k)
    ev = apply_sim_shrink(eval_feats, sim_k)
    use_ridge = bool(cfg["use_global_ridge"])
    alpha = float(cfg["alpha"])
    if cfg["family"] == "B":
        model = ModelB(
            base=str(cfg["base"]),
            alpha=alpha,
            use_global_ridge=use_ridge,
            residual_shift=float(cfg["residual_shift"]),
        )
        return model.fit(tr, y_tr).predict(ev)
    model = ModelA(alpha=alpha, use_global_ridge=use_ridge)
    return model.fit(tr, y_tr).predict(ev)


def run_test(params: dict):
    data = prepare_stage("test")
    labels = pd.read_csv(TEST_LABELS)
    eval_df = data["eval_df"].merge(labels, on="row_id", how="left", validate="one_to_one")
    if eval_df["rating"].isna().any():
        raise RuntimeError("test labels missing for some row_id")
    y = eval_df["rating"].to_numpy(dtype=float)
    users = eval_df["user_id"].to_numpy()

    metrics = []
    period = np.where(eval_df["date"] <= TEST_FIRST_YEAR_END, "first_year", "later")
    by_period_rows = []

    for label in ["best_mae", "best_mse"]:
        cfg = params[label]
        name = f"personalization_{label}"
        pred = fit_predict_from_config(data["train_feats"], data["eval_feats"], data["y_fit"], cfg)
        write_preds(eval_df, pred, name, "test")
        write_preds(eval_df, pred, f"{cfg['family'].lower()}_{label}", "test")
        summary = summarize(name, y, pred, users)
        summary["tuned_on"] = "MAE" if label == "best_mae" else "MSE"
        summary["family"] = cfg["family"]
        summary["config"] = json.dumps(cfg)
        metrics.append(summary)

        for g in ["first_year", "later", "overall"]:
            if g == "overall":
                m = np.ones(len(y), dtype=bool)
            else:
                m = period == g
            by_period_rows.append(
                {
                    "model": name,
                    "period": g,
                    "rows": int(m.sum()),
                    "users": int(pd.unique(users[m]).size),
                    "MAE": mae(y[m], pred[m]),
                    "MSE": mse(y[m], pred[m]),
                }
            )

    for base_name, col in [("user_mean", "user_mean"), ("user_median", "user_median")]:
        pred = data["eval_feats"][col].to_numpy(dtype=float)
        write_preds(eval_df, pred, f"ref_{base_name}", "test")
        summary = summarize(f"ref_{base_name}", y, pred, users)
        summary["tuned_on"] = "baseline-ref"
        summary["family"] = "ref"
        summary["config"] = "{}"
        metrics.append(summary)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRIC_DIR / "test_metrics.csv", index=False)
    pd.DataFrame(by_period_rows).to_csv(METRIC_DIR / "test_metrics_by_period.csv", index=False)
    print(metrics_df.to_string(index=False), flush=True)
    return metrics_df


def main():
    t0 = time.time()
    params = run_validation()
    run_test(params)
    print(f"total elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
