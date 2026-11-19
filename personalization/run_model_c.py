"""Tune and evaluate Model C: user_mean + HistGradientBoosting residual (MSE).

  F:\\venvs\\cse258_personalization\\Scripts\\python.exe personalization\\run_model_c.py
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from models import ModelA, ModelC, mae, mse
from run_experiment import (
    METRIC_DIR,
    PRED_DIR,
    TEST_FIRST_YEAR_END,
    TEST_LABELS,
    TUNING_DIR,
    _jsonable,
    apply_sim_shrink,
    fit_predict_from_config,
    prepare_stage,
    summarize,
    write_preds,
)


def fit_predict_c(train_feats, eval_feats, y_tr, cfg) -> np.ndarray:
    tr = apply_sim_shrink(train_feats, float(cfg["sim_k"]))
    ev = apply_sim_shrink(eval_feats, float(cfg["sim_k"]))
    model = ModelC(
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        max_iter=int(cfg["max_iter"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        l2_regularization=float(cfg["l2_regularization"]),
        use_global_ridge=bool(cfg["use_global_ridge"]),
    )
    return model.fit(tr, y_tr).predict(ev)


def main():
    t0 = time.time()
    data = prepare_stage("val")
    y_tr, y_va = data["y_fit"], data["y_eval"]
    users_va = data["eval_df"]["user_id"].to_numpy()

    grid = {
        "sim_k": [0.0, 5.0, 20.0],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "max_iter": [200],
        "min_samples_leaf": [50],
        "l2_regularization": [0.0, 1.0],
        "use_global_ridge": [True],
    }
    keys = list(grid.keys())
    rows = []
    print(f"[val] tuning Model C over {np.prod([len(grid[k]) for k in keys])} configs...", flush=True)
    for values in product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, values))
        t1 = time.time()
        pred = fit_predict_c(data["train_feats"], data["eval_feats"], y_tr, cfg)
        row = {
            **cfg,
            "MAE": mae(y_va, pred),
            "MSE": mse(y_va, pred),
            "fit_sec": round(time.time() - t1, 2),
        }
        rows.append(row)
        print(
            f"  MSE={row['MSE']:.5f} MAE={row['MAE']:.5f} "
            f"sim_k={cfg['sim_k']} depth={cfg['max_depth']} lr={cfg['learning_rate']} "
            f"l2={cfg['l2_regularization']} ({row['fit_sec']}s)",
            flush=True,
        )

    c_grid = pd.DataFrame(rows).sort_values("MSE").reset_index(drop=True)
    c_grid.to_csv(TUNING_DIR / "val_model_c_grid.csv", index=False)
    best_c = _jsonable(c_grid.iloc[0].to_dict())
    print("[val] best Model C:", best_c, flush=True)

    # Locked Model A from previous MSE tuning
    params_path = TUNING_DIR / "params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    best_a = params["best_mse"]

    comparisons = []
    # user_mean ref
    pred_um = data["eval_feats"]["user_mean"].to_numpy(dtype=float)
    write_preds(data["eval_df"], pred_um, "ref_user_mean", "val")
    comparisons.append({**summarize("ref_user_mean", y_va, pred_um, users_va), "tuned_on": "baseline-ref", "family": "ref"})

    # Model A
    pred_a = fit_predict_from_config(data["train_feats"], data["eval_feats"], y_tr, best_a)
    write_preds(data["eval_df"], pred_a, "personalization_best_mse", "val")
    comparisons.append({**summarize("personalization_A", y_va, pred_a, users_va), "tuned_on": "MSE", "family": "A"})

    # Model C
    pred_c = fit_predict_c(data["train_feats"], data["eval_feats"], y_tr, best_c)
    write_preds(data["eval_df"], pred_c, "personalization_C", "val")
    write_preds(data["eval_df"], pred_c, "c_best_mse", "val")
    comparisons.append({**summarize("personalization_C", y_va, pred_c, users_va), "tuned_on": "MSE", "family": "C"})

    val_metrics = pd.DataFrame(comparisons)
    val_metrics.to_csv(METRIC_DIR / "val_metrics_with_c.csv", index=False)
    print("\n[val] metrics\n", val_metrics.to_string(index=False), flush=True)

    # ---- test ----
    data_te = prepare_stage("test")
    labels = pd.read_csv(TEST_LABELS)
    eval_df = data_te["eval_df"].merge(labels, on="row_id", how="left", validate="one_to_one")
    if eval_df["rating"].isna().any():
        raise RuntimeError("missing test labels")
    y = eval_df["rating"].to_numpy(dtype=float)
    users = eval_df["user_id"].to_numpy()
    period = np.where(eval_df["date"] <= TEST_FIRST_YEAR_END, "first_year", "later")

    test_rows, period_rows = [], []
    for name, family, pred in [
        ("ref_user_mean", "ref", data_te["eval_feats"]["user_mean"].to_numpy(dtype=float)),
        ("personalization_A", "A", fit_predict_from_config(data_te["train_feats"], data_te["eval_feats"], data_te["y_fit"], best_a)),
        ("personalization_C", "C", fit_predict_c(data_te["train_feats"], data_te["eval_feats"], data_te["y_fit"], best_c)),
    ]:
        write_preds(eval_df, pred, name if name != "personalization_C" else "personalization_C", "test")
        if name == "personalization_C":
            write_preds(eval_df, pred, "c_best_mse", "test")
        if name == "personalization_A":
            write_preds(eval_df, pred, "personalization_best_mse", "test")
        summary = summarize(name, y, pred, users)
        summary["tuned_on"] = "MSE" if family != "ref" else "baseline-ref"
        summary["family"] = family
        test_rows.append(summary)
        for g in ["first_year", "later", "overall"]:
            m = np.ones(len(y), dtype=bool) if g == "overall" else (period == g)
            period_rows.append(
                {
                    "model": name,
                    "period": g,
                    "rows": int(m.sum()),
                    "users": int(pd.unique(users[m]).size),
                    "MAE": mae(y[m], pred[m]),
                    "MSE": mse(y[m], pred[m]),
                }
            )

    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(METRIC_DIR / "test_metrics_with_c.csv", index=False)
    pd.DataFrame(period_rows).to_csv(METRIC_DIR / "test_metrics_by_period_with_c.csv", index=False)
    print("\n[test] metrics\n", test_metrics.to_string(index=False), flush=True)

    # Update params: submit C if it beats A on the same validation predictions
    mse_a_val = mse(y_va, pred_a)
    mse_c_val = float(best_c["MSE"])
    params["best_c"] = best_c
    params["primary_metric"] = "MSE"
    if mse_c_val < mse_a_val:
        params["submit_model"] = "best_c"
        params["submit_family"] = "C"
        write_preds(data["eval_df"], pred_c, "personalization_best_mse", "val")
        pred_c_te = fit_predict_c(data_te["train_feats"], data_te["eval_feats"], data_te["y_fit"], best_c)
        write_preds(eval_df, pred_c_te, "personalization_best_mse", "test")
        print(f"[select] Model C wins on val MSE ({mse_c_val:.5f} < {mse_a_val:.5f}) → submit_model=best_c", flush=True)
    else:
        params["submit_model"] = "best_mse"
        params["submit_family"] = "A"
        print(f"[select] Model A still best on val MSE ({mse_a_val:.5f} <= {mse_c_val:.5f}) → keep A", flush=True)

    params_path.write_text(json.dumps(_jsonable(params), indent=2) + "\n", encoding="utf-8")
    print(f"total elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
