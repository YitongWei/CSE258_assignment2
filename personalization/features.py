"""Leakage-safe user/content features for personalized rating models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

STARS = np.arange(1, 6, dtype=float)


def find_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if (p / "data_prepared" / "shared").is_dir():
            return p
    raise FileNotFoundError("data_prepared/shared not found: unzip shared data at repo root")


def word_analyzer(ingredients: list[str]) -> list[str]:
    return [w for item in ingredients for w in item.split()]


def read_split(shared: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(shared / f"{name}.csv", dtype={"user_id": str, "recipe_id": str})
    parsed = {s: json.loads(s) for s in df["ingredients"].unique()}
    df["ingredients"] = df["ingredients"].map(parsed)
    return df


def load_stage(shared: Path, stage: str):
    """stage='val' uses train fit + cold-start validation rows; 'test' merges train+val."""
    train = read_split(shared, "train")
    if stage == "val":
        validation = read_split(shared, "validation")
        eval_df = validation.loc[validation["is_cold_start_eval"] == 1].reset_index(drop=True)
        return train, eval_df
    if stage == "test":
        validation = read_split(shared, "validation").drop(columns="is_cold_start_eval")
        fit_df = pd.concat([train, validation], ignore_index=True)
        eval_df = read_split(shared, "test_features")
        return fit_df, eval_df
    raise ValueError(stage)


def shrunk_user_mean(sum_r: float, n: int, global_mean: float, k: float) -> float:
    if n <= 0:
        return float(global_mean)
    return float((sum_r + k * global_mean) / (n + k))


def shrunk_user_median(counts: np.ndarray, global_median: float, k: float) -> float:
    c = counts.astype(float).copy()
    c[int(global_median) - 1] += k
    n = c.sum()
    if n <= 0:
        return float(global_median)
    cum = np.cumsum(c)
    lo, hi = int((n - 1) // 2), int(n // 2)
    v_lo = int(np.argmax(cum > lo) + 1)
    v_hi = int(np.argmax(cum > hi) + 1)
    return 0.5 * (v_lo + v_hi)


class ContentEncoder:
    """TF-IDF (word) + SVD recipe vectors, plus a global ingredient Ridge score."""

    def __init__(self, n_components: int = 128, min_df: int = 5, ridge_alpha: float = 300.0):
        self.n_components = n_components
        self.min_df = min_df
        self.ridge_alpha = ridge_alpha
        self.vectorizer = TfidfVectorizer(analyzer=word_analyzer, min_df=min_df)
        self.svd = TruncatedSVD(n_components=n_components, random_state=0)
        self.ridge = Ridge(alpha=ridge_alpha, solver="sparse_cg")
        self.recipe_vec: dict[str, np.ndarray] = {}

    def fit(self, fit_df: pd.DataFrame) -> "ContentEncoder":
        recipes = fit_df.drop_duplicates("recipe_id")
        tfidf_recipes = self.vectorizer.fit_transform(recipes["ingredients"])
        low = self.svd.fit_transform(tfidf_recipes)
        self.recipe_vec = {rid: low[i] for i, rid in enumerate(recipes["recipe_id"].to_numpy())}

        tfidf_rows = self.vectorizer.transform(fit_df["ingredients"])
        self.ridge.fit(tfidf_rows, fit_df["rating"].to_numpy(dtype=float))
        return self

    def transform_recipes(self, recipe_ids: pd.Series) -> np.ndarray:
        dim = self.n_components
        out = np.zeros((len(recipe_ids), dim), dtype=np.float64)
        lookup = self.recipe_vec
        for i, rid in enumerate(recipe_ids.to_numpy()):
            vec = lookup.get(rid)
            if vec is not None:
                out[i] = vec
            # unseen cold-start recipes: encode from ingredients via vectorizer+SVD below if needed
        return out

    def transform_ingredients(self, ingredients: pd.Series) -> np.ndarray:
        return self.svd.transform(self.vectorizer.transform(ingredients))

    def recipe_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Prefer fitted recipe vectors; fall back to transforming ingredients for new recipes."""
        known = df["recipe_id"].isin(self.recipe_vec)
        out = np.zeros((len(df), self.n_components), dtype=np.float64)
        if known.any():
            out[known.to_numpy()] = self.transform_recipes(df.loc[known, "recipe_id"])
        if (~known).any():
            out[(~known).to_numpy()] = self.transform_ingredients(df.loc[~known, "ingredients"])
        return out

    def global_ridge_predict(self, ingredients: pd.Series) -> np.ndarray:
        return self.ridge.predict(self.vectorizer.transform(ingredients))


def _similarity(sum_rx: np.ndarray, sum_x: np.ndarray, mu: float, x: np.ndarray, n: int, sim_k: float) -> float:
    if n <= 0:
        return 0.0
    p = sum_rx - mu * sum_x
    pn = float(np.linalg.norm(p))
    xn = float(np.linalg.norm(x))
    if pn < 1e-12 or xn < 1e-12:
        s = 0.0
    else:
        s = float(np.dot(p, x) / (pn * xn))
    return s * (n / (n + sim_k)) if sim_k > 0 else s


def build_prefix_feature_table(
    df: pd.DataFrame,
    X: np.ndarray,
    global_mean: float,
    global_median: float,
    user_mean_k: float,
    user_median_k: float,
    sim_k: float,
    global_ridge: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-row features using only strictly earlier ratings of the same user."""
    order = np.lexsort((df["row_id"].to_numpy(), df["date"].to_numpy(), df["user_id"].to_numpy()))
    users = df["user_id"].to_numpy()[order]
    ratings = df["rating"].to_numpy(dtype=float)[order]
    Xo = X[order]
    ridge = None if global_ridge is None else global_ridge[order]

    n_rows, dim = Xo.shape
    user_mean = np.empty(n_rows, dtype=np.float64)
    user_median = np.empty(n_rows, dtype=np.float64)
    sim = np.empty(n_rows, dtype=np.float64)
    log_n = np.empty(n_rows, dtype=np.float64)
    n_hist = np.empty(n_rows, dtype=np.int32)

    i = 0
    while i < n_rows:
        j = i + 1
        while j < n_rows and users[j] == users[i]:
            j += 1
        sum_x = np.zeros(dim, dtype=np.float64)
        sum_rx = np.zeros(dim, dtype=np.float64)
        counts = np.zeros(5, dtype=np.float64)
        sum_r = 0.0
        n = 0
        for t in range(i, j):
            mu = shrunk_user_mean(sum_r, n, global_mean, user_mean_k)
            med = shrunk_user_median(counts, global_median, user_median_k) if n > 0 else float(global_median)
            user_mean[t] = mu
            user_median[t] = med
            sim[t] = _similarity(sum_rx, sum_x, mu, Xo[t], n, sim_k)
            log_n[t] = np.log1p(n)
            n_hist[t] = n

            r = ratings[t]
            sum_x += Xo[t]
            sum_rx += r * Xo[t]
            sum_r += r
            counts[int(r) - 1] += 1.0
            n += 1
        i = j

    inv = np.empty(n_rows, dtype=np.int64)
    inv[order] = np.arange(n_rows)
    out = {
        "user_mean": user_mean[inv],
        "user_median": user_median[inv],
        "sim": sim[inv],
        "log_n": log_n[inv],
        "n_hist": n_hist[inv],
    }
    if ridge is not None:
        out["global_ridge"] = ridge[inv]
    return pd.DataFrame(out)


def build_user_profiles(
    fit_df: pd.DataFrame,
    X_fit: np.ndarray,
    global_mean: float,
    global_median: float,
    user_mean_k: float,
    user_median_k: float,
) -> dict[str, dict]:
    """Freeze one profile per user from the full fitting history."""
    order = np.lexsort((fit_df["row_id"].to_numpy(), fit_df["date"].to_numpy(), fit_df["user_id"].to_numpy()))
    users = fit_df["user_id"].to_numpy()[order]
    ratings = fit_df["rating"].to_numpy(dtype=float)[order]
    Xo = X_fit[order]
    profiles: dict[str, dict] = {}

    i = 0
    n_rows, dim = Xo.shape
    while i < n_rows:
        j = i + 1
        while j < n_rows and users[j] == users[i]:
            j += 1
        block = Xo[i:j]
        r = ratings[i:j]
        n = j - i
        sum_r = float(r.sum())
        counts = np.zeros(5, dtype=np.float64)
        for star in r:
            counts[int(star) - 1] += 1.0
        mu = shrunk_user_mean(sum_r, n, global_mean, user_mean_k)
        med = shrunk_user_median(counts, global_median, user_median_k)
        sum_x = block.sum(axis=0)
        sum_rx = (block * r[:, None]).sum(axis=0)
        p = sum_rx - mu * sum_x
        pn = float(np.linalg.norm(p))
        if pn > 1e-12:
            p = p / pn
        else:
            p = np.zeros(dim, dtype=np.float64)
        profiles[users[i]] = {
            "user_mean": mu,
            "user_median": med,
            "taste": p,
            "n": n,
            "log_n": float(np.log1p(n)),
        }
        i = j
    return profiles


def build_eval_feature_table(
    eval_df: pd.DataFrame,
    X_eval: np.ndarray,
    profiles: dict[str, dict],
    global_mean: float,
    global_median: float,
    sim_k: float,
    global_ridge: np.ndarray | None = None,
) -> pd.DataFrame:
    n_rows = len(eval_df)
    user_mean = np.full(n_rows, global_mean, dtype=np.float64)
    user_median = np.full(n_rows, global_median, dtype=np.float64)
    sim = np.zeros(n_rows, dtype=np.float64)
    log_n = np.zeros(n_rows, dtype=np.float64)
    n_hist = np.zeros(n_rows, dtype=np.int32)
    users = eval_df["user_id"].to_numpy()

    for i, uid in enumerate(users):
        prof = profiles.get(uid)
        if prof is None:
            continue
        user_mean[i] = prof["user_mean"]
        user_median[i] = prof["user_median"]
        log_n[i] = prof["log_n"]
        n = prof["n"]
        n_hist[i] = n
        x = X_eval[i]
        xn = float(np.linalg.norm(x))
        if n > 0 and xn > 1e-12:
            s = float(np.dot(prof["taste"], x) / xn)
            sim[i] = s * (n / (n + sim_k)) if sim_k > 0 else s

    out = {
        "user_mean": user_mean,
        "user_median": user_median,
        "sim": sim,
        "log_n": log_n,
        "n_hist": n_hist,
    }
    if global_ridge is not None:
        out["global_ridge"] = global_ridge
    return pd.DataFrame(out)
