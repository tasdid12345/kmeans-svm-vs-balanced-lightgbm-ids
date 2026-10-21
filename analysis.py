"""K-means-filtered SVM hybrid (baseline) vs class-balanced LightGBM (improved) on NSL-KDD.

Both models share data loading, preprocessing, CV threshold selection and evaluation.
Only the fit/score function differs. Seeds 42, 43, 44 are run for both models.
"""
import io
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

SEEDS = [42, 43, 44]          # both models are randomised, so run all seeds
N_SUB = 20000                 # fixed random subsample of training rows (seed 42)
BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
TRAIN_URL, TEST_URL = BASE + "KDDTrain+.txt", BASE + "KDDTest+.txt"
CAT = ["protocol_type", "service", "flag"]
FEATS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
         "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
         "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
         "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
         "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
         "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
         "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
         "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
         "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
COLS = FEATS + ["label", "difficulty"]


def download(url):
    """GET with timeout=60 and retry/backoff on HTTP 429 and 5xx."""
    for attempt in range(5):
        r = requests.get(url, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            print(f"  HTTP {r.status_code} for {url}, retrying...")
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError(f"could not download {url}")


def preprocess(tr, te):
    """One-hot encode categoricals, min-max scale with training statistics, binary labels."""
    y_tr = (tr["label"].str.strip() != "normal").astype(int).values
    y_te = (te["label"].str.strip() != "normal").astype(int).values
    X_tr = pd.get_dummies(tr[FEATS], columns=CAT).astype(float)
    X_te = pd.get_dummies(te[FEATS], columns=CAT).astype(float).reindex(columns=X_tr.columns, fill_value=0.0)
    scaler = MinMaxScaler().fit(X_tr.values)          # statistics from training data only
    return scaler.transform(X_tr.values), y_tr, scaler.transform(X_te.values), y_te


def make_baseline(seed):
    """K-means (k=20) label-purity filter, then RBF SVM. Returns decision scores."""
    def fit_score(Xtr, ytr, Xte):
        clusters = KMeans(n_clusters=20, n_init=10, random_state=seed).fit_predict(Xtr)
        majority = pd.Series(ytr).groupby(clusters).transform("mean").values >= 0.5
        keep = ytr == majority                         # drop rows disagreeing with cluster majority
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000, random_state=seed)
        svm.fit(Xtr[keep], ytr[keep])
        return svm.decision_function(Xte)
    return fit_score


def make_improved(seed):
    """Class-balanced LightGBM. Returns malicious-class probabilities."""
    def fit_score(Xtr, ytr, Xte):
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           class_weight="balanced", random_state=seed, verbose=-1)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    return fit_score


def best_threshold(y, s):
    """Threshold maximising F1 on out-of-fold scores (quantile grid of candidates)."""
    cands = np.unique(np.quantile(s, np.linspace(0, 1, 501)))
    f1s = [f1_score(y, s >= t, zero_division=0) for t in cands]
    return cands[int(np.argmax(f1s))]


def metrics(y, pred):
    """Malicious-class F1, detection rate (recall) and false alarm rate (FPR)."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"f1": f1_score(y, pred, zero_division=0),
            "detection_rate": tp / (tp + fn), "false_alarm_rate": fp / (fp + tn)}


def evaluate(fit_score, Xtr, ytr, Xte, yte, seed):
    """Same protocol for every model: 5-fold CV threshold on train, one test application."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(ytr))
    for tr, va in skf.split(Xtr, ytr):
        oof[va] = fit_score(Xtr[tr], ytr[tr], Xtr[va])
    thr = best_threshold(ytr, oof)
    test_scores = fit_score(Xtr, ytr, Xte)
    res = metrics(yte, test_scores >= thr)
    res["threshold"] = float(thr)
    return res


def main():
    t0 = time.time()
    print("Downloading NSL-KDD...")
    tr = pd.read_csv(io.StringIO(download(TRAIN_URL)), header=None, names=COLS)
    te = pd.read_csv(io.StringIO(download(TEST_URL)), header=None, names=COLS)
    n_rows = len(tr) + len(te)
    print(f"  train file {len(tr)} rows, test file {len(te)} rows")

    Xtr_full, ytr_full, Xte, yte = preprocess(tr, te)
    idx = np.random.default_rng(42).choice(len(ytr_full), N_SUB, replace=False)
    Xtr, ytr = Xtr_full[idx], ytr_full[idx]
    print(f"  features={Xtr.shape[1]}, subsample={len(ytr)}, malicious share={ytr.mean():.3f}")

    runs = {"baseline": [], "improved": []}
    for seed in SEEDS:
        print(f"Seed {seed}: baseline (K-means filter + RBF SVM)...")
        runs["baseline"].append(evaluate(make_baseline(seed), Xtr, ytr, Xte, yte, seed))
        print(f"Seed {seed}: improved (balanced LightGBM)...")
        runs["improved"].append(evaluate(make_improved(seed), Xtr, ytr, Xte, yte, seed))
        print(f"  elapsed {time.time() - t0:.0f}s")

    agg = {}
    for name, rs in runs.items():
        for k in ["f1", "detection_rate", "false_alarm_rate"]:
            vals = np.array([r[k] for r in rs])
            agg[f"{name}_{k}"] = float(vals.mean())
            if k == "f1":
                agg[f"{name}_score_std"] = float(vals.std())   # population std over 3 seeds
    agg["baseline_score"] = agg["baseline_f1"]
    agg["improved_score"] = agg["improved_f1"]
    agg["improvement"] = agg["improved_score"] - agg["baseline_score"]
    vals = list(agg.values())
    if not np.all(np.isfinite(vals)) or len(np.unique(yte)) < 2:
        raise SystemExit("could not produce meaningful metrics")

    names = ["F1", "Detection rate", "False alarm rate"]
    keys = ["f1", "detection_rate", "false_alarm_rate"]
    x = np.arange(3)
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, [agg[f"baseline_{k}"] for k in keys], 0.4, label="K-means + SVM (baseline)")
    plt.bar(x + 0.2, [agg[f"improved_{k}"] for k in keys], 0.4, label="Balanced LightGBM (improved)")
    plt.xticks(x, names)
    plt.ylabel("Test value (mean over seeds)")
    plt.legend()
    plt.tight_layout()
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/baseline_vs_improved.png", dpi=120)

    results = {
        "question": "Can a class-balanced LightGBM with a CV-tuned threshold beat a K-means-filtered SVM hybrid at detecting intrusions on NSL-KDD while producing fewer false alarms?",
        "method": ("Train on a fixed 20,000-row subsample of KDDTrain+, test once on KDDTest+. "
                   "Both models use one-hot + min-max features, a 5-fold stratified CV threshold "
                   "maximising F1 on training data, and seeds 42-44."),
        "dataset": {"name": "NSL-KDD (KDDTrain+ / KDDTest+)", "source_url": BASE,
                    "n_rows": int(n_rows), "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "same_as_paper": True, "baseline_is_papers_method": False},
        "metric_name": "malicious-class F1 on KDDTest+",
        "higher_is_better": True,
        "baseline": "K-means (k=20) label-purity filter followed by an RBF SVM (paper-style hybrid).",
        "improved": "Class-balanced LightGBM (class_weight='balanced') on the same features, same threshold protocol.",
        "metrics": {k: round(v, 6) for k, v in agg.items()},
        "notes": ("KDDTest+ is used as the test set as specified, so the split is the predefined "
                  "NSL-KDD split (about 15% test), not a 20-30% resplit. Training uses a 20,000-row "
                  "subsample of KDDTrain+. The threshold is chosen on out-of-fold training scores "
                  "only. Std is the population std over 3 seeds. The SVM is the main runtime cost. "
                  "The test set has known distribution shift (novel attacks), and results are not "
                  "a claim about the original 2015 paper's numbers, which are not stated."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s. Baseline F1 {agg['baseline_f1']:.4f}, "
          f"improved F1 {agg['improved_f1']:.4f}. Wrote results.json.")


if __name__ == "__main__":
    main()
