"""NMF top-sentence baseline vs LightGBM sentence scorer on CNN/DailyMail (ROUGE-1 F1)."""
import json, re, sys, time
from collections import Counter

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split

SEED = 42
SEEDS = [42, 43, 44]
N_ROWS = 700            # articles fetched from the CNN/DM test split
TEST_FRAC = 0.3         # article-level split (within 20-30% test)
K_TOPICS = 3            # NMF topics per document
N_PICK = 3              # sentences per summary
MAX_ORACLE = 4          # max positive sentences per article in the greedy oracle
API = "https://datasets-server.huggingface.co/rows"
# Candidate (dataset, config) pairs for the same CNN/DailyMail 3.0.0 test articles.
# ccdv/cnn_dailymail returned 404 for both config names on the datasets-server,
# so the official abisee/cnn_dailymail 3.0.0 mirror is tried as a fallback.
CANDIDATES = [
    ("ccdv/cnn_dailymail", "default"),
    ("ccdv/cnn_dailymail", "3.0.0"),
    ("abisee/cnn_dailymail", "3.0.0"),
]
GRID = [dict(num_leaves=7, n_estimators=100), dict(num_leaves=15, n_estimators=200),
        dict(num_leaves=31, n_estimators=300)]


class _NotFound(Exception):
    pass


def _fetch_config(n, dataset, config):
    """Fetch n rows of the test split for one dataset/config (100 per request), retry on 429/5xx."""
    rows, off = [], 0
    while len(rows) < n:
        params = dict(dataset=dataset, config=config, split="test",
                      offset=off, length=min(100, n - len(rows)))
        for attempt in range(6):
            r = requests.get(API, params=params, timeout=60)
            if r.status_code == 404:
                raise _NotFound(f"{dataset}/{config}")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError("datasets-server kept failing")
        batch = [x["row"] for x in r.json()["rows"]]
        if not batch:
            break
        rows += batch
        off += len(batch)
        print(f"fetched {len(rows)} rows (dataset={dataset}, config={config})")
    return rows[:n]


def fetch_rows(n):
    """Fetch n test-split rows, trying each candidate dataset/config on 404.
    Returns (rows, dataset, config) for the first candidate that works."""
    for dataset, config in CANDIDATES:
        try:
            return _fetch_config(n, dataset, config), dataset, config
        except _NotFound:
            print(f"{dataset} config '{config}' not found; trying next")
    raise RuntimeError(f"no working candidate among {CANDIDATES}")


def split_sents(text):
    """Simple regex sentence splitter."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 2]


def tok(s):
    """Lowercased alphanumeric tokens, shared by ROUGE and length features."""
    return re.findall(r"[a-z0-9]+", s.lower())


def ngram_f1(pred, ref, n):
    """ROUGE-N F1 from clipped n-gram overlap."""
    p = Counter(tuple(pred[i:i + n]) for i in range(len(pred) - n + 1))
    r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def oracle_labels(sent_toks, ref_toks):
    """Greedy ROUGE-1 oracle: add the sentence that most improves F1, up to MAX_ORACLE."""
    chosen, best = [], 0.0
    while len(chosen) < MAX_ORACLE:
        cands = []
        for i in range(len(sent_toks)):
            if i in chosen:
                continue
            pred = sum([sent_toks[j] for j in sorted(chosen + [i])], [])
            cands.append((ngram_f1(pred, ref_toks, 1), i))
        if not cands:
            break
        score_i, i = max(cands)
        if score_i <= best:
            break
        best = score_i
        chosen.append(i)
    y = np.zeros(len(sent_toks), dtype=int)
    y[chosen] = 1
    return y


def featurize(sents, vec, seed):
    """Per-sentence features and per-document NMF loadings W (n x k)."""
    X = vec.transform(sents)
    n = len(sents)
    k = min(K_TOPICS, n)
    W = np.zeros((n, k))
    if X.nnz > 0:  # per-document NMF on the TF-IDF sentence-term matrix
        W = NMF(n_components=k, init="random", random_state=seed, max_iter=300).fit_transform(X)
    Wn = W / (W.sum(1, keepdims=True) + 1e-12)           # normalised topic loadings
    P = np.zeros((n, K_TOPICS))
    P[:, :k] = Wn                                        # zero-pad to fixed width
    cent = np.asarray(X.mean(axis=0)).ravel()            # TF-IDF document centroid
    cen = X @ cent / (np.linalg.norm(cent) + 1e-12)      # centrality (cosine to centroid)
    rank = np.argsort(np.argsort(-cen)) / max(n - 1, 1)  # centrality rank in document
    L = np.array([len(tok(s)) for s in sents])
    pos = np.arange(n)
    F = np.column_stack([pos, pos / max(n - 1, 1), np.full(n, n), L, cen, rank, P, P.max(1)])
    return F, W


def nmf_pick(W):
    """Baseline: top-weighted sentence per NMF topic, filled up to N_PICK by total weight."""
    m = min(N_PICK, len(W))
    picks = []
    for j in range(W.shape[1]):
        i = int(np.argmax(W[:, j]))
        if i not in picks:
            picks.append(i)
    for i in np.argsort(-W.sum(1), kind="stable"):
        if len(picks) >= m:
            break
        if int(i) not in picks:
            picks.append(int(i))
    return sorted(picks[:m])


def lgb_pick(model, F):
    """Improved: the N_PICK sentences with highest predicted probability."""
    p = model.predict_proba(F)[:, 1]
    return sorted(np.argsort(-p, kind="stable")[:min(N_PICK, len(F))].tolist())


def score(docs, picks):
    """Mean ROUGE-1 F1 and ROUGE-2 F1 against reference highlights."""
    r1, r2 = [], []
    for d, p in zip(docs, picks):
        pred = sum([d["toks"][i] for i in p], [])
        ref = tok(d["ref"])
        r1.append(ngram_f1(pred, ref, 1))
        r2.append(ngram_f1(pred, ref, 2))
    return float(np.mean(r1)), float(np.mean(r2))


def fit_lgb(docs, params, seed):
    X = np.vstack([d["F"] for d in docs])
    y = np.concatenate([d["y"] for d in docs])
    m = lgb.LGBMClassifier(learning_rate=0.05, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, verbose=-1, **params)
    return m.fit(X, y)


def cv_score(docs, params):
    """3-fold CV on training articles only (ROUGE-1 of top-3 selection)."""
    out = []
    for a, b in KFold(3, shuffle=True, random_state=SEED).split(docs):
        m = fit_lgb([docs[i] for i in a], params, SEED)
        held = [docs[i] for i in b]
        out.append(score(held, [lgb_pick(m, d["F"]) for d in held])[0])
    return float(np.mean(out))


def build(docs, vec, seed):
    for d in docs:
        d["F"], d["W"] = featurize(d["sents"], vec, seed)


def main():
    rows, used_dataset, used_config = fetch_rows(N_ROWS)
    docs = []
    for r in rows:
        sents = split_sents(r["article"])
        if sents:
            docs.append({"sents": sents, "toks": [tok(s) for s in sents], "ref": r["highlights"]})
    if len(docs) < 50:
        sys.exit("too few usable articles")
    tr, te = train_test_split(docs, test_size=TEST_FRAC, random_state=SEED)
    print(f"articles: {len(docs)} (train {len(tr)}, test {len(te)})")

    # Shared TF-IDF vocabulary fit on training sentences only
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    vec.fit([s for d in tr for s in d["sents"]])
    for d in tr:
        d["y"] = oracle_labels(d["toks"], tok(d["ref"]))
    print(f"oracle positives per train article: {np.mean([d['y'].sum() for d in tr]):.2f}")

    # Small hyper-parameter search, CV on training set only
    build(tr, vec, SEED)
    cvs = [cv_score(tr, g) for g in GRID]
    for g, s in zip(GRID, cvs):
        print(f"CV ROUGE-1 {g}: {s:.4f}")
    best = GRID[int(np.argmax(cvs))]
    print(f"selected: {best}")

    # Final evaluation on the held-out test articles, seeds 42-44
    b1, b2, i1, i2 = [], [], [], []
    for s in SEEDS:
        build(tr + te, vec, s)
        r1, r2 = score(te, [nmf_pick(d["W"]) for d in te])
        b1.append(r1); b2.append(r2)
        m = fit_lgb(tr, best, s)
        r1, r2 = score(te, [lgb_pick(m, d["F"]) for d in te])
        i1.append(r1); i2.append(r2)
        print(f"seed {s}: baseline R1={b1[-1]:.4f} improved R1={i1[-1]:.4f}")

    b1, i1, b2, i2 = map(np.array, (b1, i1, b2, i2))
    res = {
        "question": "Can a LightGBM sentence scorer trained on sentence features beat a standard per-document NMF sentence-selection summarizer on ROUGE-1 F1 for CNN/DailyMail news articles?",
        "method": "Articles from CNN/DM 3.0.0 test split via datasets-server; article-level train/test split; greedy ROUGE-1 oracle labels; per-document NMF top-sentence baseline vs LightGBM top-3 probability selection; ROUGE-1 F1 on test; seeds 42-44.",
        "dataset": {"name": f"CNN/DailyMail 3.0.0 ({used_dataset}, config {used_config}, test split rows)",
                    "source_url": f"https://huggingface.co/datasets/{used_dataset}",
                    "n_rows": len(docs), "n_train": len(tr), "n_test": len(te),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "ROUGE-1 F1 (mean over test articles, vs reference highlights)",
        "higher_is_better": True,
        "baseline": "Per-document NMF on TF-IDF sentence-term matrix; pick top-weighted sentence per topic (3 topics, 3 sentences). Stand-in for simplicial NMF, not a faithful reimplementation.",
        "improved": "LightGBM classifier on sentence features (position, length, TF-IDF centrality, per-document NMF topic loadings) trained on greedy ROUGE-1 oracle labels; selects 3 highest-probability sentences.",
        "metrics": {
            "baseline_score": float(b1.mean()), "improved_score": float(i1.mean()),
            "improvement": float(i1.mean() - b1.mean()),
            "baseline_score_std": float(b1.std()), "improved_score_std": float(i1.std()),
            "baseline_rouge2_f1": float(b2.mean()), "improved_rouge2_f1": float(i2.mean()),
            "improvement_rouge2_f1": float(i2.mean() - b2.mean()),
            "cv_best_rouge1": float(max(cvs)), "selected_num_leaves": best["num_leaves"],
        },
        "notes": (f"Single run of data split (seed 42); seeds 42-44 vary NMF init and LightGBM. "
                  f"Split is 70/30 of ~700 articles rather than exactly 400/200 to satisfy the 20-30% test-size rule. "
                  f"Articles come from the CNN/DM test split ({used_dataset}, config {used_config}), not the paper's dataset. "
                  "Baseline is a stand-in for simplicial NMF. Std is population std over 3 seeds. "
                  "Only a 3-point hyper-parameter grid was searched, using CV on training articles. "
                  "Dataset source is resolved at runtime: ccdv/cnn_dailymail configs 'default' and '3.0.0' "
                  "returned 404 on the datasets-server, so it falls back to abisee/cnn_dailymail 3.0.0 (same articles)."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    import os
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar(["baseline (NMF)", "improved (LightGBM)"], [b1.mean(), i1.mean()],
            yerr=[b1.std(), i1.std()], capsize=6, color=["gray", "steelblue"])
    plt.ylabel("ROUGE-1 F1 (test)")
    plt.tight_layout()
    plt.savefig("figures/rouge1_bar.png", dpi=120)
    print("wrote results.json and figures/rouge1_bar.png")


if __name__ == "__main__":
    main()
