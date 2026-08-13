"""Mushroom DCS experiment: Run DCS on full Mushroom dataset (61,069 samples).

This script addresses reviewer issue M4/M5: Mushroom was listed in the dataset
table but never evaluated with DCS because the sample (5,000 rows) was below
TabPFN's 10,000 context limit. Using the full dataset (61,069 rows), n_train
exceeds 10,000, making DCS applicable.

Methods: Random, KNN, DCS-Logistic, DCS-LightGBM
Splits: iid, temporal
Seeds: 42, 123, 456

Results saved to: results/mushroom_dcs_results.json
"""
import os
import sys
import json
import time
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import prepare_split

CONTEXT_SIZE = 10000
N_CLUSTERS = 50
SEEDS = [42, 123, 456]
METHODS = ['TabPFN-Random', 'TabPFN-KNN', 'TabPFN-DCS-Logistic', 'TabPFN-DCS-LightGBM']


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = 0.0
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def estimate_density_ratio(X_train, X_test, method='logistic', seed=42):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    if n_test > 5000:
        rng = np.random.RandomState(seed)
        X_test_sample = X_test[rng.choice(n_test, 5000, replace=False)]
    else:
        X_test_sample = X_test

    X_domain = np.vstack([X_train, X_test_sample])
    y_domain = np.concatenate([np.zeros(n_train), np.ones(len(X_test_sample))])
    scaler = StandardScaler()
    X_domain_s = scaler.fit_transform(X_domain)

    if method == 'lightgbm':
        try:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                      random_state=seed, verbose=-1, n_jobs=-1)
            clf.fit(X_domain_s, y_domain)
            p_test = clf.predict_proba(scaler.transform(X_train))[:, 1]
        except ImportError:
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_domain_s, y_domain)
            p_test = clf.predict_proba(scaler.transform(X_train))[:, 1]
    else:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_domain_s, y_domain)
        p_test = clf.predict_proba(scaler.transform(X_train))[:, 1]

    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    return p_test / (1 - p_test)


def dcs_selection(X_train, X_test, n_select, n_clusters=50, method='logistic', seed=42):
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)

    density_ratios = estimate_density_ratio(X_train, X_test, method=method, seed=seed)
    n_clusters = min(n_clusters, n_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train)

    cluster_ratios = np.zeros(n_clusters)
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_ratios[c] = density_ratios[mask].mean() if mask.sum() > 0 else 0
        cluster_sizes[c] = mask.sum()

    cluster_weights = cluster_ratios * cluster_sizes
    total_weight = cluster_weights.sum()
    if total_weight == 0:
        allocation = np.full(n_clusters, n_select // n_clusters)
    else:
        allocation = np.maximum(1, (cluster_weights / total_weight * n_select).astype(int))

    while allocation.sum() > n_select:
        c_min = np.argmin(cluster_weights * (allocation > 1))
        allocation[c_min] -= 1
    while allocation.sum() < n_select:
        c_max = np.argmax(cluster_weights)
        allocation[c_max] += 1

    selected = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_indices = np.where(mask)[0]
        cluster_dr = density_ratios[cluster_indices]
        n_from_cluster = min(allocation[c], len(cluster_indices))
        top_local = np.argsort(cluster_dr)[-n_from_cluster:]
        selected.extend(cluster_indices[top_local].tolist())

    return np.array(selected[:n_select])


def knn_context_selection(X_train, X_test, n_select, k_neighbors=5, seed=42):
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)
    k = min(k_neighbors, n_train)
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_test)
    counter = Counter(indices.flatten())
    selected = [idx for idx, _ in counter.most_common(n_select)]
    if len(selected) < n_select:
        remaining = sorted(set(range(n_train)) - set(selected))
        rng = np.random.RandomState(seed)
        extra = rng.choice(remaining, n_select - len(selected), replace=False)
        selected.extend(extra.tolist())
    return np.array(selected)


def random_selection(X_train, n_select, seed=42):
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)
    rng = np.random.RandomState(seed)
    return rng.choice(n_train, n_select, replace=False)


def run_tabpfn(X_train, y_train, X_test, y_test, context_indices=None):
    import tabpfn_client
    if not getattr(run_tabpfn, '_initialized', False):
        tabpfn_client.init()
        run_tabpfn._initialized = True
    from tabpfn_client import TabPFNClassifier as _TabPFN

    if context_indices is not None:
        X_ctx = X_train[context_indices]
        y_ctx = y_train[context_indices]
    else:
        X_ctx = X_train
        y_ctx = y_train

    clf = _TabPFN()
    t0 = time.time()
    clf.fit(X_ctx, y_ctx)
    fit_time = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - t0
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics['fit_time'] = float(fit_time)
    metrics['predict_time'] = float(predict_time)
    metrics['n_context'] = int(len(y_ctx))
    return metrics


def main():
    print("=" * 80)
    print("Mushroom DCS Experiment: Full dataset (61,069 samples)")
    print("=" * 80)

    all_results = {
        'experiment': 'mushroom_dcs_exp',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': 'DCS on full Mushroom dataset to address reviewer M4/M5',
        'config': {
            'dataset': 'mushroom',
            'context_size': CONTEXT_SIZE,
            'n_clusters': N_CLUSTERS,
            'splits': ['iid', 'temporal'],
            'seeds': SEEDS,
            'methods': METHODS,
        },
        'results': [],
    }

    for split_type in ['iid', 'temporal']:
        for seed in SEEDS:
            print(f"\n[mushroom/{split_type}/seed={seed}] (ctx={CONTEXT_SIZE})")
            np.random.seed(seed)

            try:
                split_data = prepare_split('mushroom', split_type, seed=seed)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            if split_data is None:
                print(f"  Skip: mushroom does not support {split_type}")
                continue

            X_train = split_data['X_train']
            y_train = split_data['y_train']
            X_test = split_data['X_test']
            y_test = split_data['y_test']
            info = split_data['split_info']
            print(f"  train={X_train.shape}, test={X_test.shape}, "
                  f"features={info['n_features']}, classes={info['n_classes']}")

            # --- TabPFN-Random ---
            try:
                idx = random_selection(X_train, CONTEXT_SIZE, seed=seed)
                m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
                print(f"  TabPFN-Random         acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")
                all_results['results'].append({
                    'dataset': 'mushroom', 'split': split_type, 'seed': seed,
                    'method': 'TabPFN-Random', 'metrics': m,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                    'context_size': CONTEXT_SIZE,
                })
            except Exception as e:
                print(f"  TabPFN-Random FAILED: {e}")

            # --- TabPFN-KNN ---
            try:
                t0 = time.time()
                idx = knn_context_selection(X_train, X_test, CONTEXT_SIZE,
                                             k_neighbors=5, seed=seed)
                sel_time = time.time() - t0
                m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
                m['selection_time'] = float(sel_time)
                print(f"  TabPFN-KNN            acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")
                all_results['results'].append({
                    'dataset': 'mushroom', 'split': split_type, 'seed': seed,
                    'method': 'TabPFN-KNN', 'metrics': m,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                    'context_size': CONTEXT_SIZE,
                })
            except Exception as e:
                print(f"  TabPFN-KNN FAILED: {e}")

            # --- TabPFN-DCS-Logistic ---
            try:
                t0 = time.time()
                idx = dcs_selection(X_train, X_test, CONTEXT_SIZE,
                                    n_clusters=N_CLUSTERS, method='logistic', seed=seed)
                sel_time = time.time() - t0
                m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
                m['selection_time'] = float(sel_time)
                print(f"  TabPFN-DCS-Logistic   acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")
                all_results['results'].append({
                    'dataset': 'mushroom', 'split': split_type, 'seed': seed,
                    'method': 'TabPFN-DCS-Logistic', 'metrics': m,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                    'context_size': CONTEXT_SIZE,
                })
            except Exception as e:
                print(f"  TabPFN-DCS-Logistic FAILED: {e}")

            # --- TabPFN-DCS-LightGBM ---
            try:
                t0 = time.time()
                idx = dcs_selection(X_train, X_test, CONTEXT_SIZE,
                                    n_clusters=N_CLUSTERS, method='lightgbm', seed=seed)
                sel_time = time.time() - t0
                m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
                m['selection_time'] = float(sel_time)
                print(f"  TabPFN-DCS-LightGBM   acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")
                all_results['results'].append({
                    'dataset': 'mushroom', 'split': split_type, 'seed': seed,
                    'method': 'TabPFN-DCS-LightGBM', 'metrics': m,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                    'context_size': CONTEXT_SIZE,
                })
            except Exception as e:
                print(f"  TabPFN-DCS-LightGBM FAILED: {e}")

            # Save incrementally
            with open(os.path.join(RESULT_DIR, 'mushroom_dcs_results.json'), 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: Mean over seeds")
    print("=" * 80)

    summary = {}
    for split_type in ['iid', 'temporal']:
        for method in METHODS:
            accs = [r['metrics']['accuracy'] for r in all_results['results']
                    if r['split'] == split_type
                    and r['method'] == method and r.get('metrics')]
            f1s = [r['metrics']['f1_macro'] for r in all_results['results']
                   if r['split'] == split_type
                   and r['method'] == method and r.get('metrics')]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                mean_f1 = np.mean(f1s) if f1s else 0.0
                std_f1 = np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0
                key = f"mushroom_{split_type}_{method}"
                summary[key] = {
                    'accuracy_mean': float(mean_acc),
                    'accuracy_std': float(std_acc),
                    'f1_macro_mean': float(mean_f1),
                    'f1_macro_std': float(std_f1),
                    'n_seeds': len(accs),
                }
                print(f"  mushroom {split_type:<10} {method:<25} "
                      f"acc={mean_acc:.4f}±{std_acc:.4f}, f1={mean_f1:.4f}±{std_f1:.4f} (n={len(accs)})")

    # Improvement analysis
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS (vs TabPFN-Random)")
    print("=" * 80)
    for split_type in ['iid', 'temporal']:
        baseline_key = f"mushroom_{split_type}_TabPFN-Random"
        baseline = summary.get(baseline_key, {})
        if not baseline:
            continue
        print(f"\n  [mushroom/{split_type}] Random acc={baseline['accuracy_mean']:.4f}, f1={baseline['f1_macro_mean']:.4f}")
        for method in ['TabPFN-KNN', 'TabPFN-DCS-Logistic', 'TabPFN-DCS-LightGBM']:
            key = f"mushroom_{split_type}_{method}"
            m = summary.get(key)
            if m:
                delta_acc = (m['accuracy_mean'] - baseline['accuracy_mean']) * 100
                delta_f1 = (m['f1_macro_mean'] - baseline['f1_macro_mean']) * 100
                print(f"    {method:<25} acc={m['accuracy_mean']:.4f} ({delta_acc:+.2f}pp), "
                      f"f1={m['f1_macro_mean']:.4f} ({delta_f1:+.2f}pp)")

    all_results['summary'] = summary
    with open(os.path.join(RESULT_DIR, 'mushroom_dcs_results.json'), 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {os.path.join(RESULT_DIR, 'mushroom_dcs_results.json')}")
    print("=" * 80)
    print("Mushroom DCS Experiment Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
