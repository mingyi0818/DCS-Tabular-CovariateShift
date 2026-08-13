"""Orthogonality Experiment: DCS-Logistic context selection + Drift-Resilient TabPFN.

Tests whether ContextShield's DCS-Logistic selection is orthogonal to
Drift-Resilient TabPFN's dist-shift-aware pre-training.

If the improvements stack (DCS helps dist models), this proves the two methods
address different aspects of distribution shift:
  - Drift-Resilient: modifies the synthetic prior at pre-training time
  - ContextShield (DCS): optimizes the context at inference time

Experimental design (Adult/temporal, 3 seeds):
  1. TabPFN-base + Random context     (baseline)
  2. TabPFN-base + DCS-Logistic context
  3. TabPFN-dist + Random context     (Drift-Resilient baseline)
  4. TabPFN-dist + DCS-Logistic context (orthogonality test)

If (4) > (3) > (1) and (4) > (2) > (1), the methods are orthogonal.

Results saved to: results/orthogonality_exp_results.json
"""
import os
import sys
import json
import time
import numpy as np
import torch

# Add Drift-Resilient TabPFN to path
DRIFT_TABPFN_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference', 'Drift-Resilient_TabPFN-main')
sys.path.insert(0, DRIFT_TABPFN_PATH)

# Add our code to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR
from splits import prepare_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEEDS = [42, 123, 456]
CONTEXT_SIZE = 10000


def json_safe(obj):
    """Recursively convert numpy types to Python native types."""
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


def construct_dist_shift_domain(split_data, n_train_domains=5):
    """Construct dist_shift_domain from temporal ordering."""
    n_train = len(split_data['X_train'])
    n_test = len(split_data['X_test'])
    train_domain = np.zeros(n_train, dtype=np.int64)
    domain_size = n_train // n_train_domains
    for d in range(n_train_domains):
        start = d * domain_size
        end = (d + 1) * domain_size if d < n_train_domains - 1 else n_train
        train_domain[start:end] = d
    test_domain = np.full(n_test, n_train_domains, dtype=np.int64)
    return torch.LongTensor(train_domain), torch.LongTensor(test_domain)


def estimate_density_ratio(X_train, X_test, seed=42):
    """Estimate density ratio using logistic regression domain classifier."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if n_test > 5000:
        rng = np.random.RandomState(seed)
        test_idx = rng.choice(n_test, 5000, replace=False)
        X_test_sample = X_test[test_idx]
    else:
        X_test_sample = X_test

    X_domain = np.vstack([X_train, X_test_sample])
    y_domain = np.concatenate([np.zeros(n_train), np.ones(len(X_test_sample))])

    scaler = StandardScaler()
    X_domain_s = scaler.fit_transform(X_domain)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_domain_s, y_domain)
    p_test = clf.predict_proba(scaler.transform(X_train))[:, 1]
    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    return p_test / (1 - p_test)


def dcs_logistic_selection(X_train, X_test, n_select, n_clusters=50, seed=42):
    """DCS-Logistic selection (best method from context_shield experiments)."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)

    density_ratios = estimate_density_ratio(X_train, X_test, seed=seed)
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


def random_selection(X_train, n_select, seed=42):
    """Random context selection."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)
    rng = np.random.RandomState(seed)
    return rng.choice(n_train, n_select, replace=False)


def load_drift_resilient_models():
    """Load dist and base models for Drift-Resilient TabPFN."""
    from importlib import resources
    import tabpfn
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig

    libpath = str(resources.files(tabpfn))

    def get_model(model_path, model_type):
        model_path_config = TabPFNModelPathsConfig(
            paths=[f"{libpath}/model_cache/{model_path}.cpkt"],
            task_type="dist_shift_multiclass"
        )
        model = get_best_tabpfn(
            task_type="dist_shift_multiclass",
            model_type=model_type,
            paths_config=model_path_config,
            debug=True,
            device="auto"
        )
        model.show_progress = False
        model.seed = 42
        return model

    dist_models = []
    base_models = []
    for i in [1, 2, 3]:
        print(f"  Loading tabpfn_dist_model_{i}...")
        dist_models.append(get_model(f"tabpfn_dist_model_{i}", "best_dist"))
        print(f"  Loading tabpfn_base_model_{i}...")
        base_models.append(get_model(f"tabpfn_base_model_{i}", "best_base"))

    return dist_models, base_models


def run_model_ensemble(models, X_train, y_train, X_test, train_domain, test_domain):
    """Run ensemble of models, average predicted probabilities."""
    all_preds = []
    for i, clf in enumerate(models):
        t0 = time.time()
        try:
            clf.fit(
                X_train, y_train,
                additional_x={"dist_shift_domain": train_domain}
            )
            fit_time = time.time() - t0

            t0 = time.time()
            preds = clf.predict_proba(
                X_test,
                additional_x={"dist_shift_domain": test_domain}
            )
            predict_time = time.time() - t0

            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            all_preds.append(preds)
            print(f"    Model {i+1}/{len(models)}: fit={fit_time:.1f}s, predict={predict_time:.1f}s")
        except Exception as e:
            print(f"    Model {i+1}/{len(models)} FAILED: {e}")
            continue

    if not all_preds:
        return None, None

    avg_proba = np.mean(all_preds, axis=0)
    y_pred = np.argmax(avg_proba, axis=1)
    return y_pred, avg_proba


def main():
    print("=" * 80)
    print("Orthogonality Experiment: DCS-Logistic + Drift-Resilient TabPFN")
    print("=" * 80)

    results = {
        'experiment': 'orthogonality_exp',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': 'Test if DCS-Logistic context selection is orthogonal to Drift-Resilient TabPFN',
        'config': {
            'dataset': 'adult',
            'split': 'temporal',
            'seeds': SEEDS,
            'context_size': CONTEXT_SIZE,
            'methods': [
                'TabPFN-base-Random',
                'TabPFN-base-DCS-Logistic',
                'TabPFN-dist-Random',
                'TabPFN-dist-DCS-Logistic',
            ],
        },
        'results': [],
    }

    # ---- Load models ----
    print("\n[1/3] Loading Drift-Resilient TabPFN models...")
    try:
        dist_models, base_models = load_drift_resilient_models()
        print(f"  Loaded {len(dist_models)} dist models, {len(base_models)} base models")
    except Exception as e:
        print(f"  FATAL: Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---- Run experiments ----
    print("\n[2/3] Running orthogonality experiments...")

    for seed in SEEDS:
        print(f"\n[seed={seed}]")
        np.random.seed(seed)
        torch.manual_seed(seed)

        split_data = prepare_split('adult', 'temporal', seed=seed)
        X_train = split_data['X_train']
        y_train = split_data['y_train']
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        print(f"  train={X_train.shape}, test={X_test.shape}")

        train_domain_full, test_domain = construct_dist_shift_domain(split_data, n_train_domains=5)

        # --- 1. TabPFN-base + Random ---
        print("  [1] TabPFN-base + Random context...")
        try:
            t0 = time.time()
            random_idx = random_selection(X_train, CONTEXT_SIZE, seed=seed)
            sel_time = time.time() - t0
            X_ctx = X_train[random_idx]
            y_ctx = y_train[random_idx]
            train_domain_ctx = train_domain_full[random_idx]
            y_pred, y_proba = run_model_ensemble(
                base_models, X_ctx, y_ctx, X_test, train_domain_ctx, test_domain
            )
            if y_pred is not None:
                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics['selection_time'] = float(sel_time)
                metrics['n_context'] = int(len(y_ctx))
                print(f"      acc={metrics['accuracy']:.4f} f1m={metrics['f1_macro']:.4f}")
                results['results'].append({
                    'dataset': 'adult', 'split': 'temporal', 'seed': seed,
                    'method': 'TabPFN-base-Random', 'metrics': metrics,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                })
        except Exception as e:
            print(f"      FAILED: {e}")

        # --- 2. TabPFN-base + DCS-Logistic ---
        print("  [2] TabPFN-base + DCS-Logistic context...")
        try:
            t0 = time.time()
            dcs_idx = dcs_logistic_selection(X_train, X_test, CONTEXT_SIZE,
                                              n_clusters=50, seed=seed)
            sel_time = time.time() - t0
            X_ctx = X_train[dcs_idx]
            y_ctx = y_train[dcs_idx]
            train_domain_ctx = train_domain_full[dcs_idx]
            y_pred, y_proba = run_model_ensemble(
                base_models, X_ctx, y_ctx, X_test, train_domain_ctx, test_domain
            )
            if y_pred is not None:
                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics['selection_time'] = float(sel_time)
                metrics['n_context'] = int(len(y_ctx))
                print(f"      acc={metrics['accuracy']:.4f} f1m={metrics['f1_macro']:.4f}")
                results['results'].append({
                    'dataset': 'adult', 'split': 'temporal', 'seed': seed,
                    'method': 'TabPFN-base-DCS-Logistic', 'metrics': metrics,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                })
        except Exception as e:
            print(f"      FAILED: {e}")

        # --- 3. TabPFN-dist + Random ---
        print("  [3] TabPFN-dist + Random context...")
        try:
            t0 = time.time()
            random_idx = random_selection(X_train, CONTEXT_SIZE, seed=seed)
            sel_time = time.time() - t0
            X_ctx = X_train[random_idx]
            y_ctx = y_train[random_idx]
            train_domain_ctx = train_domain_full[random_idx]
            y_pred, y_proba = run_model_ensemble(
                dist_models, X_ctx, y_ctx, X_test, train_domain_ctx, test_domain
            )
            if y_pred is not None:
                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics['selection_time'] = float(sel_time)
                metrics['n_context'] = int(len(y_ctx))
                print(f"      acc={metrics['accuracy']:.4f} f1m={metrics['f1_macro']:.4f}")
                results['results'].append({
                    'dataset': 'adult', 'split': 'temporal', 'seed': seed,
                    'method': 'TabPFN-dist-Random', 'metrics': metrics,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                })
        except Exception as e:
            print(f"      FAILED: {e}")

        # --- 4. TabPFN-dist + DCS-Logistic (ORTHOGONALITY TEST) ---
        print("  [4] TabPFN-dist + DCS-Logistic context (ORTHOGONALITY)...")
        try:
            t0 = time.time()
            dcs_idx = dcs_logistic_selection(X_train, X_test, CONTEXT_SIZE,
                                              n_clusters=50, seed=seed)
            sel_time = time.time() - t0
            X_ctx = X_train[dcs_idx]
            y_ctx = y_train[dcs_idx]
            train_domain_ctx = train_domain_full[dcs_idx]
            y_pred, y_proba = run_model_ensemble(
                dist_models, X_ctx, y_ctx, X_test, train_domain_ctx, test_domain
            )
            if y_pred is not None:
                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics['selection_time'] = float(sel_time)
                metrics['n_context'] = int(len(y_ctx))
                print(f"      acc={metrics['accuracy']:.4f} f1m={metrics['f1_macro']:.4f}")
                results['results'].append({
                    'dataset': 'adult', 'split': 'temporal', 'seed': seed,
                    'method': 'TabPFN-dist-DCS-Logistic', 'metrics': metrics,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                })
        except Exception as e:
            print(f"      FAILED: {e}")

        # Save incrementally
        with open(os.path.join(RESULT_DIR, 'orthogonality_exp_results.json'), 'w') as f:
            json.dump(json_safe(results), f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: Mean over seeds")
    print("=" * 80)
    print(f"{'Method':<35} {'Accuracy':<14} {'F1-Macro':<10} {'Δ vs base-Random':<18}")
    print("-" * 80)

    methods = ['TabPFN-base-Random', 'TabPFN-base-DCS-Logistic',
               'TabPFN-dist-Random', 'TabPFN-dist-DCS-Logistic']
    summary = {}
    for method in methods:
        accs = [r['metrics']['accuracy'] for r in results['results']
                if r['method'] == method and r.get('metrics')]
        f1s = [r['metrics']['f1_macro'] for r in results['results']
               if r['method'] == method and r.get('metrics')]
        if accs:
            mean_acc = np.mean(accs)
            mean_f1 = np.mean(f1s)
            std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
            summary[method] = {
                'accuracy_mean': float(mean_acc),
                'accuracy_std': float(std_acc),
                'f1_macro_mean': float(mean_f1),
                'n_seeds': len(accs),
            }

    base_random_acc = summary.get('TabPFN-base-Random', {}).get('accuracy_mean', 0)
    for method in methods:
        s = summary.get(method)
        if s:
            delta = (s['accuracy_mean'] - base_random_acc) * 100
            print(f"{method:<35} {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}  "
                  f"{s['f1_macro_mean']:.4f}     {delta:+.2f}pp")

    # ---- Orthogonality analysis ----
    print("\n" + "=" * 80)
    print("ORTHOGONALITY ANALYSIS")
    print("=" * 80)

    base_random = summary.get('TabPFN-base-Random', {}).get('accuracy_mean')
    base_dcs = summary.get('TabPFN-base-DCS-Logistic', {}).get('accuracy_mean')
    dist_random = summary.get('TabPFN-dist-Random', {}).get('accuracy_mean')
    dist_dcs = summary.get('TabPFN-dist-DCS-Logistic', {}).get('accuracy_mean')

    if None not in (base_random, base_dcs, dist_random, dist_dcs):
        dcs_effect_on_base = (base_dcs - base_random) * 100
        dcs_effect_on_dist = (dist_dcs - dist_random) * 100
        dist_effect_on_base = (dist_random - base_random) * 100
        dist_effect_on_dcs = (dist_dcs - base_dcs) * 100

        print(f"\n  DCS-Logistic effect on base models:  {dcs_effect_on_base:+.2f}pp")
        print(f"  DCS-Logistic effect on dist models:  {dcs_effect_on_dist:+.2f}pp")
        print(f"  Drift-Resilient effect on random:    {dist_effect_on_base:+.2f}pp")
        print(f"  Drift-Resilient effect on DCS:       {dist_effect_on_dcs:+.2f}pp")

        if dcs_effect_on_dist > 0:
            print(f"\n  ✅ ORTHOGONAL: DCS-Logistic improves dist models by {dcs_effect_on_dist:+.2f}pp")
            print(f"     Combined effect: {(dist_dcs - base_random)*100:+.2f}pp "
                  f"(dist: {dist_effect_on_base:+.2f}pp + DCS: {dcs_effect_on_base:+.2f}pp)")
        else:
            print(f"\n  ❌ NOT ORTHOGONAL: DCS-Logistic does not improve dist models")

    results['summary'] = summary
    with open(os.path.join(RESULT_DIR, 'orthogonality_exp_results.json'), 'w') as f:
        json.dump(json_safe(results), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {os.path.join(RESULT_DIR, 'orthogonality_exp_results.json')}")
    print("=" * 80)
    print("Orthogonality Experiment Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
