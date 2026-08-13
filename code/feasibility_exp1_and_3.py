"""Feasibility Experiments 1 & 3: TabPFN temporal degradation + KNN context selection.

Exp1: Confirm TabPFN v2 performance degradation on Adult/temporal split.
      Compare with XGBoost to verify TabPFN is MORE vulnerable than tree models.
Exp3: Implement KNN context selection, verify it improves TabPFN's performance
      on temporal shift.

Hypothesis:
  H1: TabPFN drops more than XGBoost on temporal shift (TabPFN more vulnerable)
  H2: KNN context selection improves TabPFN on temporal shift
  H3: KNN improvement is larger on temporal than IID (shift-aware benefit)

Results saved to: results/feasibility_exp1_and_3.json
"""
import os
import sys
import json
import time
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import prepare_split

# Datasets to evaluate (Adult is primary; Telco/Bank for robustness of claim)
DATASETS_TO_TEST = ['adult', 'telco', 'bank']
SPLITS_TO_TEST = ['iid', 'temporal']
SEEDS = [42, 123, 456]
CONTEXT_SIZE = 10000  # TabPFN limit


def set_seed(seed):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute accuracy, F1-macro, AUC."""
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = 0.0
    if y_proba is not None:
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def knn_context_selection(X_train, y_train, X_test, n_select, k_neighbors=5, seed=42):
    """Select training samples whose distribution best matches test set.

    Strategy: For each test sample, find its k nearest neighbors in train.
    Count how often each train sample is selected as a neighbor.
    Pick the train samples with the highest counts (most "representative"
    of the test distribution).

    This implements the simplest form of distribution-aware context selection:
    train samples that are near many test samples are likely under the test
    distribution, so they form a better context for in-context learning.

    Args:
        X_train, y_train: training data
        X_test: test data (used ONLY to guide selection, NOT for fitting)
        n_select: number of training samples to select
        k_neighbors: k for KNN (test->train neighbor lookup)
        seed: for fallback random selection

    Returns:
        selected_indices: indices into X_train of selected samples
    """
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)

    k = min(k_neighbors, n_train)
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    nn.fit(X_train)
    # For each test sample, find its k nearest training neighbors
    _, indices = nn.kneighbors(X_test)
    # Count occurrences: train samples frequently near test points
    counter = Counter(indices.flatten())
    # Sort train samples by count (descending), take top n_select
    selected = [idx for idx, _ in counter.most_common(n_select)]
    # If counts tie or insufficient, fill with random
    if len(selected) < n_select:
        remaining = sorted(set(range(n_train)) - set(selected))
        rng = np.random.RandomState(seed)
        extra = rng.choice(remaining, n_select - len(selected), replace=False)
        selected.extend(extra.tolist())
    return np.array(selected)


def random_context_selection(X_train, n_select, seed=42):
    """Random subsampling (baseline context selection)."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)
    rng = np.random.RandomState(seed)
    return rng.choice(n_train, n_select, replace=False)


def run_tabpfn(X_train, y_train, X_test, y_test, context_indices=None, label='TabPFN'):
    """Run TabPFN (cloud client) with optional pre-selected context.

    Args:
        context_indices: if provided, use X_train[context_indices] as context.
                        If None, TabPFN does its own random subsampling.
    """
    import tabpfn_client
    if not getattr(run_tabpfn, '_initialized', False):
        tabpfn_client.init()
        run_tabpfn._initialized = True

    from tabpfn_client import TabPFNClassifier as _TabPFN

    if context_indices is not None:
        X_ctx = X_train[context_indices]
        y_ctx = y_train[context_indices]
    else:
        # Default: random subsample to CONTEXT_SIZE (replicate existing behavior)
        if X_train.shape[0] > CONTEXT_SIZE:
            idx = np.random.RandomState(42).choice(
                X_train.shape[0], CONTEXT_SIZE, replace=False
            )
            X_ctx = X_train[idx]
            y_ctx = y_train[idx]
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


def run_xgboost(X_train, y_train, X_test, y_test):
    """Run XGBoost as a reference baseline."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
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
    return metrics


def main():
    print("=" * 78)
    print("Feasibility Exp 1 & 3: TabPFN temporal degradation + KNN context selection")
    print("=" * 78)

    results = {
        'experiment': 'feasibility_exp1_and_3',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'hypotheses': {
            'H1': 'TabPFN drops more than XGBoost on temporal shift',
            'H2': 'KNN context selection improves TabPFN on temporal shift',
            'H3': 'KNN improvement is larger on temporal than IID (shift-aware benefit)',
        },
        'config': {
            'datasets': DATASETS_TO_TEST,
            'splits': SPLITS_TO_TEST,
            'seeds': SEEDS,
            'context_size': CONTEXT_SIZE,
        },
        'results': [],
    }

    for ds_name in DATASETS_TO_TEST:
        for split_type in SPLITS_TO_TEST:
            for seed in SEEDS:
                print(f"\n[{ds_name}/{split_type}/seed={seed}]")
                set_seed(seed)

                try:
                    split_data = prepare_split(ds_name, split_type, seed=seed)
                except Exception as e:
                    print(f"  ERROR split failed: {e}")
                    continue

                if split_data is None:
                    print(f"  Skip: {ds_name} does not support {split_type} split")
                    continue

                X_train = split_data['X_train']
                y_train = split_data['y_train']
                X_test = split_data['X_test']
                y_test = split_data['y_test']
                info = split_data['split_info']
                print(f"  train={X_train.shape}, test={X_test.shape}, "
                      f"features={info['n_features']}, classes={info['n_classes']}")

                # ---- XGBoost (reference) ----
                try:
                    xgb_metrics = run_xgboost(X_train, y_train, X_test, y_test)
                    print(f"  XGBoost        acc={xgb_metrics['accuracy']:.4f} "
                          f"f1m={xgb_metrics['f1_macro']:.4f}")
                    results['results'].append({
                        'dataset': ds_name, 'split': split_type, 'seed': seed,
                        'method': 'XGBoost', 'metrics': xgb_metrics,
                        'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                    })
                except Exception as e:
                    print(f"  XGBoost FAILED: {e}")

                # ---- TabPFN-Random (default random subsampling) ----
                try:
                    tabpfn_random = run_tabpfn(
                        X_train, y_train, X_test, y_test,
                        context_indices=None, label='TabPFN-Random'
                    )
                    print(f"  TabPFN-Random  acc={tabpfn_random['accuracy']:.4f} "
                          f"f1m={tabpfn_random['f1_macro']:.4f} "
                          f"(ctx={tabpfn_random['n_context']})")
                    results['results'].append({
                        'dataset': ds_name, 'split': split_type, 'seed': seed,
                        'method': 'TabPFN-Random', 'metrics': tabpfn_random,
                        'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                    })
                except Exception as e:
                    print(f"  TabPFN-Random FAILED: {e}")

                # ---- TabPFN-KNN (KNN context selection) ----
                # Only apply KNN selection if train > context_size
                if X_train.shape[0] > CONTEXT_SIZE:
                    try:
                        t0 = time.time()
                        knn_idx = knn_context_selection(
                            X_train, y_train, X_test,
                            n_select=CONTEXT_SIZE, k_neighbors=5, seed=seed
                        )
                        sel_time = time.time() - t0
                        tabpfn_knn = run_tabpfn(
                            X_train, y_train, X_test, y_test,
                            context_indices=knn_idx, label='TabPFN-KNN'
                        )
                        tabpfn_knn['selection_time'] = float(sel_time)
                        print(f"  TabPFN-KNN     acc={tabpfn_knn['accuracy']:.4f} "
                              f"f1m={tabpfn_knn['f1_macro']:.4f} "
                              f"(sel={sel_time:.1f}s, ctx={tabpfn_knn['n_context']})")
                        results['results'].append({
                            'dataset': ds_name, 'split': split_type, 'seed': seed,
                            'method': 'TabPFN-KNN', 'metrics': tabpfn_knn,
                            'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                        })
                    except Exception as e:
                        print(f"  TabPFN-KNN FAILED: {e}")
                else:
                    # Small dataset: TabPFN uses full train, KNN selection not needed
                    print(f"  TabPFN-KNN     SKIPPED (train={X_train.shape[0]} <= {CONTEXT_SIZE})")
                    results['results'].append({
                        'dataset': ds_name, 'split': split_type, 'seed': seed,
                        'method': 'TabPFN-KNN', 'metrics': None,
                        'skip_reason': f'train_size={X_train.shape[0]} <= context_size={CONTEXT_SIZE}',
                        'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                    })

                # Save incrementally (in case of crash)
                with open(os.path.join(RESULT_DIR, 'feasibility_exp1_and_3.json'), 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("SUMMARY: Mean over seeds")
    print("=" * 78)
    print(f"{'Dataset':<10} {'Split':<10} {'Method':<16} {'Accuracy':<10} {'F1-Macro':<10}")
    print("-" * 60)

    summary = {}
    for ds_name in DATASETS_TO_TEST:
        for split_type in SPLITS_TO_TEST:
            for method in ['XGBoost', 'TabPFN-Random', 'TabPFN-KNN']:
                accs = [r['metrics']['accuracy'] for r in results['results']
                        if r['dataset'] == ds_name and r['split'] == split_type
                        and r['method'] == method and r.get('metrics')]
                f1s = [r['metrics']['f1_macro'] for r in results['results']
                       if r['dataset'] == ds_name and r['split'] == split_type
                       and r['method'] == method and r.get('metrics')]
                if accs:
                    mean_acc = np.mean(accs)
                    mean_f1 = np.mean(f1s)
                    std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                    print(f"{ds_name:<10} {split_type:<10} {method:<16} "
                          f"{mean_acc:.4f}±{std_acc:.4f}  {mean_f1:.4f}")
                    key = f"{ds_name}_{split_type}_{method}"
                    summary[key] = {
                        'accuracy_mean': float(mean_acc),
                        'accuracy_std': float(std_acc),
                        'f1_macro_mean': float(mean_f1),
                        'n_seeds': len(accs),
                    }

    # ---- Hypothesis testing ----
    print("\n" + "=" * 78)
    print("HYPOTHESIS TESTING")
    print("=" * 78)

    # H1: TabPFN drops more than XGBoost on temporal shift
    print("\nH1: TabPFN drops more than XGBoost on temporal shift?")
    for ds_name in DATASETS_TO_TEST:
        xgb_iid = summary.get(f"{ds_name}_iid_XGBoost", {}).get('accuracy_mean')
        xgb_tmp = summary.get(f"{ds_name}_temporal_XGBoost", {}).get('accuracy_mean')
        tfn_iid = summary.get(f"{ds_name}_iid_TabPFN-Random", {}).get('accuracy_mean')
        tfn_tmp = summary.get(f"{ds_name}_temporal_TabPFN-Random", {}).get('accuracy_mean')
        if None not in (xgb_iid, xgb_tmp, tfn_iid, tfn_tmp):
            xgb_drop = xgb_iid - xgb_tmp
            tfn_drop = tfn_iid - tfn_tmp
            h1_holds = tfn_drop > xgb_drop
            print(f"  {ds_name}: XGBoost drop={xgb_drop:.4f}, TabPFN drop={tfn_drop:.4f} "
                  f"=> H1 {'HOLDS' if h1_holds else 'FAILS'} "
                  f"(TabPFN {'more' if h1_holds else 'less'} vulnerable)")

    # H2: KNN improves TabPFN on temporal shift
    print("\nH2: KNN context selection improves TabPFN on temporal shift?")
    for ds_name in DATASETS_TO_TEST:
        tfn_tmp = summary.get(f"{ds_name}_temporal_TabPFN-Random", {}).get('accuracy_mean')
        knn_tmp = summary.get(f"{ds_name}_temporal_TabPFN-KNN", {}).get('accuracy_mean')
        if None not in (tfn_tmp, knn_tmp):
            delta = knn_tmp - tfn_tmp
            h2_holds = delta > 0
            print(f"  {ds_name}: TabPFN-Random={tfn_tmp:.4f}, TabPFN-KNN={knn_tmp:.4f} "
                  f"=> Δ={delta:+.4f}, H2 {'HOLDS' if h2_holds else 'FAILS'}")

    # H3: KNN improvement larger on temporal than IID
    print("\nH3: KNN improvement larger on temporal than IID?")
    for ds_name in DATASETS_TO_TEST:
        tfn_iid = summary.get(f"{ds_name}_iid_TabPFN-Random", {}).get('accuracy_mean')
        knn_iid = summary.get(f"{ds_name}_iid_TabPFN-KNN", {}).get('accuracy_mean')
        tfn_tmp = summary.get(f"{ds_name}_temporal_TabPFN-Random", {}).get('accuracy_mean')
        knn_tmp = summary.get(f"{ds_name}_temporal_TabPFN-KNN", {}).get('accuracy_mean')
        if None not in (tfn_iid, knn_iid, tfn_tmp, knn_tmp):
            delta_iid = knn_iid - tfn_iid
            delta_tmp = knn_tmp - tfn_tmp
            h3_holds = delta_tmp > delta_iid
            print(f"  {ds_name}: ΔIID={delta_iid:+.4f}, ΔTemporal={delta_tmp:+.4f} "
                  f"=> H3 {'HOLDS' if h3_holds else 'FAILS'} "
                  f"(shift-aware benefit {'present' if h3_holds else 'absent'})")

    results['summary'] = summary
    with open(os.path.join(RESULT_DIR, 'feasibility_exp1_and_3.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {os.path.join(RESULT_DIR, 'feasibility_exp1_and_3.json')}")
    print("=" * 78)
    print("Feasibility Exp 1 & 3 Complete!")
    print("=" * 78)


if __name__ == '__main__':
    main()
