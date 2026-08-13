"""Feasibility Experiment 2: Drift-Resilient TabPFN on Adult/temporal split.

Tests whether Drift-Resilient TabPFN (NeurIPS 2024) outperforms standard TabPFN
on temporal distribution shift, using its pre-trained dist-shift-aware models.

Key design:
  - Load 3 dist models (tabpfn_dist_model_1/2/3) with best_dist config
  - Load 3 base models (tabpfn_base_model_1/2/3) with best_base config
  - Construct dist_shift_domain from temporal ordering
  - Compare: TabPFN-base vs TabPFN-dist vs XGBoost

Results saved to: results/feasibility_exp2_drift_resilient.json
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
    """Construct dist_shift_domain from temporal ordering.

    For temporal split, training data is sorted by temporal column (age for Adult).
    We divide training set into n_train_domains equal-sized bins (domains 0..N-1),
    and assign test set to domain N (the "future" domain).

    Returns:
        train_domain: torch.LongTensor of shape (n_train,)
        test_domain: torch.LongTensor of shape (n_test,)
    """
    n_train = len(split_data['X_train'])
    n_test = len(split_data['X_test'])

    # Training samples are already in temporal order (sorted by temporal_col).
    # Assign domain indices: divide training set into n_train_domains bins.
    train_domain = np.zeros(n_train, dtype=np.int64)
    domain_size = n_train // n_train_domains
    for d in range(n_train_domains):
        start = d * domain_size
        end = (d + 1) * domain_size if d < n_train_domains - 1 else n_train
        train_domain[start:end] = d

    # Test set is the "future" domain
    test_domain = np.full(n_test, n_train_domains, dtype=np.int64)

    return torch.LongTensor(train_domain), torch.LongTensor(test_domain)


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
            debug=True,  # Small N_ensemble for speed (2 instead of 32)
            device="auto"
        )
        model.show_progress = False
        model.seed = 42
        return model

    dist_models = []
    base_models = []
    for i in [1, 2, 3]:
        print(f"  Loading tabpfn_dist_model_{i}...")
        m = get_model(f"tabpfn_dist_model_{i}", "best_dist")
        dist_models.append(m)
        print(f"  Loading tabpfn_base_model_{i}...")
        m = get_model(f"tabpfn_base_model_{i}", "best_base")
        base_models.append(m)

    return dist_models, base_models


def run_model_ensemble(models, X_train, y_train, X_test, train_domain, test_domain):
    """Run ensemble of models, average predicted probabilities."""
    all_preds = []
    for i, clf in enumerate(models):
        print(f"    Model {i+1}/{len(models)}...")
        t0 = time.time()
        try:
            # Fit with dist_shift_domain
            clf.fit(
                X_train, y_train,
                additional_x={"dist_shift_domain": train_domain}
            )
            fit_time = time.time() - t0

            # Predict
            t0 = time.time()
            preds = clf.predict_proba(
                X_test,
                additional_x={"dist_shift_domain": test_domain}
            )
            predict_time = time.time() - t0

            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            all_preds.append(preds)
            print(f"      fit={fit_time:.1f}s, predict={predict_time:.1f}s")
        except Exception as e:
            print(f"      FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_preds:
        return None, None

    # Average probabilities across models
    avg_proba = np.mean(all_preds, axis=0)
    y_pred = np.argmax(avg_proba, axis=1)
    return y_pred, avg_proba


def main():
    print("=" * 78)
    print("Feasibility Exp 2: Drift-Resilient TabPFN on Adult/temporal split")
    print("=" * 78)

    results = {
        'experiment': 'feasibility_exp2_drift_resilient',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': 'Test Drift-Resilient TabPFN (NeurIPS 2024) on Adult/temporal',
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
        results['error'] = str(e)
        with open(os.path.join(RESULT_DIR, 'feasibility_exp2_drift_resilient.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return

    # ---- Prepare data ----
    print("\n[2/3] Preparing Adult/temporal split...")
    split_data = prepare_split('adult', 'temporal', seed=42)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    info = split_data['split_info']
    print(f"  train={X_train.shape}, test={X_test.shape}, "
          f"features={info['n_features']}, classes={info['n_classes']}")

    # ---- Construct dist_shift_domain ----
    train_domain, test_domain = construct_dist_shift_domain(split_data, n_train_domains=5)
    print(f"  dist_shift_domain: train has {len(torch.unique(train_domain))} domains "
          f"({torch.unique(train_domain).tolist()}), test domain={test_domain[0].item()}")

    # ---- Run experiments ----
    print("\n[3/3] Running experiments...")

    # --- TabPFN-dist (Drift-Resilient) ---
    print("\n  [a] TabPFN-dist (Drift-Resilient, 3-model ensemble)...")
    y_pred, y_proba = run_model_ensemble(
        dist_models, X_train, y_train, X_test, train_domain, test_domain
    )
    if y_pred is not None:
        metrics = compute_metrics(y_test, y_pred, y_proba)
        print(f"      acc={metrics['accuracy']:.4f}, f1m={metrics['f1_macro']:.4f}, "
              f"auc={metrics['auc']:.4f}")
        results['results'].append({
            'dataset': 'adult', 'split': 'temporal', 'seed': 42,
            'method': 'TabPFN-dist', 'metrics': metrics,
            'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
            'n_dist_models': len(dist_models),
            'note': 'Drift-Resilient TabPFN with dist_shift_domain'
        })

    # --- TabPFN-base (standard, no dist-shift awareness) ---
    print("\n  [b] TabPFN-base (standard, 3-model ensemble)...")
    y_pred, y_proba = run_model_ensemble(
        base_models, X_train, y_train, X_test, train_domain, test_domain
    )
    if y_pred is not None:
        metrics = compute_metrics(y_test, y_pred, y_proba)
        print(f"      acc={metrics['accuracy']:.4f}, f1m={metrics['f1_macro']:.4f}, "
              f"auc={metrics['auc']:.4f}")
        results['results'].append({
            'dataset': 'adult', 'split': 'temporal', 'seed': 42,
            'method': 'TabPFN-base', 'metrics': metrics,
            'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
            'n_base_models': len(base_models),
            'note': 'Standard TabPFN base models (no dist-shift awareness)'
        })

    # ---- Also run on IID split for comparison ----
    print("\n  [c] Running on Adult/IID for comparison...")
    split_iid = prepare_split('adult', 'iid', seed=42)
    X_train_iid = split_iid['X_train']
    y_train_iid = split_iid['y_train']
    X_test_iid = split_iid['X_test']
    y_test_iid = split_iid['y_test']

    # For IID, dist_shift_domain is less meaningful but still provide it
    train_domain_iid, test_domain_iid = construct_dist_shift_domain(split_iid, n_train_domains=5)

    print("\n  [d] TabPFN-dist on IID...")
    y_pred, y_proba = run_model_ensemble(
        dist_models, X_train_iid, y_train_iid, X_test_iid, train_domain_iid, test_domain_iid
    )
    if y_pred is not None:
        metrics = compute_metrics(y_test_iid, y_pred, y_proba)
        print(f"      acc={metrics['accuracy']:.4f}, f1m={metrics['f1_macro']:.4f}, "
              f"auc={metrics['auc']:.4f}")
        results['results'].append({
            'dataset': 'adult', 'split': 'iid', 'seed': 42,
            'method': 'TabPFN-dist', 'metrics': metrics,
            'n_train': int(len(y_train_iid)), 'n_test': int(len(y_test_iid)),
            'note': 'Drift-Resilient TabPFN on IID (dist_shift_domain still provided)'
        })

    print("\n  [e] TabPFN-base on IID...")
    y_pred, y_proba = run_model_ensemble(
        base_models, X_train_iid, y_train_iid, X_test_iid, train_domain_iid, test_domain_iid
    )
    if y_pred is not None:
        metrics = compute_metrics(y_test_iid, y_pred, y_proba)
        print(f"      acc={metrics['accuracy']:.4f}, f1m={metrics['f1_macro']:.4f}, "
              f"auc={metrics['auc']:.4f}")
        results['results'].append({
            'dataset': 'adult', 'split': 'iid', 'seed': 42,
            'method': 'TabPFN-base', 'metrics': metrics,
            'n_train': int(len(y_train_iid)), 'n_test': int(len(y_test_iid)),
            'note': 'Standard TabPFN base models on IID'
        })

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Split':<12} {'Method':<16} {'Accuracy':<10} {'F1-Macro':<10} {'AUC':<10}")
    print("-" * 60)
    for r in results['results']:
        m = r['metrics']
        print(f"{r['split']:<12} {r['method']:<16} "
              f"{m['accuracy']:.4f}     {m['f1_macro']:.4f}     {m['auc']:.4f}")

    with open(os.path.join(RESULT_DIR, 'feasibility_exp2_drift_resilient.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {os.path.join(RESULT_DIR, 'feasibility_exp2_drift_resilient.json')}")
    print("=" * 78)
    print("Feasibility Exp 2 Complete!")
    print("=" * 78)


if __name__ == '__main__':
    main()
