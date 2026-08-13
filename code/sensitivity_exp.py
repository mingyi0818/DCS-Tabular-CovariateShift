"""Sensitivity analysis: n_clusters and context_size parameters for DCS-Logistic.

Tests:
  1. n_clusters sensitivity: [5, 10, 20, 30, 50, 100, 200] on Adult/temporal, 3 seeds
  2. context_size sensitivity: [1000, 2000, 5000, 8000, 10000] on Adult/temporal, 3 seeds

Results saved to: results/sensitivity_results.json
Also extracts timing data for complexity analysis: results/complexity_timing.json
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR, CONFIG
from splits import prepare_split
from context_shield_methods import (
    dcs_selection, random_context_selection, estimate_density_ratio,
    run_tabpfn, json_safe
)

SEEDS = [42, 123, 456]
DATASET = 'adult'
SPLIT = 'temporal'

# Sensitivity parameters
N_CLUSTERS_VALUES = [5, 10, 20, 30, 50, 100, 200]
CONTEXT_SIZE_VALUES = [1000, 2000, 5000, 8000, 10000]
DEFAULT_N_CLUSTERS = 50
DEFAULT_CONTEXT_SIZE = 10000


def run_single(X_train, y_train, X_test, y_test, n_select, n_clusters, seed):
    """Run DCS-Logistic with specific parameters."""
    t0 = time.time()
    idx = dcs_selection(X_train, X_test, n_select,
                        n_clusters=n_clusters, method='logistic', seed=seed)
    selection_time = time.time() - t0

    metrics = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
    metrics['selection_time'] = float(selection_time)
    metrics['n_clusters'] = int(n_clusters)
    metrics['n_select'] = int(n_select)
    metrics['n_context_actual'] = int(len(idx))
    return metrics


def main():
    print("=" * 80)
    print("Sensitivity Analysis: n_clusters and context_size")
    print("=" * 80)

    all_results = {
        'experiment': 'sensitivity_analysis',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'dataset': DATASET,
            'split': SPLIT,
            'seeds': SEEDS,
            'n_clusters_values': N_CLUSTERS_VALUES,
            'context_size_values': CONTEXT_SIZE_VALUES,
            'default_n_clusters': DEFAULT_N_CLUSTERS,
            'default_context_size': DEFAULT_CONTEXT_SIZE,
        },
        'n_clusters_sensitivity': [],
        'context_size_sensitivity': [],
    }

    # Prepare data once
    print("\nPreparing data...")
    split_data = prepare_split(DATASET, SPLIT, seed=42)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # === 1. n_clusters sensitivity ===
    print(f"\n[1] n_clusters sensitivity (context_size={DEFAULT_CONTEXT_SIZE})...")
    for n_clusters in N_CLUSTERS_VALUES:
        for seed in SEEDS:
            print(f"  n_clusters={n_clusters}, seed={seed}...", end=' ', flush=True)
            try:
                split_data = prepare_split(DATASET, SPLIT, seed=seed)
                X_tr = split_data['X_train']
                y_tr = split_data['y_train']
                X_te = split_data['X_test']
                y_te = split_data['y_test']

                metrics = run_single(X_tr, y_tr, X_te, y_te,
                                     DEFAULT_CONTEXT_SIZE, n_clusters, seed)
                print(f"acc={metrics['accuracy']:.4f}, sel={metrics['selection_time']:.2f}s")

                all_results['n_clusters_sensitivity'].append({
                    'n_clusters': int(n_clusters),
                    'seed': int(seed),
                    'metrics': metrics,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                all_results['n_clusters_sensitivity'].append({
                    'n_clusters': int(n_clusters),
                    'seed': int(seed),
                    'error': str(e),
                })

            # Save incrementally
            with open(os.path.join(RESULT_DIR, 'sensitivity_results.json'), 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # === 2. context_size sensitivity ===
    print(f"\n[2] context_size sensitivity (n_clusters={DEFAULT_N_CLUSTERS})...")
    for ctx_size in CONTEXT_SIZE_VALUES:
        for seed in SEEDS:
            print(f"  context_size={ctx_size}, seed={seed}...", end=' ', flush=True)
            try:
                split_data = prepare_split(DATASET, SPLIT, seed=seed)
                X_tr = split_data['X_train']
                y_tr = split_data['y_train']
                X_te = split_data['X_test']
                y_te = split_data['y_test']

                metrics = run_single(X_tr, y_tr, X_te, y_te,
                                     ctx_size, DEFAULT_N_CLUSTERS, seed)
                print(f"acc={metrics['accuracy']:.4f}, sel={metrics['selection_time']:.2f}s")

                all_results['context_size_sensitivity'].append({
                    'context_size': int(ctx_size),
                    'seed': int(seed),
                    'metrics': metrics,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                all_results['context_size_sensitivity'].append({
                    'context_size': int(ctx_size),
                    'seed': int(seed),
                    'error': str(e),
                })

            with open(os.path.join(RESULT_DIR, 'sensitivity_results.json'), 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: n_clusters sensitivity")
    print("=" * 80)
    print(f"{'n_clusters':<12} {'Accuracy':<14} {'F1-Macro':<14} {'Sel Time':<12}")
    for nc in N_CLUSTERS_VALUES:
        accs = [r['metrics']['accuracy'] for r in all_results['n_clusters_sensitivity']
                if r['n_clusters'] == nc and 'metrics' in r]
        f1s = [r['metrics']['f1_macro'] for r in all_results['n_clusters_sensitivity']
               if r['n_clusters'] == nc and 'metrics' in r]
        sels = [r['metrics']['selection_time'] for r in all_results['n_clusters_sensitivity']
                if r['n_clusters'] == nc and 'metrics' in r]
        if accs:
            print(f"{nc:<12} {np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
                  f"{np.mean(sels):.2f}s")

    print("\n" + "=" * 80)
    print("SUMMARY: context_size sensitivity")
    print("=" * 80)
    print(f"{'ctx_size':<12} {'Accuracy':<14} {'F1-Macro':<14} {'Sel Time':<12}")
    for cs in CONTEXT_SIZE_VALUES:
        accs = [r['metrics']['accuracy'] for r in all_results['context_size_sensitivity']
                if r['context_size'] == cs and 'metrics' in r]
        f1s = [r['metrics']['f1_macro'] for r in all_results['context_size_sensitivity']
               if r['context_size'] == cs and 'metrics' in r]
        sels = [r['metrics']['selection_time'] for r in all_results['context_size_sensitivity']
                if r['context_size'] == cs and 'metrics' in r]
        if accs:
            print(f"{cs:<12} {np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
                  f"{np.mean(sels):.2f}s")

    # === Extract timing for complexity analysis ===
    print("\n" + "=" * 80)
    print("Extracting timing data for complexity analysis...")
    print("=" * 80)

    # Load context_shield_results for timing comparison
    cs_path = os.path.join(RESULT_DIR, 'context_shield_results.json')
    timing_data = {'methods': {}}
    if os.path.exists(cs_path):
        with open(cs_path, 'r') as f:
            cs_data = json.load(f)
        for r in cs_data['results']:
            if r['split'] == 'temporal' and r.get('metrics'):
                method = r['method']
                if method not in timing_data['methods']:
                    timing_data['methods'][method] = {
                        'selection_times': [],
                        'fit_times': [],
                        'predict_times': [],
                    }
                m = r['metrics']
                timing_data['methods'][method]['selection_times'].append(
                    m.get('selection_time', 0))
                timing_data['methods'][method]['fit_times'].append(m.get('fit_time', 0))
                timing_data['methods'][method]['predict_times'].append(
                    m.get('predict_time', 0))

        # Compute averages
        print(f"\n{'Method':<37} {'Sel Time':<12} {'Fit Time':<12} {'Pred Time':<12} {'Total':<12}")
        for method, times in sorted(timing_data['methods'].items()):
            sel = np.mean(times['selection_times'])
            fit = np.mean(times['fit_times'])
            pred = np.mean(times['predict_times'])
            total = sel + fit + pred
            timing_data['methods'][method]['mean_selection_time'] = float(sel)
            timing_data['methods'][method]['mean_fit_time'] = float(fit)
            timing_data['methods'][method]['mean_predict_time'] = float(pred)
            timing_data['methods'][method]['mean_total_time'] = float(total)
            print(f"{method:<37} {sel:<12.2f} {fit:<12.2f} {pred:<12.2f} {total:<12.2f}")

    # Also add n_clusters timing data
    timing_data['n_clusters_timing'] = {}
    for nc in N_CLUSTERS_VALUES:
        sels = [r['metrics']['selection_time'] for r in all_results['n_clusters_sensitivity']
                if r['n_clusters'] == nc and 'metrics' in r]
        if sels:
            timing_data['n_clusters_timing'][str(nc)] = {
                'mean_selection_time': float(np.mean(sels)),
                'std_selection_time': float(np.std(sels)),
            }

    timing_data['context_size_timing'] = {}
    for cs in CONTEXT_SIZE_VALUES:
        sels = [r['metrics']['selection_time'] for r in all_results['context_size_sensitivity']
                if r['context_size'] == cs and 'metrics' in r]
        if sels:
            timing_data['context_size_timing'][str(cs)] = {
                'mean_selection_time': float(np.mean(sels)),
                'std_selection_time': float(np.std(sels)),
            }

    with open(os.path.join(RESULT_DIR, 'complexity_timing.json'), 'w') as f:
        json.dump(json_safe(timing_data), f, indent=2, ensure_ascii=False)
    print(f"\nTiming data saved to results/complexity_timing.json")

    # Final save
    with open(os.path.join(RESULT_DIR, 'sensitivity_results.json'), 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
    print(f"Sensitivity results saved to results/sensitivity_results.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
