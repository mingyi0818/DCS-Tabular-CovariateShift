"""Attention weight visualization and feature-space proximity analysis for DCS.

This script performs two complementary analyses:

1. Feature-space proximity analysis (always works):
   - Compute RBF kernel similarity between test samples and context samples
   - Show DCS-selected contexts are closer to test distribution than Random
   - PCA/t-SNE visualization of context distributions

2. Attention weight extraction from local TabPFN model (best-effort):
   - Register forward hooks on transformer attention layers
   - Extract attention weights from test samples to context samples
   - Compare attention patterns: Random context vs DCS context
   - Visualize attention entropy and distribution

Data sources:
  - data/raw/adult/adult.csv
  - code/context_shield_methods.py (DCS selection)
  - code/splits.py (data splitting)

Results saved to:
  - results/attention_analysis_results.json
  - plots/fig_feature_space_pca.png
  - plots/fig_similarity_distribution.png
  - plots/fig_attention_heatmap.png (if attention extraction succeeds)
  - plots/fig_attention_entropy.png (if attention extraction succeeds)
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 10

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR, PLOT_DIR, DRIFT_TABPFN_PATH
from splits import prepare_split
from context_shield_methods import (
    dcs_selection, random_context_selection,
    estimate_density_ratio, json_safe
)

SEED = 42
CONTEXT_SIZE = 10000
N_TEST_SAMPLE = 500  # Subsample test for visualization clarity
N_CONTEXT_VISUALIZE = 2000  # Subsample context for PCA/t-SNE speed


def compute_similarity_stats(X_context, X_test, gamma=None):
    """Compute average RBF kernel similarity between context and test."""
    if gamma is None:
        # Use median heuristic for gamma
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X_test[:200], X_context[:200])
        gamma = 1.0 / (2 * np.median(dists[dists > 0])**2)

    sim = rbf_kernel(X_test[:N_TEST_SAMPLE], X_context[:N_CONTEXT_VISUALIZE], gamma=gamma)
    return {
        'mean_similarity': float(sim.mean()),
        'std_similarity': float(sim.std()),
        'median_similarity': float(np.median(sim)),
        'p75_similarity': float(np.percentile(sim, 75)),
        'p95_similarity': float(np.percentile(sim, 95)),
        'gamma': float(gamma),
    }


def run_feature_space_analysis():
    """Analyze feature-space proximity of DCS vs Random contexts."""
    print("\n[1] Feature-space proximity analysis...")

    split_data = prepare_split('adult', 'temporal', seed=SEED)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Select contexts
    idx_random = random_context_selection(X_train, CONTEXT_SIZE, seed=SEED)
    idx_dcs = dcs_selection(X_train, X_test, CONTEXT_SIZE, n_clusters=50,
                            method='logistic', seed=SEED)

    # Compute similarity stats
    print("  Computing RBF kernel similarities...")
    stats_random = compute_similarity_stats(X_train[idx_random], X_test)
    stats_dcs = compute_similarity_stats(X_train[idx_dcs], X_test)

    print(f"  Random context: mean_sim={stats_random['mean_similarity']:.6f}")
    print(f"  DCS context:    mean_sim={stats_dcs['mean_similarity']:.6f}")
    print(f"  Improvement:    {(stats_dcs['mean_similarity'] - stats_random['mean_similarity']) / stats_random['mean_similarity'] * 100:.1f}%")

    # Also compute density ratio stats
    print("  Computing density ratio distributions...")
    dr_all = estimate_density_ratio(X_train, X_test, method='logistic', seed=SEED)
    dr_random = dr_all[idx_random]
    dr_dcs = dr_all[idx_dcs]

    density_ratio_stats = {
        'random': {
            'mean': float(dr_random.mean()),
            'std': float(dr_random.std()),
            'median': float(np.median(dr_random)),
        },
        'dcs': {
            'mean': float(dr_dcs.mean()),
            'std': float(dr_dcs.std()),
            'median': float(np.median(dr_dcs)),
        },
        'all_train': {
            'mean': float(dr_all.mean()),
            'std': float(dr_all.std()),
            'median': float(np.median(dr_all)),
        },
    }

    # PCA visualization
    print("  Generating PCA visualization...")
    n_pca = min(N_CONTEXT_VISUALIZE, len(idx_random), len(idx_dcs))
    n_test_pca = min(N_TEST_SAMPLE, len(X_test))

    rng = np.random.RandomState(SEED)
    test_subset_idx = rng.choice(len(X_test), n_test_pca, replace=False)

    X_combined = np.vstack([
        X_train[idx_random[:n_pca]],
        X_train[idx_dcs[:n_pca]],
        X_test[test_subset_idx],
    ])
    labels_combined = (
        ['Random Context'] * n_pca +
        ['DCS Context'] * n_pca +
        ['Test Set'] * n_test_pca
    )

    scaler = StandardScaler()
    X_combined_s = scaler.fit_transform(X_combined)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined_s)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Random Context': '#2196F3', 'DCS Context': '#F44336', 'Test Set': '#4CAF50'}
    markers = {'Random Context': 'o', 'DCS Context': 's', 'Test Set': '^'}
    sizes = {'Random Context': 15, 'DCS Context': 15, 'Test Set': 30}
    alphas = {'Random Context': 0.3, 'DCS Context': 0.5, 'Test Set': 0.9}

    for label in ['Random Context', 'DCS Context', 'Test Set']:
        mask = np.array([l == label for l in labels_combined])
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[label], marker=markers[label],
                   s=sizes[label], alpha=alphas[label], label=label,
                   edgecolors='none')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Feature Space: DCS vs Random Context vs Test Distribution\n(Adult/Temporal Split, PCA)')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    pca_path = os.path.join(PLOT_DIR, 'fig_feature_space_pca.png')
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pca_path}")

    # Similarity distribution plot
    print("  Generating similarity distribution plot...")
    gamma_val = stats_random['gamma']
    sim_random = rbf_kernel(X_test[test_subset_idx],
                            X_train[idx_random[:n_pca]], gamma=gamma_val)
    sim_dcs = rbf_kernel(X_test[test_subset_idx],
                         X_train[idx_dcs[:n_pca]], gamma=gamma_val)

    # Average similarity per test sample
    avg_sim_random = sim_random.mean(axis=1)
    avg_sim_dcs = sim_dcs.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histogram of per-test-sample average similarity
    ax = axes[0]
    bins = np.linspace(0, max(avg_sim_random.max(), avg_sim_dcs.max()) * 1.1, 50)
    ax.hist(avg_sim_random, bins=bins, alpha=0.6, color='#2196F3', label='Random Context', density=True)
    ax.hist(avg_sim_dcs, bins=bins, alpha=0.6, color='#F44336', label='DCS Context', density=True)
    ax.axvline(avg_sim_random.mean(), color='#1565C0', linestyle='--', linewidth=2,
               label=f'Random mean={avg_sim_random.mean():.4f}')
    ax.axvline(avg_sim_dcs.mean(), color='#C62828', linestyle='--', linewidth=2,
               label=f'DCS mean={avg_sim_dcs.mean():.4f}')
    ax.set_xlabel('Average RBF Similarity to Context')
    ax.set_ylabel('Density')
    ax.set_title('Per-Test-Sample Similarity Distribution')
    ax.legend(fontsize=8)

    # Right: Density ratio distribution
    ax = axes[1]
    bins_dr = np.linspace(0, max(dr_random.max(), dr_dcs.max()) * 0.9, 60)
    ax.hist(dr_random, bins=bins_dr, alpha=0.6, color='#2196F3', label='Random Context', density=True)
    ax.hist(dr_dcs, bins=bins_dr, alpha=0.6, color='#F44336', label='DCS Context', density=True)
    ax.axvline(dr_random.mean(), color='#1565C0', linestyle='--', linewidth=2,
               label=f'Random mean={dr_random.mean():.2f}')
    ax.axvline(dr_dcs.mean(), color='#C62828', linestyle='--', linewidth=2,
               label=f'DCS mean={dr_dcs.mean():.2f}')
    ax.set_xlabel('Estimated Density Ratio p_test(x)/p_train(x)')
    ax.set_ylabel('Density')
    ax.set_title('Density Ratio Distribution of Selected Contexts')
    ax.legend(fontsize=8)

    plt.suptitle('DCS-Logistic vs Random: Distribution Matching Quality (Adult/Temporal)', y=1.02)
    plt.tight_layout()
    sim_path = os.path.join(PLOT_DIR, 'fig_similarity_distribution.png')
    plt.savefig(sim_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {sim_path}")

    return {
        'similarity_stats': {
            'random': stats_random,
            'dcs': stats_dcs,
            'improvement_pct': float((stats_dcs['mean_similarity'] - stats_random['mean_similarity']) /
                                     stats_random['mean_similarity'] * 100),
        },
        'density_ratio_stats': density_ratio_stats,
        'pca_variance_explained': [float(v) for v in pca.explained_variance_ratio_],
        'plots': {
            'pca': pca_path,
            'similarity': sim_path,
        },
    }


def compute_pseudo_attention(X_context, X_test, gamma=None):
    """Compute pseudo-attention weights using RBF kernel.

    TabPFN's transformer attention on tabular data approximates an RBF kernel
    similarity between test queries and context keys. We use this as a
    theoretically motivated proxy for actual attention weights when the
    model internals are not directly accessible.

    Args:
        X_context: context features (n_context, d)
        X_test: test features (n_test, d)
        gamma: RBF kernel parameter (auto-estimated if None)

    Returns:
        attention: (n_test, n_context) matrix, row-normalized to sum=1
        gamma: the gamma value used
    """
    if gamma is None:
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X_test[:200], X_context[:200])
        med_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * med_dist ** 2)

    # RBF kernel as attention scores
    raw_attn = rbf_kernel(X_test, X_context, gamma=gamma)
    # Row-normalize to get attention distribution
    attention = raw_attn / (raw_attn.sum(axis=1, keepdims=True) + 1e-12)
    return attention, float(gamma)


def analyze_pseudo_attention(X_train, y_train, X_test, y_test, seed=42):
    """Analyze pseudo-attention patterns for Random vs DCS contexts.

    This serves as a theoretically motivated proxy for TabPFN's actual
    transformer attention when model internals are not directly accessible.
    """
    print("\n[2] Pseudo-attention analysis (RBF kernel proxy)...")

    n_ctx = 500
    n_test = 200

    rng = np.random.RandomState(seed)
    test_idx = rng.choice(len(X_test), n_test, replace=False)
    X_test_sub = X_test[test_idx]

    # Select contexts
    idx_random = random_context_selection(X_train, n_ctx, seed=seed)
    idx_dcs = dcs_selection(X_train, X_test, n_ctx, n_clusters=30,
                            method='logistic', seed=seed)

    X_ctx_random = X_train[idx_random]
    X_ctx_dcs = X_train[idx_dcs]

    # Compute pseudo-attention
    attn_random, gamma = compute_pseudo_attention(X_ctx_random, X_test_sub)
    attn_dcs, _ = compute_pseudo_attention(X_ctx_dcs, X_test_sub)

    print(f"  Gamma (RBF): {gamma:.6f}")
    print(f"  Attention matrix: Random={attn_random.shape}, DCS={attn_dcs.shape}")

    # Attention entropy per test sample
    def entropy(attn):
        h = -np.sum(attn * np.log(attn + 1e-12), axis=1)
        h_norm = h / np.log(attn.shape[1])
        return h, h_norm

    ent_random, ent_norm_random = entropy(attn_random)
    ent_dcs, ent_norm_dcs = entropy(attn_dcs)

    # Attention concentration (max and top-5)
    def concentration(attn):
        sorted_attn = np.sort(attn, axis=1)
        max_attn = sorted_attn[:, -1]
        top5 = sorted_attn[:, -5:].sum(axis=1)
        top10 = sorted_attn[:, -10:].sum(axis=1)
        return max_attn, top5, top10

    max_r, top5_r, top10_r = concentration(attn_random)
    max_d, top5_d, top10_d = concentration(attn_dcs)

    print(f"\n  Attention Entropy (normalized, 0=concentrated, 1=uniform):")
    print(f"    Random: {ent_norm_random.mean():.4f} ± {ent_norm_random.std():.4f}")
    print(f"    DCS:    {ent_norm_dcs.mean():.4f} ± {ent_norm_dcs.std():.4f}")
    print(f"    Δ:      {ent_norm_dcs.mean() - ent_norm_random.mean():+.4f}")

    print(f"\n  Attention Concentration (max weight):")
    print(f"    Random: {max_r.mean():.4f}")
    print(f"    DCS:    {max_d.mean():.4f}")

    print(f"\n  Top-5 Attention Mass:")
    print(f"    Random: {top5_r.mean():.4f}")
    print(f"    DCS:    {top5_d.mean():.4f}")

    # Generate heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_show = 20
    n_ctx_show = 50

    for ax, attn, title in [
        (axes[0], attn_random, 'Random Context'),
        (axes[1], attn_dcs, 'DCS-Logistic Context')
    ]:
        attn_vis = attn[:n_show, :n_ctx_show]
        im = ax.imshow(attn_vis, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Context Sample Index')
        ax.set_ylabel('Test Sample Index')
        ax.set_title(f'Pseudo-Attention: {title}')
        plt.colorbar(im, ax=ax, label='Attention Weight')

    plt.suptitle('TabPFN Pseudo-Attention Pattern (RBF Kernel Proxy)\nRandom vs DCS-Logistic Context (Adult/Temporal)', y=1.02)
    plt.tight_layout()
    heatmap_path = os.path.join(PLOT_DIR, 'fig_attention_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {heatmap_path}")

    # Entropy distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Entropy histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    ax.hist(ent_norm_random, bins=bins, alpha=0.6, color='#2196F3',
            label='Random Context', density=True)
    ax.hist(ent_norm_dcs, bins=bins, alpha=0.6, color='#F44336',
            label='DCS Context', density=True)
    ax.axvline(ent_norm_random.mean(), color='#1565C0', linestyle='--', linewidth=2,
               label=f'Random mean={ent_norm_random.mean():.4f}')
    ax.axvline(ent_norm_dcs.mean(), color='#C62828', linestyle='--', linewidth=2,
               label=f'DCS mean={ent_norm_dcs.mean():.4f}')
    ax.set_xlabel('Normalized Attention Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Attention Entropy Distribution')
    ax.legend(fontsize=8)

    # Right: Max attention histogram
    ax = axes[1]
    bins_m = np.linspace(0, max(max_r.max(), max_d.max()) * 1.1, 30)
    ax.hist(max_r, bins=bins_m, alpha=0.6, color='#2196F3',
            label='Random Context', density=True)
    ax.hist(max_d, bins=bins_m, alpha=0.6, color='#F44336',
            label='DCS Context', density=True)
    ax.axvline(max_r.mean(), color='#1565C0', linestyle='--', linewidth=2,
               label=f'Random mean={max_r.mean():.4f}')
    ax.axvline(max_d.mean(), color='#C62828', linestyle='--', linewidth=2,
               label=f'DCS mean={max_d.mean():.4f}')
    ax.set_xlabel('Max Attention Weight per Test Sample')
    ax.set_ylabel('Density')
    ax.set_title('Attention Concentration Distribution')
    ax.legend(fontsize=8)

    plt.suptitle('Pseudo-Attention Analysis: Random vs DCS-Logistic (Adult/Temporal)', y=1.02)
    plt.tight_layout()
    entropy_path = os.path.join(PLOT_DIR, 'fig_attention_entropy.png')
    plt.savefig(entropy_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {entropy_path}")

    return {
        'method': 'pseudo_attention_rbf_kernel',
        'gamma': gamma,
        'n_context': n_ctx,
        'n_test': n_test,
        'entropy': {
            'random': {
                'mean_normalized': float(ent_norm_random.mean()),
                'std_normalized': float(ent_norm_random.std()),
                'mean_raw': float(ent_random.mean()),
            },
            'dcs': {
                'mean_normalized': float(ent_norm_dcs.mean()),
                'std_normalized': float(ent_norm_dcs.std()),
                'mean_raw': float(ent_dcs.mean()),
            },
            'delta_normalized': float(ent_norm_dcs.mean() - ent_norm_random.mean()),
        },
        'concentration': {
            'random': {
                'mean_max': float(max_r.mean()),
                'mean_top5': float(top5_r.mean()),
                'mean_top10': float(top10_r.mean()),
            },
            'dcs': {
                'mean_max': float(max_d.mean()),
                'mean_top5': float(top5_d.mean()),
                'mean_top10': float(top10_d.mean()),
            },
        },
        'plots': {
            'heatmap': heatmap_path,
            'entropy': entropy_path,
        },
    }


def try_extract_attention_weights():
    """Attempt to extract attention weights from local TabPFN model.

    Uses PyTorch forward hooks on the transformer attention layers.
    Falls back to pseudo-attention (RBF kernel proxy) if model internals
    are not directly accessible.
    """
    print("\n[2] Attempting attention weight extraction from local TabPFN model...")

    try:
        import torch
        import torch.nn as nn

        # Add Drift-Resilient TabPFN to path
        sys.path.insert(0, DRIFT_TABPFN_PATH)

        from importlib import resources
        import tabpfn
        from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig

        libpath = str(resources.files(tabpfn))

        # Load base model (lightweight, debug mode for speed)
        model_path_config = TabPFNModelPathsConfig(
            paths=[f"{libpath}/model_cache/tabpfn_base_model_2.cpkt"],
            task_type="multiclass"
        )
        model = get_best_tabpfn(
            task_type="multiclass",
            model_type="base",
            paths_config=model_path_config,
            debug=True,
            device="cuda"
        )
        model.show_progress = False
        model.seed = SEED

        # Inspect model architecture to find attention layers
        print("  Inspecting model architecture for attention layers...")

        # Try to access the underlying PyTorch model
        attention_layers = []

        def find_attention_modules(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                # Check for common attention module names
                if any(kw in type(child).__name__.lower() for kw in ['attention', 'multihead', 'attn']):
                    attention_layers.append((full_name, child))
                # Also check for nn.MultiheadAttention
                if isinstance(child, nn.MultiheadAttention):
                    attention_layers.append((full_name, child))
                # Recurse
                find_attention_modules(child, full_name)

        # The TabPFN model might store the network inside
        if hasattr(model, 'model_'):
            inner = model.model_
        elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            inner = model.estimators_[0]
        elif hasattr(model, 'model'):
            inner = model.model
        else:
            inner = model

        print(f"  Model type: {type(inner).__name__}")
        find_attention_modules(inner)

        if not attention_layers:
            print("  No standard attention layers found. Trying module enumeration...")
            for name, module in inner.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    attention_layers.append((name, module))

        print(f"  Found {len(attention_layers)} attention-related modules")
        for name, mod in attention_layers[:5]:
            print(f"    - {name}: {type(mod).__name__}")

        if not attention_layers:
            print("  [WARNING] No attention layers found. Skipping attention extraction.")
            return None

        # Register hooks to capture attention weights
        captured_weights = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                # For nn.MultiheadAttention, output is (attn_output, attn_weights)
                # if need_weights=True
                if isinstance(output, tuple) and len(output) == 2:
                    captured_weights[name] = output[1]
                # Some custom implementations return dict
                elif isinstance(output, dict) and 'attn_weights' in output:
                    captured_weights[name] = output['attn_weights']
            return hook_fn

        hooks = []
        for name, mod in attention_layers:
            h = mod.register_forward_hook(make_hook(name))
            hooks.append(h)

        # Prepare data: small subset for attention extraction
        split_data = prepare_split('adult', 'temporal', seed=SEED)
        X_train = split_data['X_train']
        y_train = split_data['y_train']
        X_test = split_data['X_test']
        y_test = split_data['y_test']

        # Use small context for attention visualization (TabPFN can handle up to 10K,
        # but for attention extraction we use smaller for memory)
        n_ctx = 500
        n_test = 100

        rng = np.random.RandomState(SEED)
        test_idx = rng.choice(len(X_test), n_test, replace=False)

        # Random context
        idx_random = random_context_selection(X_train, n_ctx, seed=SEED)
        X_ctx_random = X_train[idx_random].astype(np.float32)
        y_ctx_random = y_train[idx_random]
        X_test_small = X_test[test_idx].astype(np.float32)
        y_test_small = y_test[test_idx]

        print(f"  Running inference with Random context (ctx={n_ctx}, test={n_test})...")
        captured_weights.clear()
        try:
            model.fit(X_ctx_random, y_ctx_random)
            preds_random = model.predict(X_test_small)
            print(f"    Prediction done. Accuracy: {(preds_random == y_test_small).mean():.4f}")
        except Exception as e:
            print(f"    Prediction failed: {e}")
            for h in hooks:
                h.remove()
            return None

        weights_random = {}
        for name, w in captured_weights.items():
            if w is not None:
                if isinstance(w, torch.Tensor):
                    weights_random[name] = w.cpu().numpy()
                else:
                    weights_random[name] = np.array(w)
                print(f"    Captured {name}: shape={weights_random[name].shape}")

        # DCS context
        idx_dcs = dcs_selection(X_train, X_test, n_ctx, n_clusters=30,
                                method='logistic', seed=SEED)
        X_ctx_dcs = X_train[idx_dcs].astype(np.float32)
        y_ctx_dcs = y_train[idx_dcs]

        print(f"  Running inference with DCS context (ctx={n_ctx}, test={n_test})...")
        captured_weights.clear()
        try:
            model.fit(X_ctx_dcs, y_ctx_dcs)
            preds_dcs = model.predict(X_test_small)
            print(f"    Prediction done. Accuracy: {(preds_dcs == y_test_small).mean():.4f}")
        except Exception as e:
            print(f"    Prediction failed: {e}")
            for h in hooks:
                h.remove()
            return None

        weights_dcs = {}
        for name, w in captured_weights.items():
            if w is not None:
                if isinstance(w, torch.Tensor):
                    weights_dcs[name] = w.cpu().numpy()
                else:
                    weights_dcs[name] = np.array(w)

        # Remove hooks
        for h in hooks:
            h.remove()

        if not weights_random or not weights_dcs:
            print("  [WARNING] No attention weights were captured. Model may not expose them.")
            return None

        # Analyze attention weights
        # Use the first captured attention layer for visualization
        layer_name = list(weights_random.keys())[0]
        attn_random = weights_random[layer_name]
        attn_dcs = weights_dcs[layer_name]

        print(f"\n  Analyzing attention from layer: {layer_name}")
        print(f"  Random attention shape: {attn_random.shape}")
        print(f"  DCS attention shape: {attn_dcs.shape}")

        # Compute attention entropy per test sample
        def attention_entropy(attn_weights):
            """Compute entropy of attention distribution for each test sample."""
            # attn_weights shape: (n_heads, n_test, n_context) or (n_test, n_context)
            if attn_weights.ndim == 3:
                # Average over heads
                attn_avg = attn_weights.mean(axis=0)
            else:
                attn_avg = attn_weights

            # Normalize to probability distribution
            attn_norm = attn_avg / (attn_avg.sum(axis=-1, keepdims=True) + 1e-12)

            # Entropy: H = -sum(p * log(p))
            entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-12), axis=-1)
            # Normalize by max entropy
            max_entropy = np.log(attn_norm.shape[-1])
            normalized_entropy = entropy / max_entropy

            return {
                'mean_entropy': float(entropy.mean()),
                'std_entropy': float(entropy.std()),
                'mean_normalized_entropy': float(normalized_entropy.mean()),
                'std_normalized_entropy': float(normalized_entropy.std()),
                'per_sample_entropy': entropy.tolist(),
                'per_sample_normalized': normalized_entropy.tolist(),
            }

        entropy_random = attention_entropy(attn_random)
        entropy_dcs = attention_entropy(attn_dcs)

        # Compute attention concentration (max attention weight per test sample)
        def attention_concentration(attn_weights):
            if attn_weights.ndim == 3:
                attn_avg = attn_weights.mean(axis=0)
            else:
                attn_avg = attn_weights
            attn_norm = attn_avg / (attn_avg.sum(axis=-1, keepdims=True) + 1e-12)
            max_attn = attn_norm.max(axis=-1)
            top5_attn = np.sort(attn_norm, axis=-1)[:, -5:].sum(axis=-1)
            return {
                'mean_max_attention': float(max_attn.mean()),
                'mean_top5_attention': float(top5_attn.mean()),
            }

        conc_random = attention_concentration(attn_random)
        conc_dcs = attention_concentration(attn_dcs)

        print(f"\n  Attention Entropy (normalized):")
        print(f"    Random: {entropy_random['mean_normalized_entropy']:.4f} ± {entropy_random['std_normalized_entropy']:.4f}")
        print(f"    DCS:    {entropy_dcs['mean_normalized_entropy']:.4f} ± {entropy_dcs['std_normalized_entropy']:.4f}")
        print(f"\n  Attention Concentration (max weight):")
        print(f"    Random: {conc_random['mean_max_attention']:.4f}")
        print(f"    DCS:    {conc_dcs['mean_max_attention']:.4f}")

        # Generate attention heatmap (first 20 test samples, first 50 context samples)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        n_show_test = min(20, attn_random.shape[-2] if attn_random.ndim == 3 else attn_random.shape[0])
        n_show_ctx = min(50, attn_random.shape[-1])

        for ax, attn, title in [
            (axes[0], attn_random, 'Random Context'),
            (axes[1], attn_dcs, 'DCS-Logistic Context')
        ]:
            if attn.ndim == 3:
                attn_vis = attn.mean(axis=0)[:n_show_test, :n_show_ctx]
            else:
                attn_vis = attn[:n_show_test, :n_show_ctx]
            # Normalize per row (per test sample)
            attn_vis = attn_vis / (attn_vis.sum(axis=-1, keepdims=True) + 1e-12)
            im = ax.imshow(attn_vis, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_xlabel('Context Sample Index')
            ax.set_ylabel('Test Sample Index')
            ax.set_title(f'Attention Weights: {title}')
            plt.colorbar(im, ax=ax, label='Attention Weight')

        plt.suptitle(f'TabPFN Attention Pattern: Random vs DCS Context (Layer: {layer_name[:30]})', y=1.02)
        plt.tight_layout()
        heatmap_path = os.path.join(PLOT_DIR, 'fig_attention_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {heatmap_path}")

        # Attention entropy distribution plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ent_random = np.array(entropy_random['per_sample_normalized'])
        ent_dcs = np.array(entropy_dcs['per_sample_normalized'])

        bins = np.linspace(0, 1, 30)
        ax.hist(ent_random, bins=bins, alpha=0.6, color='#2196F3', label='Random Context', density=True)
        ax.hist(ent_dcs, bins=bins, alpha=0.6, color='#F44336', label='DCS Context', density=True)
        ax.axvline(ent_random.mean(), color='#1565C0', linestyle='--', linewidth=2,
                   label=f'Random mean={ent_random.mean():.4f}')
        ax.axvline(ent_dcs.mean(), color='#C62828', linestyle='--', linewidth=2,
                   label=f'DCS mean={ent_dcs.mean():.4f}')
        ax.set_xlabel('Normalized Attention Entropy (0=concentrated, 1=uniform)')
        ax.set_ylabel('Density')
        ax.set_title('Attention Entropy Distribution: Random vs DCS Context\n(Lower entropy = more focused attention)')
        ax.legend(fontsize=9)
        plt.tight_layout()
        entropy_path = os.path.join(PLOT_DIR, 'fig_attention_entropy.png')
        plt.savefig(entropy_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {entropy_path}")

        return {
            'layer_name': layer_name,
            'attention_shape': list(attn_random.shape),
            'entropy': {
                'random': entropy_random,
                'dcs': entropy_dcs,
            },
            'concentration': {
                'random': conc_random,
                'dcs': conc_dcs,
            },
            'prediction_accuracy': {
                'random': float((preds_random == y_test_small).mean()),
                'dcs': float((preds_dcs == y_test_small).mean()),
            },
            'plots': {
                'heatmap': heatmap_path,
                'entropy': entropy_path,
            },
        }

    except Exception as e:
        import traceback
        print(f"  [WARNING] Attention extraction failed: {e}")
        traceback.print_exc()
        return None


def main():
    print("=" * 80)
    print("Attention Weight Visualization & Feature-Space Proximity Analysis")
    print("=" * 80)

    output = {
        'experiment': 'attention_analysis',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'dataset': 'adult',
            'split': 'temporal',
            'seed': SEED,
            'context_size': CONTEXT_SIZE,
        },
    }

    # 1. Feature-space analysis (always works)
    feature_analysis = run_feature_space_analysis()
    output['feature_space_analysis'] = feature_analysis

    # 2. Attention weight extraction (best-effort from model internals)
    attention_analysis = try_extract_attention_weights()
    output['attention_extraction'] = attention_analysis

    # 2b. If model attention not available, use pseudo-attention (RBF kernel proxy)
    if attention_analysis is None:
        print("\n  [FALLBACK] Using pseudo-attention analysis (RBF kernel proxy)...")
        split_data = prepare_split('adult', 'temporal', seed=SEED)
        pseudo_analysis = analyze_pseudo_attention(
            split_data['X_train'], split_data['y_train'],
            split_data['X_test'], split_data['y_test'],
            seed=SEED
        )
        output['pseudo_attention_analysis'] = pseudo_analysis
        attention_analysis = pseudo_analysis  # For summary printing

    # Save
    output_path = os.path.join(RESULT_DIR, 'attention_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_safe(output), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFeature-Space Analysis:")
    print(f"  RBF Similarity improvement: {feature_analysis['similarity_stats']['improvement_pct']:.1f}%")
    print(f"  Random context density ratio mean: {feature_analysis['density_ratio_stats']['random']['mean']:.2f}")
    print(f"  DCS context density ratio mean:    {feature_analysis['density_ratio_stats']['dcs']['mean']:.2f}")

    if attention_analysis:
        method = attention_analysis.get('method', 'model_internals')
        if method == 'pseudo_attention_rbf_kernel':
            print(f"\nPseudo-Attention Analysis (RBF kernel proxy):")
            print(f"  Entropy (Random): {attention_analysis['entropy']['random']['mean_normalized']:.4f}")
            print(f"  Entropy (DCS):    {attention_analysis['entropy']['dcs']['mean_normalized']:.4f}")
            print(f"  Max Attn (Random): {attention_analysis['concentration']['random']['mean_max']:.4f}")
            print(f"  Max Attn (DCS):    {attention_analysis['concentration']['dcs']['mean_max']:.4f}")
        else:
            print(f"\nAttention Extraction:")
            print(f"  Layer: {attention_analysis.get('layer_name', 'unknown')}")
            print(f"  Entropy (Random): {attention_analysis['entropy']['random']['mean_normalized_entropy']:.4f}")
            print(f"  Entropy (DCS):    {attention_analysis['entropy']['dcs']['mean_normalized_entropy']:.4f}")
            print(f"  Max Attn (Random): {attention_analysis['concentration']['random']['mean_max_attention']:.4f}")
            print(f"  Max Attn (DCS):    {attention_analysis['concentration']['dcs']['mean_max_attention']:.4f}")

    print("=" * 80)


if __name__ == '__main__':
    main()
