"""Generate key figures for IJMLC submission from existing experimental data."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r"d:\ResearchPaperPrepare\67_DCS_Tabular_CovariateShift"
RESULTS = os.path.join(BASE, "results")
PLOTS = os.path.join(BASE, "plots")

# ============================================================
# Figure 2: Main results comparison (bar chart)
# ============================================================
def fig2_results_comparison():
    with open(os.path.join(RESULTS, "statistical_test_results.json"), 'r') as f:
        data = json.load(f)

    # Extract Adult/temporal accuracy for each method (5-seed mean)
    cs = data.get("context_shield_analysis", {})
    temporal = cs.get("adult_temporal_accuracy", {})

    methods = []
    means = []
    stds = []
    colors = []

    method_order = [
        ("TabPFN-Random", "#4472C4"),
        ("TabPFN-KNN", "#5B9BD5"),
        ("TabPFN-DRWS-Logistic", "#70AD47"),
        ("TabPFN-DCS-Logistic", "#ED7D31"),
        ("TabPFN-DCS-LightGBM", "#FFC000"),
    ]

    for method, color in method_order:
        comp = temporal.get("comparisons", {})
        if method == "TabPFN-Random":
            vals = temporal.get("baseline_values", [])
        elif method in comp:
            vals = comp[method].get("values", [])
        else:
            continue
        if vals:
            methods.append(method.replace("TabPFN-", ""))
            means.append(np.mean(vals) * 100)
            stds.append(np.std(vals, ddof=1) * 100)
            colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)

    # Highlight best
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Method Comparison on Adult/Temporal Split (5-seed mean ± std)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(min(means) - 1.5, max(means) + 1.5)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS, "fig2_results_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 3: Extended dataset results (grouped bar chart)
# ============================================================
def fig3_extended_datasets():
    with open(os.path.join(RESULTS, "extended_dataset_results.json"), 'r') as f:
        data = json.load(f)

    # Extract temporal accuracy for each dataset and method
    datasets = {}
    for r in data.get("results", []):
        if r["split"] != "temporal":
            continue
        ds = r["dataset"]
        method = r["method"].replace("TabPFN-", "")
        if ds not in datasets:
            datasets[ds] = {}
        datasets[ds][method] = r["metrics"]["accuracy"] * 100

    ds_names = list(datasets.keys())
    methods = ["Random", "KNN", "DCS-Logistic", "DCS-LightGBM"]
    colors = ["#4472C4", "#5B9BD5", "#ED7D31", "#FFC000"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ds_names))
    width = 0.2

    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [datasets[ds].get(method, 0) for ds in ds_names]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=color, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('DCS Performance Across Datasets (Temporal Split)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in ds_names], fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS, "fig3_extended_datasets.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 4: Sensitivity analysis (dual panel)
# ============================================================
def fig4_sensitivity():
    with open(os.path.join(RESULTS, "sensitivity_results.json"), 'r') as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: K (number of clusters) sensitivity
    k_results = {}
    for r in data.get("n_clusters_sensitivity", []):
        k = r["n_clusters"]
        if k not in k_results:
            k_results[k] = []
        k_results[k].append(r["metrics"]["accuracy"] * 100)

    ks = sorted(k_results.keys())
    k_means = [np.mean(k_results[k]) for k in ks]
    k_stds = [np.std(k_results[k], ddof=1) for k in ks]

    ax1.errorbar(ks, k_means, yerr=k_stds, marker='o', capsize=4, linewidth=2, color='#ED7D31')
    ax1.axhline(y=k_means[ks.index(50)] if 50 in ks else k_means[-1],
                color='gray', linestyle='--', alpha=0.5, label='Default K=50')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('(a) Sensitivity to K', fontsize=11)
    ax1.set_xscale('log')
    ax1.set_xticks(ks)
    ax1.set_xticklabels(ks)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel B: Context size sensitivity
    c_results = {}
    for r in data.get("context_size_sensitivity", []):
        c = r["context_size"]
        if c not in c_results:
            c_results[c] = []
        c_results[c].append(r["metrics"]["accuracy"] * 100)

    cs = sorted(c_results.keys())
    c_means = [np.mean(c_results[c]) for c in cs]
    c_stds = [np.std(c_results[c], ddof=1) if len(c_results[c]) > 1 else 0 for c in cs]

    ax2.errorbar(cs, c_means, yerr=c_stds, marker='s', capsize=4, linewidth=2, color='#4472C4')
    ax2.axhline(y=c_means[cs.index(10000)] if 10000 in cs else c_means[-1],
                color='gray', linestyle='--', alpha=0.5, label='Default = 10000')
    ax2.set_xlabel('Context Size', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('(b) Sensitivity to Context Size', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS, "fig4_sensitivity_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    print("Generating figures for IJMLC submission...")
    fig2_results_comparison()
    fig3_extended_datasets()
    fig4_sensitivity()
    print("\nAll figures generated successfully!")
