"""Regenerate figures WITHOUT internal titles + add Figure 1 architecture diagram."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = r"d:\ResearchPaperPrepare\67_DCS_Tabular_CovariateShift"
RESULTS = os.path.join(BASE, "results")
PLOTS = os.path.join(BASE, "plots")

# ============================================================
# Figure 1: Algorithm architecture diagram
# ============================================================
def fig1_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Colors
    c_train = '#4472C4'   # blue
    c_test = '#ED7D31'    # orange
    c_proc = '#70AD47'    # green
    c_output = '#FFC000'  # yellow
    c_arrow = '#555555'

    def draw_box(x, y, w, h, text, color, fontsize=9, text_color='white'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2, color=c_arrow):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Train set
    draw_box(0.3, 5.5, 2.5, 1.0, 'Training Set\nD_train (N > 10,000)', c_train, 9)
    # Test set
    draw_box(0.3, 3.5, 2.5, 1.0, 'Test Set\nD_test (unlabeled)', c_test, 9)

    # Step 1: Density Ratio Estimation
    draw_box(3.5, 5.5, 2.5, 1.0, 'Step 1: Domain\nClassifier (Logistic)', c_proc, 8)
    # Step 2: K-Means
    draw_box(3.5, 3.5, 2.5, 1.0, 'Step 2: K-Means\nClustering (K=50)', c_proc, 8)

    draw_arrow(2.8, 6.0, 3.5, 6.0)
    draw_arrow(2.8, 4.0, 3.5, 4.0)

    # Density ratios
    draw_box(6.5, 5.5, 2.5, 1.0, 'Density Ratios\nr(x) = p_test/p_train', c_train, 8, 'black')
    # Cluster labels
    draw_box(6.5, 3.5, 2.5, 1.0, 'K Clusters\n{C_1, ..., C_K}', c_train, 8, 'black')

    draw_arrow(6.0, 6.0, 6.5, 6.0)
    draw_arrow(6.0, 4.0, 6.5, 4.0)

    # Step 3: Budget Allocation
    draw_box(6.5, 1.5, 2.5, 1.0, 'Step 3: Budget\nAllocation b_k', c_proc, 8)

    draw_arrow(7.75, 4.5, 7.75, 2.5)  # cluster to budget
    draw_arrow(7.75, 5.5, 7.75, 4.5, '#999999')  # density to budget (dashed-like)

    # Step 4: Selection
    draw_box(3.5, 1.5, 2.5, 1.0, 'Step 4: Within-Cluster\nTop-b_k Selection', c_proc, 8)
    draw_arrow(6.5, 2.0, 6.0, 2.0)

    # Output
    draw_box(0.3, 1.5, 2.5, 1.0, 'Selected Context\nS (|S| = 10,000)', c_output, 9, 'black')
    draw_arrow(3.5, 2.0, 2.8, 2.0)

    # TabPFN
    draw_box(0.3, 0.0, 2.5, 1.0, 'TabPFN v2\nIn-Context Learning', '#7030A0', 9)
    draw_arrow(1.55, 1.5, 1.55, 1.0)

    plt.tight_layout(pad=0.5)
    path = os.path.join(PLOTS, "fig1_architecture.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 2: Main results comparison (bar chart, NO INTERNAL TITLE)
# ============================================================
def fig2_results_comparison():
    with open(os.path.join(RESULTS, "statistical_test_results.json"), 'r') as f:
        data = json.load(f)

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

    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=10)
    ymin = min(means) - 1.5
    ymax = max(means) + 1.5
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS, "fig2_results_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 3: Extended dataset results (grouped bar chart, NO INTERNAL TITLE)
# ============================================================
def fig3_extended_datasets():
    with open(os.path.join(RESULTS, "extended_dataset_results.json"), 'r') as f:
        edata = json.load(f)

    # Also load mushroom results
    with open(os.path.join(RESULTS, "mushroom_dcs_results.json"), 'r') as f:
        mdata = json.load(f)

    datasets = {}

    # Process extended_dataset_results.json
    for r in edata.get("results", []):
        if r["split"] != "temporal":
            continue
        ds = r["dataset"]
        method = r["method"].replace("TabPFN-", "")
        if ds not in datasets:
            datasets[ds] = {}
        datasets[ds][method] = r["metrics"]["accuracy"] * 100

    # Process mushroom results (separate from extended results to avoid conflict)
    mushroom_acc = {}
    for r in mdata.get("results", []):
        if r["split"] != "temporal":
            continue
        method = r["method"].replace("TabPFN-", "")
        if method not in mushroom_acc:
            mushroom_acc[method] = []
        mushroom_acc[method].append(r["metrics"]["accuracy"] * 100)

    # Average mushroom results and add to datasets
    for method, vals in mushroom_acc.items():
        if "mushroom" not in datasets:
            datasets["mushroom"] = {}
        datasets["mushroom"][method] = np.mean(vals)

    ds_names = [d for d in ["adult", "mushroom", "bank", "telco"] if d in datasets]
    methods = ["Random", "KNN", "DCS-Logistic", "DCS-LightGBM"]
    colors = ["#4472C4", "#5B9BD5", "#ED7D31", "#FFC000"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ds_names))
    width = 0.2

    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [datasets[ds].get(method, 0) for ds in ds_names]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=method, color=color, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in ds_names], fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS, "fig3_extended_datasets.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 4: Sensitivity analysis (dual panel, NO INTERNAL TITLE)
# ============================================================
def fig4_sensitivity():
    with open(os.path.join(RESULTS, "sensitivity_results.json"), 'r') as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A
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
    ax1.set_xscale('log')
    ax1.set_xticks(ks)
    ax1.set_xticklabels(ks)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel B
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
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS, "fig4_sensitivity_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    print("Regenerating figures (no internal titles)...")
    fig1_architecture()
    fig2_results_comparison()
    fig3_extended_datasets()
    fig4_sensitivity()
    print("\nAll figures regenerated!")