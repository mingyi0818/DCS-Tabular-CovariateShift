"""Verify all key numbers in paper_draft.md against source JSON files."""
import json
import numpy as np
import os

RESULT_DIR = r'd:\ResearchPaperPrepare\67_DCS_Tabular_CovariateShift\results'
errors = []
checks = 0

def check(name, expected, actual, tolerance=0.001):
    global checks, errors
    checks += 1
    if abs(expected - actual) > tolerance:
        errors.append(f"MISMATCH: {name}: paper={expected}, json={actual}")
        print(f"  FAIL: {name}: paper={expected}, json={actual}")
    else:
        print(f"  OK: {name}: {expected}")

# 1. context_shield_results.json - 5-seed means
print("=== context_shield_results.json (5-seed) ===")
cs = json.load(open(os.path.join(RESULT_DIR, 'context_shield_results.json')))

# Extract 5-seed means for key methods
def get_mean(results, dataset, split, method, metric='accuracy'):
    vals = [r['metrics'][metric] for r in results
            if r['dataset']==dataset and r['split']==split and r['method']==method and r.get('metrics')]
    return np.mean(vals), np.std(vals, ddof=1) if len(vals)>1 else 0, len(vals)

for method in ['XGBoost', 'TabPFN-Random', 'TabPFN-KNN', 'TabPFN-DCS-Logistic',
                'TabPFN-DRWS-Logistic', 'TabPFN-DCS-LightGBM', 'TabPFN-ContextShield-Logistic']:
    for split in ['iid', 'temporal']:
        mean, std, n = get_mean(cs['results'], 'adult', split, method)
        print(f"  {method} {split}: acc={mean:.4f}+/-{std:.4f} (n={n})")

# Verify key paper numbers
mean_r_iid, _, _ = get_mean(cs['results'], 'adult', 'iid', 'TabPFN-Random')
mean_r_temp, _, _ = get_mean(cs['results'], 'adult', 'temporal', 'TabPFN-Random')
mean_dcs_temp, _, _ = get_mean(cs['results'], 'adult', 'temporal', 'TabPFN-DCS-Logistic')
mean_dcs_iid, _, _ = get_mean(cs['results'], 'adult', 'iid', 'TabPFN-DCS-Logistic')
mean_xgb_temp, _, _ = get_mean(cs['results'], 'adult', 'temporal', 'XGBoost')
mean_xgb_iid, _, _ = get_mean(cs['results'], 'adult', 'iid', 'XGBoost')
mean_knn_temp, _, _ = get_mean(cs['results'], 'adult', 'temporal', 'TabPFN-KNN')
mean_drws_temp, _, _ = get_mean(cs['results'], 'adult', 'temporal', 'TabPFN-DRWS-Logistic')
mean_dcs_lgb_temp, _, _ = get_mean(cs['results'], 'adult', 'temporal', 'TabPFN-DCS-LightGBM')
mean_cs_log_temp, _, _ = get_mean(cs['results'], 'adult', 'temporal', 'TabPFN-ContextShield-Logistic')

print("\n=== Verification ===")
check("Random IID acc", 0.8592, mean_r_iid)
check("Random Temporal acc", 0.8075, mean_r_temp)
check("DCS-Logistic Temporal acc", 0.8196, mean_dcs_temp)
check("DCS-Logistic IID acc", 0.8556, mean_dcs_iid)
check("XGBoost Temporal acc", 0.8327, mean_xgb_temp)
check("XGBoost IID acc", 0.8742, mean_xgb_iid)
check("KNN Temporal acc", 0.8115, mean_knn_temp)
check("DRWS-Logistic Temporal acc", 0.8193, mean_drws_temp)
check("DCS-LightGBM Temporal acc", 0.8127, mean_dcs_lgb_temp)
check("ContextShield-Logistic Temporal acc", 0.8174, mean_cs_log_temp)
check("DCS delta (temporal)", 0.0121, mean_dcs_temp - mean_r_temp)

# 2. orthogonality_exp_results.json
print("\n=== orthogonality_exp_results.json (3-seed) ===")
orth = json.load(open(os.path.join(RESULT_DIR, 'orthogonality_exp_results.json')))
s = orth['summary']
check("base+Random", 0.7694, s['TabPFN-base-Random']['accuracy_mean'])
check("base+DCS", 0.8198, s['TabPFN-base-DCS-Logistic']['accuracy_mean'])
check("dist+Random", 0.7887, s['TabPFN-dist-Random']['accuracy_mean'])
check("dist+DCS", 0.8137, s['TabPFN-dist-DCS-Logistic']['accuracy_mean'])

# 3. statistical_test_results.json
print("\n=== statistical_test_results.json ===")
stats = json.load(open(os.path.join(RESULT_DIR, 'statistical_test_results.json')))
cs_analysis = stats['context_shield_analysis']
temp_acc = cs_analysis['adult_temporal_accuracy']['comparisons']['TabPFN-DCS-Logistic']['paired_t_test']
check("DCS t-statistic", 12.58, temp_acc['t_statistic'], 0.1)
check("DCS p-value", 0.000230, temp_acc['p_value'], 0.0001)
check("DCS Cohen's d", 5.63, temp_acc['cohens_d'], 0.1)

# 4. sensitivity_results.json
print("\n=== sensitivity_results.json ===")
sens = json.load(open(os.path.join(RESULT_DIR, 'sensitivity_results.json')))
# n_clusters=30 should be best
nc30 = [r['metrics']['accuracy'] for r in sens['n_clusters_sensitivity'] if r['n_clusters']==30 and 'metrics' in r]
check("n_clusters=30 acc", 0.8202, np.mean(nc30), 0.001)
# context_size=8000 should be best
cs8000 = [r['metrics']['accuracy'] for r in sens['context_size_sensitivity'] if r['context_size']==8000 and 'metrics' in r]
check("context_size=8000 acc", 0.8206, np.mean(cs8000), 0.001)

# 5. extended_dataset_results.json - Bank
print("\n=== extended_dataset_results.json (Bank) ===")
ext = json.load(open(os.path.join(RESULT_DIR, 'extended_dataset_results.json')))
bank_random = [r['metrics']['accuracy'] for r in ext['results']
               if r['dataset']=='bank' and r['split']=='temporal' and r['method']=='TabPFN-Random' and r.get('metrics')]
bank_dcs = [r['metrics']['accuracy'] for r in ext['results']
            if r['dataset']=='bank' and r['split']=='temporal' and r['method']=='TabPFN-DCS-Logistic' and r.get('metrics')]
if bank_random:
    check("Bank Random temporal acc", 0.7753, np.mean(bank_random), 0.005)
if bank_dcs:
    check("Bank DCS temporal acc", 0.7570, np.mean(bank_dcs), 0.005)

# 6. complexity_timing.json
print("\n=== complexity_timing.json ===")
timing = json.load(open(os.path.join(RESULT_DIR, 'complexity_timing.json')))
dcs_timing = timing['methods']['TabPFN-DCS-Logistic']
check("DCS selection time", 2.43, dcs_timing['mean_selection_time'], 0.1)
check("DCS total time", 12.52, dcs_timing['mean_total_time'], 0.5)

# 7. attention_analysis_results.json
print("\n=== attention_analysis_results.json ===")
attn = json.load(open(os.path.join(RESULT_DIR, 'attention_analysis_results.json')))
sim = attn['feature_space_analysis']['similarity_stats']
check("Similarity improvement %", 5.3, sim['improvement_pct'], 0.5)

# 8. feasibility_exp1_and_3.json - for abstract numbers
print("\n=== feasibility_exp1_and_3.json (3-seed, for abstract) ===")
feas = json.load(open(os.path.join(RESULT_DIR, 'feasibility_exp1_and_3.json')))
fs = feas['summary']
check("Feasibility XGBoost IID", 0.8742, fs['adult_iid_XGBoost']['accuracy_mean'], 0.001)
check("Feasibility TabPFN-Random IID", 0.8587, fs['adult_iid_TabPFN-Random']['accuracy_mean'], 0.001)
check("Feasibility XGBoost Temporal", 0.8330, fs['adult_temporal_XGBoost']['accuracy_mean'], 0.001)
check("Feasibility TabPFN-Random Temporal", 0.8053, fs['adult_temporal_TabPFN-Random']['accuracy_mean'], 0.001)

print(f"\n=== SUMMARY ===")
print(f"Total checks: {checks}")
print(f"Passed: {checks - len(errors)}")
print(f"Failed: {len(errors)}")
if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  {e}")
else:
    print("\nAll checks passed!")
