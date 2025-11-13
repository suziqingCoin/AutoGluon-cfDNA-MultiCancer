# -*- coding: UTF-8 -*-
import sys
import pandas as pd
import itertools
from scipy.stats import binomtest, chi2
import numpy as np

def print_help_info():
    print("""
    Usage: python BinomialFisherSelector.py input.csv output.txt

    This script performs binomial test for multi-class labels using pairwise comparison,
    then combines p-values using Fisher's method to rank features.
    """)
    exit(0)

def compute_fisher_combined_p(p_values):
    """使用 Fisher 方法合并多个 p 值"""
    adjusted_ps = [p if p > 1e-300 else 1e-300 for p in p_values]
    stat = -2 * sum(np.log(adjusted_ps))
    df = 2 * len(p_values)
    return chi2.sf(stat, df)

def generate_binomial_fisher_sorted_file(csvfile, outputfile):
    df = pd.read_csv(csvfile)
    
    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]
    unique_classes = sorted(labels.unique())

    print(f"Detected {len(unique_classes)} classes: {unique_classes}")
    
    all_class_pairs = list(itertools.combinations(unique_classes, 2))  # 所有类别组合

    feature_scores = {}

    for feature_name in features.columns:
        p_values = []
        for class_a, class_b in all_class_pairs:
            group_a = features[labels == class_a][feature_name]
            group_b = features[labels == class_b][feature_name]
            pooled = pd.concat([group_a, group_b])
            threshold = pooled.median()

            k1 = sum(group_a > threshold)
            n1 = len(group_a)
            k2 = sum(group_b > threshold)
            n2 = len(group_b)

            # 如果 n=0，跳过这对组合
            if n1 == 0 or n2 == 0:
                continue

            p1 = binomtest(k1, n=n1, p=0.5, alternative='two-sided').pvalue
            p2 = binomtest(k2, n=n2, p=0.5, alternative='two-sided').pvalue

            # 记录两组中的最小 p 值
            p_values.append(min(p1, p2))

        if len(p_values) == 0:
            combined_p = 1.0
        else:
            combined_p = compute_fisher_combined_p(p_values)

        feature_scores[feature_name] = combined_p

    # 排序输出
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1])

    with open(outputfile, 'w') as f:
        f.write("Rank\tFeature\tFisher_Combined_P\n")
        for i, (feature, pval) in enumerate(sorted_features, start=1):
            f.write(f"{i}\t{feature}\t{pval:.6e}\n")

    print("Feature selection complete. Results saved to:", outputfile)

# 入口处理
if sys.version[0] == '2':
    print("Please use Python 3 or later.")
    exit(0)

try:
    if sys.argv[1] == "--help":
        print_help_info()
    else:
        csvfile = sys.argv[1]
        outputfile = sys.argv[2]
except:
    print_help_info()

if __name__ == '__main__':
    generate_binomial_fisher_sorted_file(csvfile, outputfile)
