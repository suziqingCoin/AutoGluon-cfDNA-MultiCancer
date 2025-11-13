# -*- coding: UTF-8 -*-
"""
Script: BinomialToCSV.py
Usage: python BinomialToCSV.py binomial_sorted.csv input_data.csv
Description: Reorders features in input_data.csv according to the binomial test result file.
"""

import pandas as pd
import os
import sys

def print_help_info():
    print("""
Usage: python BinomialToCSV.py binomial_sorted.csv input_data.csv

* binomial_sorted.csv - The file containing sorted features by binomial test (e.g., output from Binomial.py)
* input_data.csv - The raw CSV file with original samples and features
    """)
    exit(0)

def get_sorted_feature_list(binomial_sorted_file):
    sorted_features = []
    with open(binomial_sorted_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('Rank') or line.strip() == '':
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                sorted_features.append(parts[1])
    return sorted_features

def generate_binomial_sorted_csv(input_csv, binomial_sorted_file):
    output_filename = os.path.splitext(input_csv)[0] + "_BinomialSorted.csv"

    df = pd.read_csv(input_csv)

    # 将标签列 'Label' 改名为 'class'（如果存在）
    if 'Label' in df.columns:
        df.rename(columns={'Label': 'class'}, inplace=True)

    sorted_features = get_sorted_feature_list(binomial_sorted_file)

    # 检查缺失特征
    missing = [fea for fea in sorted_features if fea not in df.columns]
    if missing:
        print("Warning: 以下特征在输入文件中不存在，将跳过：", missing)

    valid_features = [fea for fea in sorted_features if fea in df.columns]

    # 生成排序后的 DataFrame
    reordered_df = df[['class'] + valid_features]
    reordered_df.to_csv(output_filename, index=False)

    print(f"\n排序后的数据已保存至：{output_filename}")
    print(f"使用特征数: {len(valid_features)}")

# 主程序入口
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print_help_info()

    binomial_sorted_file = sys.argv[1]
    input_csv = sys.argv[2]

    generate_binomial_sorted_csv(input_csv, binomial_sorted_file)
