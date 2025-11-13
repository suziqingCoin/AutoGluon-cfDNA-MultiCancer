# -*- coding: UTF-8 -*-

import pandas as pd
import sys

def print_help_info():
    print("""
Usage: python ReliefFtoCSV.py ReliefF_sorted.csv input_data.csv

- ReliefF_sorted.csv: The output from ReliefF.py containing ranked features.
- input_data.csv: The original data file.

This script reorders the input CSV file based on feature importance.
""")
    exit(0)

def obtain_relieff_sorted_features(relief_sorted_file):
    sorted_features = []
    with open(relief_sorted_file) as f:
        for line in f.readlines()[1:]:  # 跳过表头
            sorted_features.append(line.strip().split(",")[1])  # 读取特征名
    return sorted_features

def reorder_csv_based_on_relieff(input_csv, relief_sorted_file):
    # 读取原始数据
    df = pd.read_csv(input_csv)

    # 确保 `class` 列存在
    if "Label" in df.columns:
        df.rename(columns={"Label": "class"}, inplace=True)
    
    # 获取排序后的特征名
    sorted_features = obtain_relieff_sorted_features(relief_sorted_file)
    
    # 重新排列数据集（保留第一列 `class` 作为分类标签）
    reordered_df = df[['class'] + sorted_features]

    # 保存结果
    output_filename = input_csv.replace(".csv", "_ReliefFSorted.csv")
    reordered_df.to_csv(output_filename, index=False)

    print("Reordered data saved in:", output_filename)

# 检测 Python 版本并获取命令行参数
if sys.version[0] == '2':
    print("\nVersionError: Please use Python 3 or later!\n")
    exit(0)

try:
    if sys.argv[1] == "--help":
        print_help_info()
    else:
        relief_sorted_file = sys.argv[1]
        input_csv = sys.argv[2]
except:
    print_help_info()

if __name__ == '__main__':
    reorder_csv_based_on_relieff(input_csv, relief_sorted_file)
