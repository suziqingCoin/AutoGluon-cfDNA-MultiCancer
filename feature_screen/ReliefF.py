
from skrebate import ReliefF
import pandas as pd
import sys

def print_help_info():
    print("""
    Usage: python ReliefF.py input.csv output.csv

    This script applies ReliefF to rank features based on importance.
    """)
    exit(0)

def compute_relieff_feature_importance(csvfile, outputfile):
    # 读取数据
    df = pd.read_csv(csvfile)
    
    # 提取特征和标签
    X = df.iloc[:, 1:]  # 去除第一列（样本ID）
    y = df.iloc[:, 0]   # 目标变量

    # 运行 ReliefF 计算特征重要性
    relieff = ReliefF(n_neighbors=10)  # 选择10个最近邻
    relieff.fit(X.values, y.values)

    # 获取特征重要性得分并排序
    feature_importances = relieff.feature_importances_
    sorted_features = sorted(zip(feature_importances, X.columns), reverse=True)

    # 保存结果到 CSV
    with open(outputfile, 'w') as g:
        g.write("Rank,Feature,Importance Score\n")
        for i, (score, feature) in enumerate(sorted_features):
            g.write(f"{i+1},{feature},{score:.6f}\n")

    print("ReliefF feature selection completed! Results saved in:", outputfile)


# 检测 Python 版本并获取命令行参数
if sys.version[0] == '2':
    print("\nVersionError: Please use Python 3 or later!\n")
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
    compute_relieff_feature_importance(csvfile, outputfile)
