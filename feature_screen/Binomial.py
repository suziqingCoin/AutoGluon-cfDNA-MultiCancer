# -*- coding: UTF-8 -*- 

import sys
import scipy.stats as stats

def print_help_info():
    print("""
    Usage: python BinomialSelector.py input.csv output.txt

    Other details, please read the README file!
    """)
    exit(0)

def generate_Binomial_sorted_file(csvfile, outputfile):
    [scores, group_num, sample_num, total_sample, feature_num] = obtain_binomial_scores(csvfile)
    scores = sorted(scores.items(), key=lambda x: x[1])  # p值越小越显著

    g = open(outputfile, 'w')
    g.write(get_output_prompt(group_num, sample_num, total_sample, feature_num))
    g.write('-------Binomial Test Sorted Feature Set-------\n')
    g.write('Rank\tfeature\tp_value\n')
    for i, (feature, p_value) in enumerate(scores):
        g.write('%d\t%s\t%f\n' % (i + 1, feature, p_value))
    g.close()
    print('Successful!')

def obtain_binomial_scores(filename):
    note_label, key_words, csv_dics = csv_to_dict(filename)
    feature_num = len(note_label)
    scores = dict()
    group_num, sample_num, total_sample = get_sample_num(key_words)

    for label_index in range(len(note_label)):
        label = note_label[label_index]
        assign_label_values = get_assign_label_values(csv_dics, label_index)
        scores[label] = calculate_binomial_p(assign_label_values)

    return scores, group_num, sample_num, total_sample, feature_num

def csv_to_dict(filename):
    csv_dics = dict()
    key_words = []

    f = open(filename).readlines()
    note_label = get_csv_note_label(f[0])  # 第一行是特征名
    f = f[1:]
    for line in f:
        parts = line.strip().split(',')
        key_word = parts[0]
        values = parts[1:]
        key_words.append(key_word)
        csv_dics.setdefault(key_word, []).append(values)

    return note_label, key_words, csv_dics

def get_csv_note_label(seq):
    return seq.strip().split(',')[1:]

def get_sample_num(key_words):
    sample_num = {}
    total_sample = 0
    group_num = len(set(key_words))
    for label in set(key_words):
        count = key_words.count(label)
        sample_num[label] = count
        total_sample += count
    return group_num, sample_num, total_sample

def get_assign_label_values(csv_dics, label_index):
    assign_label_values = {}
    for class_label in csv_dics:
        assign_label_values[class_label] = [float(row[label_index]) for row in csv_dics[class_label]]
    return assign_label_values

def calculate_binomial_p(assign_label_values):
    keys = list(assign_label_values.keys())
    class1_vals = assign_label_values[keys[0]]
    class2_vals = assign_label_values[keys[1]]

    threshold = (sum(class1_vals) + sum(class2_vals)) / (len(class1_vals) + len(class2_vals))

    k1 = sum(1 for v in class1_vals if v > threshold)
    n1 = len(class1_vals)
    k2 = sum(1 for v in class2_vals if v > threshold)
    n2 = len(class2_vals)

    # 修复：取出 p 值
    p1 = stats.binomtest(k1, n1, p=0.5, alternative='two-sided').pvalue
    p2 = stats.binomtest(k2, n2, p=0.5, alternative='two-sided').pvalue

    return min(p1, p2)


def get_output_prompt(group_num, sample_num, total_sample, feature_num):
    content = 'There is a %d classification problem!\n\tTotal sample numbers : %d\n\tTotal feature numbers : %d\n\n' \
              % (group_num, total_sample, feature_num)
    for label in sample_num:
        content += '\t\tThe sample numbers of label \'%s\' : %d\n' % (label, sample_num[label])
    return content + '\n'

# 主程序入口
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
    generate_Binomial_sorted_file(csvfile, outputfile)
