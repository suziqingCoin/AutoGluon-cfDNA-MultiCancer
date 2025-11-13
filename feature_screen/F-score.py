# -*- coding: UTF-8 -*- 
"""
Author : LinDing Group (Dao)
Application : feature selection
Date : 20171007
Usage: Please refer to the README file
Intro : This is a feature selection technology based on the analysis of variance!
"""

def print_help_info():
    print("""
    Usage: python F-score.py test.csv test.csv.out
    
    Other details, please read the README file!
""")
    exit(0)

    
def generate_Fscore_sorted_file(csvfile,outputfile):
    [F_values,group_num,sample_num,total_sample,feature_num] = obtain_Fscore_values_dict(csvfile)
    F_values = sorted(F_values.items(), key=lambda asd:asd[1], reverse=True)

    g = open(outputfile,'w')
    g.write(get_output_prompt(group_num,sample_num,total_sample,feature_num))
    g.write('-------Fscore Sorted Feature Set-------\n')
    g.write('Rank\tfeature\tvalue\n')
    for i in range(len(F_values)):
        g.write('%d\t%s\t%f\n'%(i+1,F_values[i][0],F_values[i][1]))

    g.close()
    print('Successful!')


def obtain_Fscore_values_dict(filename):
    [note_label,key_words,csv_dics] = csv_to_dict(filename)
    feature_num = len(note_label)

    F_values = dict()
    [group_num,sample_num,total_sample] = get_sample_num(key_words)

    for label_index in range(len(note_label)):#len(note_label)为特征数，label_index = 
        label = note_label[label_index]

        assign_label_values = get_assign_label_values(csv_dics,label_index)

        F_values[label] = calculate_thisLabel_Fvalue(assign_label_values,group_num,sample_num,total_sample)

    return F_values,group_num,sample_num,total_sample,feature_num
                

def csv_to_dict(filename):
    csv_dics = dict()
    key_words = []

    f = open(filename).readlines()
    note_label = get_csv_note_label(f[0])#得到第一行标签
    f = f[1:]
    for line_index in range(len(f)):#len(f)为样本数
        line = f[line_index].strip().split(',')
        key_word = line[0]; key_value = line[1:]

        key_words.append(key_word)
        if csv_dics.get(key_word) == None:
            csv_dics[key_word] = []
        else:
            pass
        csv_dics[key_word].append(key_value)
        
    return note_label,key_words,csv_dics
    '''note_label=['GG_f', 'GA_f', 'GC_f', 'GT_f', 'AG_f']
    key_words = [2,2,1,1...]
    csv_dics = {'2':[[0.016722,0.016722,0.026756,0.036789],[0.040134,0.036789,0.043478,0.046823]];'1':[[0.016722,0.016722,0.026756,0.036789],[0.016722,0.040134,0.036789,0.043478,0.046823]]} '''


def get_csv_note_label(seq):
    seq = seq.strip().split(',')
    note_label = seq[1:]
    return note_label


def get_sample_num(key_words):
    sample_num = dict()
    total_sample = 0

    key_words_set = set(key_words)#set()具有不重复性，即只有标签1，2两个
    group_num = len(key_words_set)#group_num = 2
    for each in key_words_set:
        assign_sample_num = key_words.count(each)
        sample_num[each] = assign_sample_num#sample_num = {'2':405,'1':406}
        total_sample += assign_sample_num#样本总数 total_sample = 911

    return group_num,sample_num,total_sample


def get_assign_label_values(csv_dics,label_index):
    assign_label_values = dict()

    for each_class in csv_dics.keys():
        content = []
        key_values = csv_dics[each_class]
        for each_values_index in range(len(key_values)):
            content.append(float(key_values[each_values_index][label_index]))
        assign_label_values[each_class] = content

    return assign_label_values#将每一列放到


def calculate_thisLabel_Fvalue(assign_label_values,group_num,sample_num,total_sample):
    [within_group_mean,total_mean] = calculate_withinGroup_and_total_mean(assign_label_values,sample_num,total_sample)

    SSb = 0
    for each_assign_label in within_group_mean.keys():
        SSb += (((within_group_mean[each_assign_label]-total_mean)**2))# * sample_num[each_assign_label])分子

    SSw = calculate_SumOfSquaresWithinGroups(assign_label_values,within_group_mean, sample_num)#分母

    Fvalue = (SSb) / (SSw)#group_num = 2

    return Fvalue


def calculate_withinGroup_and_total_mean(assign_label_values,sample_num,total_sample):
    within_group_mean = dict()
    summary = 0

    for each_group_label in assign_label_values.keys():
        each_group_summary = sum(assign_label_values[each_group_label])

        within_group_mean[each_group_label] = each_group_summary/sample_num[each_group_label]
        summary += each_group_summary

    total_mean = summary/total_sample

    return within_group_mean,total_mean#计算组间均值，分别得到正负均值和整个数据均值


def calculate_SumOfSquaresWithinGroups(assign_label_values,within_group_mean, sample_num):
    '''
    value = 0

    for each_assign_label in assign_label_values.keys():
        for each_value in assign_label_values[each_assign_label]:
            value += ((each_value - within_group_mean[each_assign_label])**2)
    value
    return value#均值差平方累加，value为正负均值差平方累加和的和
    '''
    value = 0

    for each_assign_label in assign_label_values.keys():
        if each_assign_label == "1":
            value2 = 0
            for each_value in assign_label_values[each_assign_label]:
                value2 += ((each_value - within_group_mean[each_assign_label])**2)
            value2 = value2 / float(sample_num[each_assign_label]-1)
        else:
            value1 = 0
            for each_value in assign_label_values[each_assign_label]:
                value1 += ((each_value - within_group_mean[each_assign_label])**2)
            value1 = value1 / float(sample_num[each_assign_label]-1)


    value = value2 + value1
    return value




def get_output_prompt(group_num,sample_num,total_sample,feature_num):
    content = 'There is a %d classification problem!\n\tTotal sample numbers : %d\n\tTotal feature numbers : %d\n\n'\
              %(group_num,total_sample,feature_num)

    for each in sample_num.keys():
        content += '\t\tThe sample numbers of label \'%s\' : %d\n'%(each,sample_num[each])
    content += '\n'

    return content


import sys

if sys.version[0] == '2':
    print("""\nVersionError: The current python version: '%s',
You must use python version 3 or later to run this script!\n"""%((sys.version).split('\n')[0]))
    exit(0)
else:
    pass

try:
    if sys.argv[1] == "--help":
        print_help_info()
    else:
        csvfile = sys.argv[1]
        outputfile = sys.argv[2]
except:
    print_help_info()
    

if __name__ == '__main__':
    generate_Fscore_sorted_file(csvfile,outputfile)

