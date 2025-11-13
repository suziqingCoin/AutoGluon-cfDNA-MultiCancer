"""
Author : LinDing Group (ZHAO)
Application : file format conversion
Date : 20160901
Usage: Please refer to the README file
Function : Converts the SVM format file to CSV format file!
"""
def print_help_info():
    print("""
    Usage: python SVMtoCSV.py test.svm test.csv.
    
    Other details, Please read the README file
""")
    exit(0)

    
def obtain_maxFeatureNumber(svm_file):
    feature_num = 0
    
    f = open(svm_file)
    for eachline in f:
        final_set = eachline.split()[-1]
        max_feature_number = int(final_set.split(':')[0])
        feature_num = max(feature_num,max_feature_number)   

    f.close()

    return feature_num

def generate_csv_file(svm_file,csv_file,feature_num):
    g = open(csv_file,'w')
    g.write(get_note_line(feature_num))

    f = open(svm_file)
    for eachline in f:
        eachline = eachline.strip().split()
        label = eachline[0]
        value = get_csv_value_line(eachline[1:],feature_num)

        g.write('%s,%s\n'%(label,value))

    f.close()
    g.close()

    
def get_note_line(feature_num):
    note_line = 'class'

    for i in range(feature_num):
        note_line += ',f_%d'%(i+1)
    note_line = note_line + '\n'

    return note_line
        

def get_csv_value_line(temp,feature_num):
    value_line = ''

    count = 0
    for each in temp:
        count += 1
        [label,value] = each.split(':')
        label = int(label)

        if count == label:
            pass
        elif count < label:
            value_line = get_nullValue_value_line(value_line,count,label)
            count = label
        else:
            print('Note: There are some errors!')
            assert 0
            
        value_line += ',%s'%(value)

    if count < feature_num:
        value_line = value_line + ',0'*(feature_num-count)

    value_line = value_line.strip(',')
    
    return value_line
        

def get_nullValue_value_line(value_line,start,end):
    nullValue_line = ',0' * (end-start) 
    value_line = value_line + nullValue_line
    return value_line
            
        
    

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
        svm_file = sys.argv[1]
        csv_file = sys.argv[2]
except IndexError:
    print_help_info()

feature_num = obtain_maxFeatureNumber(svm_file)

if __name__ == '__main__':
    generate_csv_file(svm_file,csv_file,feature_num)


