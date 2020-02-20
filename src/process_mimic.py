# This script processes MIMIC-III dataset and builds a binary matrix or a count matrix depending on your input.
# The output matrix is a Numpy matrix of type float32, and suitable for training medGAN.
# Written by Edward Choi (mp2893@gatech.edu)
# Usage: Put this script to the folder where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file> <"binary"|"count">
# Note that the last argument "binary/count" determines whether you want to create a binary matrix or a count matrix.

# Output files
# <output file>.pids: cPickled Python list of unique Patient IDs. Used for intermediate processing
# <output file>.matrix: Numpy float32 matrix. Each row corresponds to a patient. Each column corresponds to a ICD9 diagnosis code.
# <output file>.types: cPickled Python dictionary that maps string diagnosis codes to integer diagnosis codes.

# 中文翻译 该脚本用于将 MIMIC-III 数据集转换成二进制矩阵或者计数矩阵（取决于你的参数）
# 输出矩阵是float32的numpy矩阵，适合训练medgan
# 翻译自skywf(skywf.lzj@gmail.com)
# 将该脚本放在MIMIC-III CSV文件所在位置，然后运行以下命令
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file> <"binary"|"count">
# 其中个参数含义如下：
# output file：输出路径，"binary"|"count"转换成二进制矩阵或者计数矩阵
#剩下的两个参数可见如下网址
# https://blog.csdn.net/m0_37520426/article/details/89321998

import sys
import _pickle as pickle
import numpy as np
from datetime import datetime

# def convert_to_icd9(dxStr):
#     if dxStr.startswith('E'):
#         if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
#         else: return dxStr
#     else:
#         if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
#         else: return dxStr
    
def convert_to_3digit_icd9(dxStr):
    return dxStr[:3]
    # if dxStr.startswith('E'):
    #     if len(dxStr) > 4: return dxStr[:4]
    #     else: return dxStr
    # else:
    #     if len(dxStr) > 3: return dxStr[:3]
    #     else: return dxStr

if __name__ == '__main__':
    admissionFile = sys.argv[1]
    diagnosisFile = sys.argv[2]
    outFile = sys.argv[3]
    binary_count = sys.argv[4]
    aggregate = int(sys.argv[5])

    # admissionFile = "mimic_raw/ADMISSIONS.csv"
    # diagnosisFile = "mimic_raw/DIAGNOSES_ICD.csv"

    # binary_count = "binary"

    if binary_count != 'binary' and binary_count != 'count':
        print('You must choose either binary or count.')
        sys.exit()

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        if aggregate:
            pid = int(tokens[0])
        else:
            pid = int(tokens[1])
            
        admId = int(tokens[1])
        try: 
            admTime = datetime.strptime(tokens[2], '%Y-%m-%d %H:%M:%S')
        except:
            print(line)
            exit(0)
        admDateMap[admId] = admTime
        if pid in pidAdmMap: pidAdmMap[pid].append(admId)
        else: pidAdmMap[pid] = [admId]
    infd.close()

    # if aggregate:
    #     token_id = 


    print('Building admission-dxList mapping')
    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[1])
        #dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        dxStr = 'D_' + convert_to_3digit_icd9(tokens[3])
        if admId in admDxMap: admDxMap[admId].append(dxStr)
        else: admDxMap[admId] = [dxStr]        
    infd.close()


    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        #if len(admIdList) < 2: continue
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
        pidSeqMap[pid] = sortedList
    
    print('Building pids, dates, strSeqs')
    pids = []
    dates = []
    seqs = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)
    
    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    print('Constructing the matrix')
    numPatients = len(newSeqs)
    numCodes = len(types)
    matrix = np.zeros((numPatients, numCodes)).astype('float32')
    for i, patient in enumerate(newSeqs):
        for visit in patient:
            for code in visit:
                if binary_count == 'binary':
                    matrix[i][code] = 1.
                else:
                    matrix[i][code] += 1.

    pickle.dump(pids, open(outFile+'.pids', 'wb'), -1)
    pickle.dump(matrix, open(outFile+'.matrix', 'wb'), -1)
    pickle.dump(types, open(outFile+'.types', 'wb'), -1)

    print(matrix.shape)