#!/usr/bin/env python3
# coding: utf-8
# Config

import pandas as pd
import os
import itertools

def get_model_names():
    return ['LogisticRegression','RandomForestClassifier','KNeighborsClassifier',
        'LinearSVC','XGBClassifier', 'DecisionTreeClassifier', 'LinearDiscriminantAnalysis']

def get_combinations(descriptor_list):
    # Get combinations of the molecule descriptors /
    #                         molecule descriptors + "qvina" and/or "rfscore_qvina" /
    #                         molecule descriptors + "plants" and/or "rfscore_plants"
    combinations = []
    docking = [['qvina','rfscore_qvina'], ['plants','rfscore_plants']]
    for sublist in docking:
        c = descriptor_list + sublist
        combinations += list(itertools.chain.from_iterable(
            itertools.combinations(c, i) for i in range(2, len(c)+1)))

    # Remove duplicates and return list
    return list(dict.fromkeys(combinations))

def write_job(job_id, model_name, subset, trainset, activity_label, cwd, args):
    data_file = args.data_file
    write_dir = args.write_dir
    cmd = f'''{args.EXEC} run.py -j {job_id} -m "{model_name}" \
-s "{str(subset)}" -t "{str(trainset)}" -l "{activity_label}" \
-r {data_file} -w {write_dir}'''

    # If kfold_true is False
    if 'false' in args.KFold:
        cmd += '-k false'

    if not os.path.isdir(f'{write_dir}/{job_id}'):
        os.mkdir(f'{write_dir}/{job_id}')
    with open(f'{write_dir}/{job_id}/job.sh', 'w+') as file:
        file.write(f'''#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -o {write_dir}/{job_id}/out.log
#$ -j y

{cmd}''')
    return

def write_all(combinations, model_list, trainset, cwd, args):
    data_file = args.data_file
    write_dir = args.write_dir
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f'{data_file} does not exist')
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    job_id = 0
    for activity_label in ['f_activity']:
        for subset in combinations:
            subset = list(subset)
            for model_name in model_list:
                write_job(job_id, model_name, subset, trainset, 
                    activity_label, cwd, args)
                job_id += 1
    return

def get_cmd_line():
    import argparse
    parser = argparse.ArgumentParser(description='Write SGE job files')
    parser.add_argument('-r', '--data_file', action='store', dest='data_file', 
        required=True, help='Path to the data')
    parser.add_argument('-w', '--write_dir', action='store', dest='write_dir', 
        required=True, help='Path to the directory where the output files will be written')
    parser.add_argument('--KFold', action='store', dest='KFold', 
        required=False, choices=['true','false'], type=str.lower, 
        default='true', help='Default="true"')
    parser.add_argument('EXEC', action='store', help='(Required) Python executable')
    return parser.parse_args()

def main():
    # Please make sure the data is up to date before running this script
    # Re-run the ML.ipynb notebook if needed
    args = get_cmd_line()

    # Read descriptors
    descriptors = pd.read_csv('descriptors.csv')

    # Descriptors
    descriptor_list = list(descriptors.columns[1:])
    docking_list = ['qvina','rfscore_qvina','plants','rfscore_plants']
    trainset = descriptor_list + docking_list

    cwd = os.getcwd() # current working directory
    model_list = get_model_names()
    combinations = get_combinations(descriptor_list)

    try:
        write_all(combinations, model_list, trainset, cwd, args)
    except Exception as e:
        print(str(e))

if __name__=='__main__': main()
