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

def write_job(job_id, model_name, subset, trainset, 
    activity_label, cwd, data_file, write_dir, args):

    if not os.path.isdir(f'{write_dir}/{job_id}'):
        os.mkdir(f'{write_dir}/{job_id}')

    with open(f'{write_dir}/{job_id}/job.sh', 'w+') as file:
        file.write(f'''#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o {write_dir}/{job_id}/out.log
#$ -j y

{args.EXEC} run.py -j {job_id} -m "{model_name}" \
    -s "{str(subset)}" -t "{str(trainset)}" -a "{activity_label}" \
    -r {data_file} -w {write_dir}
''')
    return

def write_all(combinations, model_list, trainset, cwd, data_file, write_dir, args):
    job_id = 0
    for activity_label in ['r_active','f_active']:
        for subset in combinations:
            subset = list(subset)
            for model_name in model_list:
                write_job(job_id, model_name, subset, trainset, 
                    activity_label, cwd, data_file, write_dir, args)
                job_id+=1
    return

def get_cmd_line():
    import argparse
    parser = argparse.ArgumentParser(description='Write SGE job files')
    parser.add_argument('EXEC', action='store', dest='EXEC', required=True, help='Python executable')
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
    data_file = 'data.csv'

    write_dir = 'KFold'
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    model_list = get_model_names()
    combinations = get_combinations(descriptor_list)

    write_all(combinations, model_list, trainset, cwd, data_file, write_dir, args)

if __name__=='__main__': main()
