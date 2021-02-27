#!/usr/bin/env python3
# coding: utf-8
# Config

import pandas as pd
import os

def get_model_names():
    return ['LogisticRegression','RandomForestClassifier','KNeighborsClassifier',
        'LinearSVC','XGBClassifier', 'DecisionTreeClassifier']

def write_job(data_file, features_file, model_name, directory, EXEC):
    if not os.path.isdir(f'{directory}/{model_name}'):
        os.makedirs(f'{directory}/{model_name}', exist_ok=True)
    
    with open(f'{directory}/{model_name}/job.sh', 'w+') as file:
        file.write(f'''#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -o {directory}/{model_name}/out.log
#$ -j y

{EXEC} --data {data_file} --features {features_file} --clf {model_name} --directory {directory}''')

    return

def get_cmd_line():
    import argparse
    parser = argparse.ArgumentParser(description='Write SGE jobs.')

    parser.add_argument('--data',
                        action='store',
                        dest='data_file',
                        required=True,
                        help='File containing the compounds and their corresponding descriptors.')
    
    parser.add_argument('--features',
                        action='store',
                        dest='features_file',
                        required=True,
                        help='File containing a list of descriptors.')

    parser.add_argument('--directory',
                        action='store',
                        dest='directory',
                        required=True,
                        help='Directory where the output file will be written.')
    
    return vars(parser.parse_args())

def main():
    # Please make sure the data is up to date before running this script
    # Re-run the ML.ipynb notebook if needed
    args = get_cmd_line()

    for model_name in get_model_names():
        write_job(args['data_file'], args['features_file'], model_name, 
                    directory=args['directory'], EXEC='grid_search.py')

if __name__=='__main__': main()
