#!/usr/bin/env python3
# coding: utf-8
# Config

import pandas as pd
import numpy as np
import os
import itertools

def get_combinations(min_len, list1, *args):
    # Get all subsets of a "main" list and N other lists
    combinations = []
    for sublist in args:
        list2 = list1 + sublist
        combinations += list(itertools.chain.from_iterable(
            itertools.combinations(list2, r) for r in range(min_len, len(list2)+1)))

    # Remove duplicates and return list
    return list(dict.fromkeys(combinations))

def write_job(EXEC, job_id, model_name, subset, trainset, random_state, 
    activity_label, cwd, read_path, write_dir):

    if not os.path.isdir(f'{write_dir}/{job_id}'):
        os.mkdir(f'{write_dir}/{job_id}')

    with open(f'{write_dir}/{job_id}/job.sh', 'w+') as file:
        file.write(f'''#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y

{EXEC} {cwd}/run_model.py -j {job_id} -m "{model_name}" \
    -s "{str(subset)}" -t "{str(trainset)}" -rs {random_state} \
    -a "{activity_label}" -r {read_path} -w {write_dir}
''')
    return

def main():
    EXEC="/home/caiocedrola/miniconda3/envs/rdkit/bin/python3.7"

    # Read activity data
    activity = pd.read_csv('activity_data.csv')
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors

    # Write into smiles
    activity[['SMILES','CID']].to_csv('smiles.smi', sep=' ', index=False, header=None)
    suppl = Chem.SmilesMolSupplier('smiles.smi')

    # Compute 2D coordinates
    ms = [x for x in suppl if x is not None]
    d_list = []
    for m in ms:
        d_list.append({'CID': m.GetProp('_Name'), 
                    'MolMR': Descriptors.MolMR(m), 
                    'NumRotatableBonds': Descriptors.NumRotatableBonds(m), 
                    'NumHAcceptors': Descriptors.NumHAcceptors(m), 
                    'NumHDonors': Descriptors.NumHDonors(m), 
                    'TPSA': Descriptors.LabuteASA(m), 
                    'LabuteASA': Descriptors.LabuteASA(m), 
                    'MolLogP': Descriptors.MolLogP(m)})
    descriptors = pd.DataFrame(d_list)
  
    # Read QuickVina02 results
    qvina         = pd.read_csv('qvina.csv')
    qvina.columns = ['CID','pose','qvina']

    # Read QVina rescored with RF-Score
    rfscore_qvina         = pd.read_csv('rfscore_qvina.csv')
    rfscore_qvina.columns = ['CID','pose','rfscore_qvina']

    # Top QVina poses
    top_qvina  = pd.merge(qvina.query('pose == 1'), rfscore_qvina.query('pose == 1'))
    top_qvina.drop('pose', axis=1, inplace=True)

    # Read PLANTS results
    plants         = pd.read_csv('plants.csv')
    plants.columns = ['CID','pose','plants']

    # Read PLANTS rescored with RF-Score
    rfscore_plants         = pd.read_csv('rfscore_plants.csv', header=None)
    rfscore_plants.columns = ['rfscore_plants','CID']
    rfscore_plants         = rfscore_plants[['CID','rfscore_plants']]
    rfscore_plants[['CID','pose']] = rfscore_plants['CID'].str.split('_', expand=True)
    rfscore_plants['pose']         = rfscore_plants['pose'].astype('int')

    # Top PLANTS poses
    top_plants = pd.merge(plants.query('pose == 1'), rfscore_plants.query('pose == 1'))
    top_plants.drop('pose', axis=1, inplace=True)

    # Merge top scores and experimental data
    tmp = pd.merge(top_qvina, top_plants)
    top = pd.merge(tmp, activity[['CID','r_inhibition_at_50_uM','f_inhibition_at_50_uM']])
    del tmp

    # Merge descriptors to "top"
    data = pd.merge(descriptors, top)

    # Label actives and inactives in relation to "RapidFire"
    data['r_active'] = data['r_inhibition_at_50_uM'] > 50
    data['r_active'] = data['r_active'].astype(int) # 1 for active, 0 for inactive

    # Label actives and inactives in relation to the fluorescence assay
    data['f_active'] = data['f_inhibition_at_50_uM'] > 50
    data['f_active'] = data['f_active'].astype(int) # 1 for active, 0 for inactive

    # Descriptors
    descriptor_list = list(descriptors.columns[1:])
    docking_list = ['qvina','rfscore_qvina','plants','rfscore_plants']
    trainset = descriptor_list + docking_list

    cwd = os.getcwd()
    read_path = 'data.csv'
    data.to_csv(read_path, index=False)

    write_dir = 'MODELS'
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    model_list = ['LogisticRegression', 
                    'RandomForestClassifier', 
                    'KNeighborsClassifier',
                    'LinearSVC', 
                    'XGBClassifier', 
                    'DecisionTreeClassifier(random_state=13)', 
                    'LinearDiscriminantAnalysis']

    # Get combinations of the molecule descriptors /
    #                         molecule descriptors + "qvina" and/or "rfscore_qvina" /
    #                         molecule descriptors + "plants" and/or "rfscore_plants"
    combinations = get_combinations(2, descriptor_list, 
                                    ['qvina','rfscore_qvina'], 
                                    ['plants','rfscore_plants'])
    # Test metrics
    scoring_metrics = ['accuracy','precision','recall','f1','f2','g_mean','roc_auc']
    test_metrics = ['test_'+ i for i in scoring_metrics]

    # Random list generated with np.random.randint()
    seed_list = np.array([46, 55, 69,  1, 87, 72, 50,  9, 58, 94])
    
    job_id = 0
    for random_state in seed_list:
        for activity_label in ['r_active','f_active']:
            for subset in combinations:
                subset = list(subset)
                for model in model_list:
                    model_name = str(model).split('(')[0]
                    write_job(EXEC, job_id, model_name, subset, trainset, 
                    random_state, activity_label, cwd, read_path, write_dir)

                    job_id+=1

if __name__=='__main__': main()

