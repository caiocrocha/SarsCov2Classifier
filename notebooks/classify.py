#!/usr/bin/env python3

def load_pickle(clf):
    import pickle
    with open(f'pickle/{clf}.pickle', 'rb') as file:
        return pickle.load(file)

def write_mordred_descriptors(smiles, csv, overwrite=False):
    import os
    import subprocess
    
    if os.path.isfile(smiles):
        if overwrite or not os.path.isfile(csv + '.gz'):
            subprocess.run(f'python -m mordred {smiles} > {csv}', 
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    if os.path.isfile(csv):
        subprocess.run(f'gzip -f {csv}', shell=True)

def get_cmd_line():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Classify one compound or a series of compounds.')
    parser.add_argument('csv', metavar='CSV file', help='CSV file containing "CID" and the SMILES structure of the ligand(s). Columns: CID,SMILES')
    parser.add_argument('clf', metavar='Classifier', help='Classifier', choices=['LogisticRegression','RandomForestClassifier','KNeighborsClassifier','LinearSVC','XGBClassifier'])
    return vars(parser.parse_args())

def main():
    import pandas as pd
    import os

    args = get_cmd_line()
    pipe = load_pickle(args['clf'])
    data = pd.read_csv(args['csv'])

    if not os.path.isdir('classify.log'):
        os.mkdir('classify.log')
    
    # Write SMILES and calculate molecular descriptors
    data[['SMILES','CID']].to_csv('classify.log/smiles.smi', sep=' ', index=False, header=None)
    write_mordred_descriptors('classify.log/smiles.smi', 'classify.log/mordred.csv', overwrite=True)

    # Read descriptors
    descriptors = pd.read_csv('classify.log/mordred.csv.gz', compression='gzip')
    descriptors.rename(columns={'name': 'CID'}, inplace=True)
    
    with open('features.lst', 'r') as file:
        features = file.read().splitlines()
    
    X = descriptors[['CID']+features]
    X.set_index('CID', inplace=True)
    y_pred = pd.Series(pipe.predict(X), index=X.index)
    y_proba = pd.Series(pipe.predict_proba(X)[:,1], index=X.index)

    predictions = pd.DataFrame(index=X.index)
    predictions['prediction'] = y_pred.replace({1: 'active', 0: 'inactive'})
    predictions['probability'] = y_proba
    predictions.sort_values('probability', ascending=False, inplace=True)
    predictions.to_csv('classify.log/predictions.csv', index=True)

    counts = predictions['prediction'].value_counts()
    if 'active' not in counts.index:
        counts['active'] = 0
    elif 'inactive' not in counts.index:
        counts['inactive'] = 0
    
    out = f'''{"Compounds":10s} {"Active":10s} {"Inactive":10s}
{len(predictions):<10d} {counts["active"]:<10d} {counts["inactive"]:<10d}'''

    print(out)
    with open('classify.log/out.txt', 'w+') as file:
        file.write(out)
    print('Outputs available in the classify.log directory')
    print('Note: the probability column in predictions.csv represents the probability of the compound being active')

if __name__ == '__main__': main()