#!/usr/bin/env python3
# coding: utf-8
# Config

import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import fbeta_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.metrics import geometric_mean_score

def get_model_by_name(model_name):
    if model_name == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(solver='liblinear', random_state=13)
    elif model_name == 'LinearSVC':
        from sklearn.svm import LinearSVC
        return LinearSVC(dual=False, random_state=13)
    elif model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=40, max_depth=6, random_state=13)
    elif model_name == 'KNeighborsClassifier':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == 'XGBClassifier':
        from xgboost import XGBClassifier
        return XGBClassifier(objective='reg:logistic', n_estimators=40, max_depth=3, eta=0.2, random_state=13)
    elif model_name == 'DecisionTreeClassifier':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=6, random_state=13)
    elif model_name == 'LinearDiscriminantAnalysis':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(solver='svd')
    else:
        raise ValueError(f'IllegalArgumentException: {model_name}')

def get_cmd_line():
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('-j', '--job', action='store', dest='job', required=True, help='Job ID')
    parser.add_argument('-m', '--model', action='store', dest='model', required=True, help='Model name')
    parser.add_argument('-s', '--subset', action='store', dest='subset', required=True, help='Subset of descriptors')
    parser.add_argument('-t', '--trainset', action='store', dest='trainset', required=True, help='Training set')
    parser.add_argument('-a', '--activity_label', action='store', dest='activity_label', required=True, help='Activity label', choices=['r_active','f_active'])
    parser.add_argument('-r', '--data_file', action='store', dest='data_file', required=True, help='Path for the data')
    parser.add_argument('-w', '--write_dir', action='store', dest='write_dir', required=True, help='Path for the directory where the output files will be written')
    arg_dict = vars(parser.parse_args())
    return arg_dict

def get_scores_list(X_train, X_test, y_train, y_test, pipe):
    scores_list = []
    model_fitted = pipe.fit(X_train, y_train)
    y_pred = model_fitted.predict(X_test)
    scores_list.append(accuracy_score(y_test, y_pred))       # test_accuracy
    scores_list.append(precision_score(y_test, y_pred))      # test_precision
    scores_list.append(recall_score(y_test, y_pred))         # test_recall
    scores_list.append(f1_score(y_test, y_pred))             # test_f1
    scores_list.append(fbeta_score(y_test, y_pred, beta=2))  # test_f2
    scores_list.append(geometric_mean_score(y_test, y_pred)) # test_geometric_mean
    scores_list.append(roc_auc_score(y_test, y_pred))        # test_roc_auc
    return scores_list

def get_scores_list_KFold(X, y, pipe, random_state_KFold):
    scores_list_KFold = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state_KFold)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index] 
        scores_list_KFold.append(get_scores_list(X_train, X_test, y_train, y_test, pipe))
    return scores_list_KFold

def get_scores_list_pipeline(X, y, scaler, model, random_state_smote):
    # Dictionary of scoring metrics and corresponding scores for each KFold iteration
    pipe = make_pipeline(SMOTE(random_state=random_state_smote), scaler, model)
    # Random list generated with np.random.randint()
    seed_list_KFold = np.array([46, 55, 69,  1, 87, 72, 50,  9, 58, 94])
    scores_list_pipeline = []
    for random_state_KFold in seed_list_KFold:
        scores_list_pipeline.extend(get_scores_list_KFold(X, y, pipe, random_state_KFold))
    return scores_list_pipeline

def get_mean_scores(X, y, scaler, model):
    # Random list generated with np.random.randint()
    seed_list_smote = np.array([40, 15, 72, 22, 43, 82, 75,  7, 34, 49])
    scores_list = []
    for random_state_smote in seed_list_smote:
        scores_list.extend(get_scores_list_pipeline(X, y, scaler, model, random_state_smote))

    df = pd.DataFrame(scores_list)
    mean_scores = list(df.mean())
    return mean_scores

def get_scores(X, y, model_name, subset, trainset, scaler, activity_label):
    try:
        model = get_model_by_name(model_name)
    except ValueError as e:
        print(str(e))
        quit()
    
    scores = get_mean_scores(X, y, scaler, model)
    scores.append(activity_label)
    scores.append(model_name)

    # Add binary list of the descriptors
    for i in trainset:
        if i in subset:
            scores.append(1)
        else:
            scores.append(0)
    return scores

def main():
    args = get_cmd_line()
    job_id = int(args['job'])
    model_name = args['model']
    subset = eval(args['subset'])
    trainset = eval(args['trainset'])
    activity_label = args['activity_label']
    data_file = args['data_file']
    write_dir = args['write_dir']
    
    # Drop NaN activity and descriptor values
    data = pd.read_csv(f'{data_file}').dropna(subset=[activity_label])
    y = data[activity_label]
    X = data[subset]
    scaler = StandardScaler()
    scores = get_scores(X, y, model_name, subset, trainset, scaler, activity_label)

    with open(f'{write_dir}/{job_id}/score.csv', 'w+') as file:
        wr = csv.writer(file)
        wr.writerow([job_id, *scores])

if __name__=='__main__': main()
