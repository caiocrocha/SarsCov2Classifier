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
        from sklearn.svm import SVC, LinearSVC
        return CalibratedClassifierCV(LinearSVC(dual=False, random_state=13))
    elif model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=13)
    elif model_name == 'KNeighborsClassifier':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == 'XGBClassifier':
        from xgboost import XGBClassifier
        return XGBClassifier(objective='reg:logistic', random_state=13)
    elif model_name == 'DecisionTreeClassifier':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=13)
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
    parser.add_argument('-rs', '--random_state', action='store', dest='random_state', required=True, help='Random seed for StratifiedKFold')
    parser.add_argument('-a', '--activity_label', action='store', dest='activity_label', required=True, help='Activity label', choices=['r_active','f_active'])
    parser.add_argument('-r', '--read_path', action='store', dest='read_path', required=True, help='Path for the data')
    parser.add_argument('-w', '--write_dir', action='store', dest='write_dir', required=True, help='Path for the directory where the output files will be written')
    arg_dict = vars(parser.parse_args())
    return arg_dict

def main():
    args = get_cmd_line()
    job_id = int(args['job'])
    model_name = args['model']
    subset = eval(args['subset'])
    trainset = eval(args['trainset'])
    random_state = int(args['random_state'])
    activity_label = args['activity_label']
    read_path = args['read_path']
    write_dir = args['write_dir']

    model = get_model_by_name(model_name)
    # Drop NaN activity and descriptor values
    data = pd.read_csv(f'{read_path}').dropna(subset=[activity_label])
    y = data[activity_label]
    X = data[subset]

    pipe = make_pipeline(SMOTE(random_state=42), StandardScaler(), model)
    scoring_metrics = ['accuracy','precision','recall','f1','f2','g_mean','roc_auc']
    metrics = {key: [] for key in scoring_metrics}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model_fitted = pipe.fit(X_train, y_train)
        y_pred = model_fitted.predict(X_test)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['f2'].append(fbeta_score(y_test, y_pred, beta=2))
        metrics['g_mean'].append(geometric_mean_score(y_test, y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))

    scores = [np.mean(value) for value in metrics.values()]
    scores.append(activity_label)
    scores.append(str(model).split('(')[0])
    scores.append(random_state)
    scores.extend([i in subset for i in trainset])
    with open(f'{write_dir}/{job_id}/score.csv', 'w+') as file:
        csv.writer(file).writerow(scores)

if __name__=='__main__': main()
