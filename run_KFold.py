#!/usr/bin/env python3
# coding: utf-8
# Config

import pandas as pd
import numpy as np
import csv

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
        return XGBClassifier(objective='reg:logistic', n_estimators=40, 
            max_depth=3, eta=0.2, random_state=13)
    elif model_name == 'DecisionTreeClassifier':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=6, random_state=13)
    elif model_name == 'LinearDiscriminantAnalysis':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(solver='svd')
    else:
        raise ValueError(f'{model_name} is not a valid module')

def get_cmd_line():
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('-j', '--job', action='store', dest='job', 
        required=True, help='Job ID')
    parser.add_argument('-m', '--model', action='store', dest='model', 
        required=True, help='Model name')
    parser.add_argument('-s', '--subset', action='store', dest='subset', 
        required=True, help='Subset of descriptors')
    parser.add_argument('-t', '--trainset', action='store', dest='trainset', 
        required=True, help='Training set')
    parser.add_argument('-l', '--activity_label', action='store', dest='activity_label', 
        required=True, help='Activity label', choices=['r_activity','f_activity'])
    parser.add_argument('-r', '--data_file', action='store', dest='data_file', 
        required=True, help='Path to the data')
    parser.add_argument('-w', '--write_dir', action='store', dest='write_dir', 
        required=True, help='Path to the directory where the output files will be written')
    arg_dict = vars(parser.parse_args())
    return arg_dict

def get_scores_list(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import roc_auc_score
    from imblearn.metrics import geometric_mean_score
    
    return np.array([accuracy_score(y_true, y_pred),      # test_accuracy
                    precision_score(y_true, y_pred),      # test_precision
                    recall_score(y_true, y_pred),         # test_recall
                    f1_score(y_true, y_pred),             # test_f1
                    fbeta_score(y_true, y_pred, beta=2),  # test_f2
                    geometric_mean_score(y_true, y_pred), # test_geometric_mean
                    roc_auc_score(y_true, y_pred)         # test_roc_auc
                    ])

def get_scores_list_KFold(X, y, pipe):
    from sklearn.model_selection import RepeatedStratifiedKFold

    scores_list_KFold = []
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index] 
        model_fitted = pipe.fit(X_train, y_train)
        y_pred = model_fitted.predict(X_test)
        scores_list_KFold.append(get_scores_list(y_test, y_pred))
    
    return scores_list_KFold

def get_mean_scores(X, y, scaler, model):
    from imblearn.pipeline import make_pipeline
    from imblearn.over_sampling import SMOTE

    scores_list_pipeline = []
    # Random list generated with np.random.randint()
    seed_list = np.array([40, 15, 72, 22, 43, 82, 75,  7, 34, 49])
    for random_state in seed_list:
        pipe = make_pipeline(SMOTE(random_state=random_state), scaler, model)
        scores_list_pipeline.extend(get_scores_list_KFold(X, y, pipe))

    all_scores = pd.DataFrame(scores_list_pipeline)
    mean_scores = list(all_scores.mean())
    return mean_scores

def get_scores(X, y, model_name, subset, trainset, scaler, activity_label):
    try:
        model = get_model_by_name(model_name)
    except Exception as e:
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
    
    data = pd.read_csv(f'{data_file}').dropna(subset=[activity_label])
    y = data[activity_label]
    X = data[subset]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scores = get_scores(X, y, model_name, subset, trainset, scaler, activity_label)

    with open(f'{write_dir}/{job_id}/score.csv', 'w+') as file:
        wr = csv.writer(file)
        wr.writerow(scores)

if __name__=='__main__': main()
