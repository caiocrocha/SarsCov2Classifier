#!/usr/bin/env python3
# coding: utf-8
# Config

import pandas as pd
import numpy as np

def get_model_and_params(model_name, nsamples):
    if model_name == 'KNeighborsClassifier':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier()
        params = {'clf__n_neighbors': [n for n in range(5, int(nsamples**0.5), 2)]}
    elif model_name == 'XGBClassifier':
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=13)
        params = {'clf__n_estimators': [50, 100, 200, 500, 1000], 
                 'clf__max_depth': [3, 4, 5, 6, 7, 8, 9, None], 
                 'clf__eta': [0.001, 0.01, 0.1, 0.2, 0.3]}
    elif model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=13)
        params = {'clf__n_estimators': [50, 100, 200, 500, 1000], 
                 'clf__max_depth': [3, 4, 5, 6, 7, 8, 9, None]}
    elif model_name == 'DecisionTreeClassifier':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=13)
        params = {'clf__max_depth': [3, 4, 5, 6, 7, 8, 9, None]}
        
    return clf, params

def feature_transfomer(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
        
    continuous = ['float16', 'float32', 'float64']
    discrete = ['int16', 'int32', 'int64']

    return ColumnTransformer(transformers=[
        ('continuous', StandardScaler(), X.select_dtypes(include=continuous).columns.tolist()), 
        ('discrete', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=discrete).columns.tolist())
    ])

def mean_score(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import fbeta_score
    from imblearn.metrics import geometric_mean_score

    auc = roc_auc_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    geometric_mean = geometric_mean_score(y_true, y_pred)

    return (auc + f2 + geometric_mean)/3

def grid_search(X, y, transformer, model, params):
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    
    mean = make_scorer(mean_score, greater_is_better=True)

    pipe = Pipeline(steps=[('smote', SMOTE(random_state=42)), ('transformer', transformer), ('clf', model)])
    grid = GridSearchCV(estimator=pipe, param_grid=params, cv=RepeatedStratifiedKFold(n_splits=10), 
                        scoring=mean, n_jobs=-1)
    grid.fit(X, y)
    return grid

def grid_dataframe(grid, model_name):
    results = []
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        test_score = {'clf': model_name, 'mean_test_score': mean, 'std_test_score': std}
        results.append({**test_score, **params}) # Concatenate two dicts
    
    return pd.DataFrame(results)

def get_cmd_line():
    import argparse
    parser = argparse.ArgumentParser(description='Run Grid Search Cross Validation.')

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

    parser.add_argument('--clf',
                        action='store',
                        dest='model_name',
                        required=True,
                        help='Name of the classifier.')

    parser.add_argument('--directory',
                        action='store',
                        dest='directory',
                        required=True,
                        help='Directory where the output file will be written.')
    
    return vars(parser.parse_args())

def main():
    args = get_cmd_line()

    with open(args['features_file'], 'r') as file:
        features = file.read().splitlines()
    
    data = pd.read_csv(args['data_file'])
    y = data['activity']
    X = data[features]

    nsamples = len(X)
    model_name = args['model_name']
    model, params = get_model_and_params(model_name, nsamples)
    transformer = feature_transfomer(X)
    grid = grid_search(X, y, transformer, model, params)
    results = grid_dataframe(grid, model_name)

    directory = args['directory']
    results.to_csv(f'{directory}/grid.csv', index=False)

if __name__ == '__main__': main()
