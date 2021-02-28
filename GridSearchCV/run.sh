#!/bin/sh

data="../notebooks/data.csv"
features="../notebooks/features.lst"

for clf in "RandomForestClassifier" "KNeighborsClassifier" "XGBClassifier" "DecisionTreeClassifier" ; do
    mkdir "$clf"
    ./grid_search.py --data "$data" --features "$features" --clf "$clf" --directory .
done