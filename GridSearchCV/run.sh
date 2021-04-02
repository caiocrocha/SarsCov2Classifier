#!/bin/sh

data="../notebooks/training_set.csv.gz"
features="../notebooks/features.lst"

for clf in "KNeighborsClassifier" "DecisionTreeClassifier" "RandomForestClassifier" "XGBClassifier" ; do
    mkdir "$clf"
    ./grid_search.py --data "$data" --features "$features" --clf "$clf" --directory .
done
