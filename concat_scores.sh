#!/bin/bash

DATA="scores_data_cluster.csv"
echo "test_accuracy,test_precision,test_recall,test_f1,test_f2,test_geometric_mean,test_roc_auc,activity_label,model,random_state,MolMR,NumRotatableBonds,NumHAcceptors,NumHDonors,TPSA,LabuteASA,MolLogP,qvina,rfscore_qvina,plants,rfscore_plants,test_mean" > ${DATA}

DIR="MODELS"

for entry in ${DIR}/* ; do
    if [[ -f ${entry}/score.csv ]] ; then
        cat ${entry}/score.csv >> ${DATA}
    else
        echo "No file \"score.csv\" in ${entry}"
    fi
done
