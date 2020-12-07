#!/bin/bash

DATA="scores_data_KFold.csv"
echo "test_accuracy,test_precision,test_recall,test_f1,test_f2,test_geometric_mean,test_roc_auc,activity_label,model,NumRotatableBonds,NumHAcceptors,NumHDonors,TPSA,LabuteASA,MolLogP,qvina,rfscore_qvina,plants,rfscore_plants" > ${DATA}

DIR="KFold"

for ENTRY in ${DIR}/* ; do
    if [[ -f ${ENTRY}/score.csv ]] ; then
        cat ${ENTRY}/score.csv
    fi
done >> ${DATA}
