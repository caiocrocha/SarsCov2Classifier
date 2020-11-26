#!/bin/bash

###################
# Config
###################
DIR="MODELS"

for entry in ${DIR}/* ; do
    if [[ ! -f ${entry}/score.csv && -f ${entry}/job.sh ]] ; then
        qsub -q all.q@compute-1* ${entry}/job.sh 2>&1 || exit 1
    else
        exit 1
    fi
done
