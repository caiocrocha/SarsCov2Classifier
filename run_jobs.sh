#!/bin/bash

###################
# Config
###################
project="MODELS"

for entry in ${project}/* ; do
    if [[ ! -f ${entry}/score.csv && -f ${entry}/job.sh ]] ; then
        qsub -q all.q@compute-1* ${entry}/job.sh || exit 1
    else
        exit 1
    fi
done
