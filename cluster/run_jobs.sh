#!/bin/bash

###################
# Config
###################
DIR="KFold"

for ENTRY in ${DIR}/* ; do
    if [[ ! -f ${ENTRY}/score.csv && -f ${ENTRY}/job.sh ]] ; then
        qsub -q all.q@compute-1* ${ENTRY}/job.sh 2>&1 || exit 1
    fi
done
