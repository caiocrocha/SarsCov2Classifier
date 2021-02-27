#!/bin/bash

###################
# Config
###################
directory="$1"

for clf in ${directory}/* ; do
    if [[ -f ${clf}/job.sh ]] ; then
        qsub -q all.q@compute-1* ${clf}/job.sh 2>&1 || exit 1
    fi
done
