#!/bin/bash -l

#------ qsub option --------#
#PBS -q regular-c
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -W group_list=gj26
#PBS -j oe

#------- Program execution -------#
module purge
module load cmake
module load gcc

cd llm-jp-sae 
source env/bin/activate
python prepare_data.py
python count_tokens.py --data-dir data/tokenized --show-all
