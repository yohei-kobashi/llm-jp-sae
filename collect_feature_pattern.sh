#!/bin/bash -l

#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -W group_list=gj26
#PBS -j oe

#------- Program execution -------#
module purge
module load python/3.10.16 cuda

cd llm-jp-sae 
source env_g/bin/activate
 collect_feature_pattern.py --layers 0 --top_n 10 --final_format compact --finalize_workers 12
python collect_feature_pattern.py --layers 0 1 2
python collect_feature_pattern.py --layers 3 4 5 6
python collect_feature_pattern.py --layers 7 8 9 10
python collect_feature_pattern.py --layers 11 12 13 14
python collect_feature_pattern.py --layers 15 16 17 18
python collect_feature_pattern.py --layers 19 20 21
python collect_feature_pattern.py --layers 22 23 24
python collect_feature_pattern.py --layers 0 1 2 3 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
python collect_feature_pattern.py --layers 4 5 6 7 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
python collect_feature_pattern.py --layers 8 9 10 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
python collect_feature_pattern.py --layers 11 12 13 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
python collect_feature_pattern.py --layers 14 15 16 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
# python collect_feature_pattern.py --layers 1 2 3 4 --label both_
# python collect_feature_pattern.py --layers 5 6 7 8 --label both_
# python collect_feature_pattern.py --layers 9 10 11 12 --label both_
# python collect_feature_pattern.py --layers 13 14 15 16 --label both_
# python collect_feature_pattern.py --layers 17 18 19 20 --label both_
# python collect_feature_pattern.py --layers 21 22 23 24 --label both_