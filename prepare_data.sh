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
# python prepare_data.py
# python prepare_data.py --label ja_ --dolma_sample_rate 0.0 --warp_sample_rate 1.0
# python prepare_data.py --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B 
python prepare_data.py --label llama3.2_ --model_name_or_dir meta-llama/Llama-3.2-1B 
# python prepare_data.py --label qwen2.5_ --model_name_or_dir Qwen/Qwen2.5-1.5B 
python count_tokens.py --data-dir data/tokenized
