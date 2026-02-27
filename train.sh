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
# python train.py --layers 0
# python train.py --layers 15 16 17 18
# python train.py --layers 19 20 21 22
# python train.py --layers 23 24
python train.py --layers 0 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
# python train.py --layers 1 2 3 4 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
# python train.py --layers 5 6 7 8 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
# python train.py --layers 9 10 11 12 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
# python train.py --layers 13 14 15 16 --label olmo2_ --model_name_or_dir allenai/OLMo-2-0425-1B
# python train.py --layers 1 2 3 4 --label both_
# python train.py --layers 5 6 7 8 --label both_
# python train.py --layers 9 10 11 12 --label both_
# python train.py --layers 13 14 15 16 --label both_
# python train.py --layers 17 18 19 20 --label both_
# python train.py --layers 21 22 23 24 --label both_
python train.py --layers 0 1 2 --label llama3.2_ --model_name_or_dir meta-llama/Llama-3.2-1B
python train.py --layers 3 4 5 --label llama3.2_ --model_name_or_dir meta-llama/Llama-3.2-1B
python train.py --layers 6 7 8 --label llama3.2_ --model_name_or_dir meta-llama/Llama-3.2-1B
python train.py --layers 9 10 11 --label llama3.2_ --model_name_or_dir meta-llama/Llama-3.2-1B
python train.py --layers 12 13 14 --label llama3.2_ --model_name_or_dir meta-llama/Llama-3.2-1B
python train.py --layers 15 16 --label llama3.2_ --model_name_or_dir meta-llama/Llama-3.2-1B
python train.py --layers 0 1 2 --label qwen2.5_ --model_name_or_dir Qwen/Qwen2.5-1.5B
python train.py --layers 3 4 5 --label qwen2.5_ --model_name_or_dir Qwen/Qwen2.5-1.5B
python train.py --layers 6 7 8 --label qwen2.5_ --model_name_or_dir Qwen/Qwen2.5-1.5B
python train.py --layers 9 10 11 --label qwen2.5_ --model_name_or_dir Qwen/Qwen2.5-1.5B
python train.py --layers 12 13 14 --label qwen2.5_ --model_name_or_dir Qwen/Qwen2.5-1.5B
python train.py --layers 15 16 --label qwen2.5_ --model_name_or_dir Qwen/Qwen2.5-1.5B
