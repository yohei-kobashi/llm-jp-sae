#!/bin/bash -l

#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=6:00:00
#PBS -W group_list=gj26
#PBS -j oe

source ~/start_gpu_nodes.sh
cd llm-jp-sae 

bash run_lingualens_pipeline.sh --target will --model-path llm-jp/llm-jp-3-1.8 
bash run_lingualens_pipeline.sh --target will --model-path ballenai/OLMo-2-0425-1B --sae-path-template olmo2_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth
# bash run_lingualens_pipeline.sh --target will,can,could,may,might,must,shall,should,would,ought to,suppose --model-path allenai/OLMo-2-0425-1B 