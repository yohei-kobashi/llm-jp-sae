#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -W group_list=gj26
#PBS -j oe

source ~/start_gpu_nodes.sh
cd llm-jp-sae 

# python collect_dev_activations_all_layers.py --model-path llm-jp/llm-jp-3-1.8b --input-dir outputs/llm-jp-3-1.8b --sae-path-template sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth
python collect_dev_activations_all_layers.py --model-path allenai/OLMo-2-0425-1B --input-dir outputs/OLMo-2-0425-1B --sae-path-template olmo2_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth
python collect_dev_activations_all_layers.py --model-path meta-llama/Llama-3.2-1B --input-dir outputs/Llama-3.2-1B --sae-path-template llama3.2_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth
# python collect_dev_activations_all_layers.py --model-path Qwen/Qwen2.5-1.5B --input-dir outputs/Qwen2.5-1.5B --sae-path-template qwen2.5_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth