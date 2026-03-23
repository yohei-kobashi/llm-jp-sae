#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

source ~/start_gpu_nodes.sh
cd llm-jp-sae 

# bash run_lingualens_pipeline.sh --target will,can,could,may,might,must,should,would --model-path Qwen/Qwen2.5-1.5B  --sae-path-template qwen2.5_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/Qwen2.5-1.5B
# bash run_lingualens_pipeline.sh --target suppose --model-path Qwen/Qwen2.5-1.5B  --sae-path-template qwen2.5_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/Qwen2.5-1.5B --input-jsonl data/minimal_pairs_acceptability_2.jsonl
bash run_lingualens_pipeline.sh --target know --model-path Qwen/Qwen2.5-1.5B  --sae-path-template qwen2.5_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/Qwen2.5-1.5B --input-jsonl data/minimal_pairs_acceptability_3.jsonl
