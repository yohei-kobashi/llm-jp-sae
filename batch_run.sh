#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

source ~/start_gpu_nodes.sh
cd llm-jp-sae 

# bash run_lingualens_pipeline.sh --target will,can,could,may,might,must,should,would --model-path llm-jp/llm-jp-3-1.8b
# bash run_lingualens_pipeline.sh --target suppose --model-path llm-jp/llm-jp-3-1.8b --input-jsonl data/minimal_pairs_acceptability_2.jsonl
# bash run_lingualens_pipeline.sh --target will,can,could,may,might,must,should,would --model-path allenai/OLMo-2-0425-1B --sae-path-template olmo2_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/OLMo-2-0425-1B
bash run_lingualens_pipeline.sh --target suppose --model-path allenai/OLMo-2-0425-1B --sae-path-template olmo2_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/OLMo-2-0425-1B --input-jsonl data/minimal_pairs_acceptability_2.jsonl
# bash run_lingualens_pipeline.sh --target will,can,could,may,might,must,should,would --model-path meta-llama/Llama-3.2-1B  --sae-path-template llama3.2_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/Llama-3.2-1B
bash run_lingualens_pipeline.sh --target suppose --model-path meta-llama/Llama-3.2-1B  --sae-path-template llama3.2_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/Llama-3.2-1B --input-jsonl data/minimal_pairs_acceptability_2.jsonl
# bash run_lingualens_pipeline.sh --target will,can,could,may,might,must,should,would --model-path Qwen/Qwen2.5-1.5B  --sae-path-template qwen2.5_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/Qwen2.5-1.5B
# bash run_lingualens_pipeline.sh --target suppose --model-path Qwen/Qwen2.5-1.5B  --sae-path-template qwen2.5_sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth --output-root outputs/Qwen2.5-1.5B --input-jsonl data/minimal_pairs_acceptability_2.jsonl
