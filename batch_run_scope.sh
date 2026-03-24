#!/bin/bash -l

#------ qsub option --------#
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -W group_list=gj26
#PBS -j oe

set -euo pipefail

source ~/start_gpu_nodes.sh
cd ~/llm-jp-sae

OUTPUT_BASE="outputs"

bash run_scope_lingualens_pipelines.sh \
  --scope gemma,llama \
  --gemma-model-path google/gemma-3-1b-pt \
  --gemma-sae-path-template 'gemma-scope-2-1b-pt-res-all:layer_{}_width_16k_l0_small' \
  --llama-model-path Qwen/Qwen3-1.7B \
  --llama-sae-path-template 'OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B:transcoder/8x/k128/layer{}_transcoder_8x_k128' \
  --output-root "${OUTPUT_BASE}" \
  --batch-size 64 \
  --torch-dtype bfloat16