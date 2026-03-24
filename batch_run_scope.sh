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
GEMMA_MODEL_PATH="google/gemma-3-1b-pt"
GEMMA_SAE_PATH_TEMPLATE='gemma-scope-2-1b-pt-res-all:layer_{}_width_16k_l0_small'
LLAMA_MODEL_PATH="Qwen/Qwen3-1.7B"
LLAMA_SAE_PATH_TEMPLATE='OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B:transcoder/8x/k128/layer{}_transcoder_8x_k128'

run_scope_group() {
  local targets="$1"
  local input_jsonl="$2"

  bash run_scope_lingualens_pipelines.sh \
    --scope gemma,llama \
    --target "$targets" \
    --input-jsonl "$input_jsonl" \
    --gemma-model-path "$GEMMA_MODEL_PATH" \
    --gemma-sae-path-template "$GEMMA_SAE_PATH_TEMPLATE" \
    --llama-model-path "$LLAMA_MODEL_PATH" \
    --llama-sae-path-template "$LLAMA_SAE_PATH_TEMPLATE" \
    --output-root "${OUTPUT_BASE}" \
    --batch-size 64 \
    --torch-dtype bfloat16
}

run_scope_group "will,can,could,may,might,must,should,would" "data/minimal_pairs_acceptability.jsonl"
run_scope_group "suppose" "data/minimal_pairs_acceptability_2.jsonl"
run_scope_group "know" "data/minimal_pairs_acceptability_3.jsonl"
