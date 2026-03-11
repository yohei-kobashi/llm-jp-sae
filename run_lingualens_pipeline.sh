#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_lingualens_pipeline.sh \
    --target TARGET \
    --model-path MODEL_PATH \
    --sae-path-template SAE_PATH_TEMPLATE \
    [--layers 0,1,2] \
    [--input-jsonl data/minimal_pairs_acceptability.jsonl] \
    [--output-root outputs/lingualens] \
    [--test-ratio 0.1] \
    [--seed 42] \
    [--top-k 10] \
    [--k 32] \
    [--normalization Scalar] \
    [--batch-size 8] \
    [--torch-dtype bfloat16] \
    [--device cuda:0] \
    [--num-generations 5] \
    [--max-new-tokens 100] \
    [--temperature 0.7]

Required arguments:
  --target               Target modality for convert_jsonl_to_lingualens_text.py
  --model-path           Base model path or Hugging Face model name
  --sae-path-template    SAE template such as sae/.../sae_layer{}.pth

Optional arguments:
  --layers               Comma-separated layer indices. If omitted, auto-detected
                         from files matching SAE_PATH_TEMPLATE.
EOF
}

# ["will", "can", "could", "may", "might", "must", "shall", "should", "would", "ought to", "suppose"]
TARGET="will"
# llm-jp/llm-jp-3-1.8b, allenai/OLMo-2-0425-1B, meta-llama/Llama-3.2-1B, Qwen/Qwen2.5-1.5B
MODEL_PATH="llm-jp/llm-jp-3-1.8b"
# sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth
SAE_PATH_TEMPLATE="sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth"
LAYERS="4,5,6,7,8,9"
INPUT_JSONL="data/minimal_pairs_acceptability.jsonl"
OUTPUT_ROOT="outputs/lingualens"
TEST_RATIO="0.1"
SEED="42"
TOP_K="10"
K="32"
NORMALIZATION="Scalar"
BATCH_SIZE="16"
TORCH_DTYPE="bfloat16"
DEVICE=""
NUM_GENERATIONS="5"
MAX_NEW_TOKENS="100"
TEMPERATURE="0.7"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --sae-path-template)
      SAE_PATH_TEMPLATE="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --input-jsonl)
      INPUT_JSONL="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --test-ratio)
      TEST_RATIO="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --normalization)
      NORMALIZATION="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --torch-dtype)
      TORCH_DTYPE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num-generations)
      NUM_GENERATIONS="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$TARGET" || -z "$MODEL_PATH" || -z "$SAE_PATH_TEMPLATE" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

TARGET_SLUG="${TARGET// /_}"
TARGET_DIR="${OUTPUT_ROOT%/}/${TARGET_SLUG}"
DATA_DIR="$TARGET_DIR/data"
CROSSLAYER_DIR="$TARGET_DIR/crosslayer"
INTERVENTION_DIR="$TARGET_DIR/interventions"
mkdir -p "$DATA_DIR" "$CROSSLAYER_DIR" "$INTERVENTION_DIR"

TRAIN_TXT="$DATA_DIR/${TARGET_SLUG}_train.txt"
TEST_TXT="$DATA_DIR/${TARGET_SLUG}_test.txt"
TEST_JSONL="$DATA_DIR/${TARGET_SLUG}_test.jsonl"
CROSSLAYER_JSON="$CROSSLAYER_DIR/$(basename "${TRAIN_TXT%.txt}")_evolution.json"
LAYER_CANDIDATES_JSON="$CROSSLAYER_DIR/$(basename "${TRAIN_TXT%.txt}")_layer_candidates.json"
SUMMARY_JSON="$INTERVENTION_DIR/summary.json"

DEVICE_ARGS=()
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARGS+=(--device "$DEVICE")
fi

echo "[1/4] Generating train/test data for target: $TARGET"
python convert_jsonl_to_lingualens_text.py \
  --input "$INPUT_JSONL" \
  --output "$TRAIN_TXT" \
  --test-output "$TEST_TXT" \
  --test-output-jsonl "$TEST_JSONL" \
  --test-ratio "$TEST_RATIO" \
  --seed "$SEED" \
  --target "$TARGET"

if [[ -z "$LAYERS" ]]; then
  echo "[2/4] Detecting layers from SAE template"
  TEMPLATE_GLOB="${SAE_PATH_TEMPLATE//\{\}/*}"
  shopt -s nullglob
  MATCHED_PATHS=($TEMPLATE_GLOB)
  shopt -u nullglob

  if [[ ${#MATCHED_PATHS[@]} -eq 0 ]]; then
    echo "Could not auto-detect layers from template: $SAE_PATH_TEMPLATE" >&2
    echo "Pass --layers explicitly." >&2
    exit 1
  fi

  DETECTED_LAYERS=()
  for path in "${MATCHED_PATHS[@]}"; do
    filename="$(basename "$path")"
    if [[ "$filename" =~ layer([0-9]+) ]]; then
      DETECTED_LAYERS+=("${BASH_REMATCH[1]}")
    fi
  done

  if [[ ${#DETECTED_LAYERS[@]} -eq 0 ]]; then
    echo "No layer indices found in filenames matching: $SAE_PATH_TEMPLATE" >&2
    exit 1
  fi

  IFS=$'\n' read -r -d '' -a SORTED_LAYERS < <(printf '%s\n' "${DETECTED_LAYERS[@]}" | sort -n -u && printf '\0')
  LAYERS="$(IFS=,; echo "${SORTED_LAYERS[*]}")"
fi

echo "[3/4] Running cross-layer analysis on layers: $LAYERS"
python crosslayer_lingualens.py \
  --model-path "$MODEL_PATH" \
  --sae-path-template "$SAE_PATH_TEMPLATE" \
  --layers "$LAYERS" \
  --feature-file "$TRAIN_TXT" \
  --output-dir "$CROSSLAYER_DIR" \
  --top-k "$TOP_K" \
  --k "$K" \
  --normalization "$NORMALIZATION" \
  --batch-size "$BATCH_SIZE" \
  --torch-dtype "$TORCH_DTYPE" \
  --export-layer-candidates \
  --layer-candidates-output "$LAYER_CANDIDATES_JSON" \
  "${DEVICE_ARGS[@]}"

if [[ ! -f "$CROSSLAYER_JSON" ]]; then
  echo "Cross-layer output not found: $CROSSLAYER_JSON" >&2
  exit 1
fi

echo "[4/4] Running layer-wise interventions with test prompts"
python intervener_lingualens.py \
  --model-path "$MODEL_PATH" \
  --sae-path-template "$SAE_PATH_TEMPLATE" \
  --crosslayer-json "$CROSSLAYER_JSON" \
  --output-dir "$INTERVENTION_DIR" \
  --selection-mode per-layer \
  --resume \
  --prompt-file "$TEST_TXT" \
  --k "$K" \
  --normalization "$NORMALIZATION" \
  --torch-dtype "$TORCH_DTYPE" \
  --num-generations "$NUM_GENERATIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  "${DEVICE_ARGS[@]}"

echo "Finished."
echo "Train data: $TRAIN_TXT"
echo "Test data: $TEST_TXT"
echo "Cross-layer result: $CROSSLAYER_JSON"
echo "Layer candidates: $LAYER_CANDIDATES_JSON"
echo "Intervention summary: $SUMMARY_JSON"
