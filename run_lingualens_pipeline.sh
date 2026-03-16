#!/bin/bash -l

#------- Program execution -------#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_lingualens_pipeline.sh \
    --target TARGET[,TARGET2,...] \
    --model-path MODEL_PATH \
    --sae-path-template SAE_PATH_TEMPLATE \
    [--layers 0,1,2] \
    [--input-jsonl data/minimal_pairs_acceptability.jsonl] \
    [--output-root outputs/lingualens] \
    [--start-step 1] \
    [--dev-ratio 0.2] \
    [--test-ratio 0.2] \
    [--seed 42] \
    [--top-k 10] \
    [--k 32] \
    [--normalization Scalar] \
    [--batch-size 8] \
    [--torch-dtype bfloat16] \
    [--device cuda:0] \
    [--random-seed 42] \
    [--num-generations 5] \
    [--max-new-tokens 100] \
    [--temperature 0.7]

Required arguments:
  --target               Target modality for convert_jsonl_to_lingualens_text.py.
                         Can be passed multiple times and/or as a comma-separated list.
  --model-path           Base model path or Hugging Face model name
  --sae-path-template    SAE template such as sae/.../sae_layer{}.pth

Optional arguments:
  --layers               Comma-separated layer indices. If omitted, auto-detected
                         from files matching SAE_PATH_TEMPLATE.
  --start-step           Step number to start from (1-4).
EOF
}

# ["will", "can", "could", "may", "might", "must", "shall", "should", "would", "ought to", "suppose"]
TARGETS=("will")
# llm-jp/llm-jp-3-1.8b, allenai/OLMo-2-0425-1B, meta-llama/Llama-3.2-1B, Qwen/Qwen2.5-1.5B
MODEL_PATH="llm-jp/llm-jp-3-1.8b"
# sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth
SAE_PATH_TEMPLATE="sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth"
LAYERS=""
INPUT_JSONL="data/minimal_pairs_acceptability.jsonl"
OUTPUT_ROOT="outputs/llm-jp-3-1.8b"
START_STEP="1"
DEV_RATIO="0.2"
TEST_RATIO="0.2"
SEED="42"
TOP_K="10"
K="32"
NORMALIZATION="Scalar"
BATCH_SIZE="64"
TORCH_DTYPE="bfloat16"
DEVICE=""
RANDOM_SEED=""
NUM_GENERATIONS="5"
MAX_NEW_TOKENS="100"
TEMPERATURE="0.7"

append_targets() {
  local raw="$1"
  local part=""

  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="${part#"${part%%[![:space:]]*}"}"
    part="${part%"${part##*[![:space:]]}"}"
    if [[ -n "$part" ]]; then
      TARGETS+=("$part")
    fi
  done
}

resolve_layers() {
  if [[ -n "$LAYERS" ]]; then
    ALL_LAYERS="$LAYERS"
    return
  fi

  if (( START_STEP <= 2 )); then
    echo "[2/4] Detecting layers from SAE template"
  else
    echo "[2/4] Skipped layer detection (--start-step=$START_STEP)"
    echo "LAYERS is not specified, so auto-detecting layers from SAE template for downstream steps"
  fi

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
  ALL_LAYERS="$(IFS=,; echo "${SORTED_LAYERS[*]}")"
}

run_for_target() {
  local target="$1"
  local target_slug target_dir data_dir crosslayer_dir intervention_dir
  local train_txt dev_txt dev_jsonl test_txt test_jsonl feature_name
  local layer layer_json
  local -a layer_array=()
  local -a missing_layer_jsons=()

  target_slug="${target// /_}"
  target_dir="${OUTPUT_ROOT%/}/${target_slug}"
  data_dir="$target_dir/data"
  crosslayer_dir="$target_dir/crosslayer"
  intervention_dir="$target_dir/interventions"
  mkdir -p "$data_dir" "$crosslayer_dir" "$intervention_dir"

  train_txt="$data_dir/${target_slug}_train.txt"
  dev_txt="$data_dir/${target_slug}_dev.txt"
  dev_jsonl="$data_dir/${target_slug}_dev.jsonl"
  test_txt="$data_dir/${target_slug}_test.txt"
  test_jsonl="$data_dir/${target_slug}_test.jsonl"

  echo "============================================================"
  echo "Processing target: $target"
  echo "Output directory: $target_dir"

  if (( START_STEP <= 1 )); then
    echo "[1/4] Generating train/dev/test data for target: $target"
    python convert_jsonl_to_lingualens_text.py \
      --input "$INPUT_JSONL" \
      --output "$train_txt" \
      --dev-output "$dev_txt" \
      --dev-output-jsonl "$dev_jsonl" \
      --test-output "$test_txt" \
      --test-output-jsonl "$test_jsonl" \
      --dev-ratio "$DEV_RATIO" \
      --test-ratio "$TEST_RATIO" \
      --seed "$SEED" \
      --target "$target"
  else
    echo "[1/4] Skipped data generation (--start-step=$START_STEP)"
    if [[ ! -f "$train_txt" || ! -f "$dev_txt" || ! -f "$dev_jsonl" || ! -f "$test_txt" || ! -f "$test_jsonl" ]]; then
      echo "Missing generated data required to skip step 1." >&2
      echo "Expected files:" >&2
      echo "  $train_txt" >&2
      echo "  $dev_txt" >&2
      echo "  $dev_jsonl" >&2
      echo "  $test_txt" >&2
      echo "  $test_jsonl" >&2
      exit 1
    fi
  fi

  if (( START_STEP <= 3 )); then
    if [[ ! -f "$train_txt" ]]; then
      echo "Missing train data required for step 3: $train_txt" >&2
      exit 1
    fi
    echo "[3/4] Running cross-layer analysis on layers: $ALL_LAYERS"
    python crosslayer_lingualens.py \
      --model-path "$MODEL_PATH" \
      --sae-path-template "$SAE_PATH_TEMPLATE" \
      --layers "$ALL_LAYERS" \
      --feature-file "$train_txt" \
      --output-dir "$crosslayer_dir" \
      --top-k "$TOP_K" \
      --k "$K" \
      --normalization "$NORMALIZATION" \
      --batch-size "$BATCH_SIZE" \
      --torch-dtype "$TORCH_DTYPE" \
      --resume \
      "${DEVICE_ARGS[@]}"
  else
    echo "[3/4] Skipped cross-layer analysis (--start-step=$START_STEP)"
  fi

  feature_name="$(basename "${train_txt%.txt}")"
  IFS=',' read -r -a layer_array <<< "$ALL_LAYERS"
  missing_layer_jsons=()
  for layer in "${layer_array[@]}"; do
    layer_json="$crosslayer_dir/${feature_name}_layer${layer}_evolution.json"
    if [[ ! -f "$layer_json" ]]; then
      missing_layer_jsons+=("$layer_json")
    fi
  done

  if [[ ${#missing_layer_jsons[@]} -gt 0 ]]; then
    echo "Missing per-layer crosslayer outputs required for step 4:" >&2
    for path in "${missing_layer_jsons[@]}"; do
      echo "  $path" >&2
    done
    exit 1
  fi

  if [[ ! -f "$test_txt" ]]; then
    echo "Missing test data required for step 4: $test_txt" >&2
    exit 1
  fi

  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for alpha tuning and LLM-as-a-Judge evaluation in step 4." >&2
    exit 1
  fi

  echo "[4/4] Running layer-wise interventions with test prompts"
  python intervener_lingualens.py \
    --model-path "$MODEL_PATH" \
    --sae-path-template "$SAE_PATH_TEMPLATE" \
    --crosslayer-json "$crosslayer_dir" \
    --output-dir "$intervention_dir" \
    --selection-mode per-layer \
    --resume \
    --prompt-file "$test_txt" \
    --dev-prompt-file "$dev_txt" \
    --test-prompt-file "$test_txt" \
    --target-modality "$target" \
    --k "$K" \
    --normalization "$NORMALIZATION" \
    --torch-dtype "$TORCH_DTYPE" \
    --batch-size "$BATCH_SIZE" \
    --random-seed "$RANDOM_SEED" \
    --num-generations "$NUM_GENERATIONS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    "${DEVICE_ARGS[@]}"

  echo "Finished target: $target"
  echo "Train data: $train_txt"
  echo "Dev data: $dev_txt"
  echo "Test data: $test_txt"
  echo "Cross-layer directory: $crosslayer_dir"
  echo "Intervention directory: $intervention_dir"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      append_targets "$2"
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
    --start-step)
      START_STEP="$2"
      shift 2
      ;;
    --dev-ratio)
      DEV_RATIO="$2"
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
    --random-seed)
      RANDOM_SEED="$2"
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

if [[ -z "$MODEL_PATH" || -z "$SAE_PATH_TEMPLATE" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

if [[ ! "$START_STEP" =~ ^[1-4]$ ]]; then
  echo "--start-step must be one of: 1, 2, 3, 4" >&2
  exit 1
fi

if [[ -z "$RANDOM_SEED" ]]; then
  RANDOM_SEED="$SEED"
fi

if [[ ${#TARGETS[@]} -gt 1 ]]; then
  TARGETS=("${TARGETS[@]:1}")
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

DEVICE_ARGS=()
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARGS+=(--device "$DEVICE")
fi

resolve_layers

for target in "${TARGETS[@]}"; do
  run_for_target "$target"
done
