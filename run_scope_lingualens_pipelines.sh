#!/bin/bash -l

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_scope_lingualens_pipelines.sh \
    --scope gemma,llama \
    --gemma-model-path MODEL \
    --gemma-sae-path-template TEMPLATE \
    --llama-model-path MODEL \
    --llama-sae-path-template TEMPLATE \
    [--target will,can,...] \
    [--layers 0,1,2] \
    [--input-jsonl data/minimal_pairs_acceptability.jsonl] \
    [--output-root outputs/scope-lingualens] \
    [--start-step 1] \
    [--dev-ratio 0.2] \
    [--test-ratio 0.2] \
    [--seed 42] \
    [--top-k 10] \
    [--batch-size 8] \
    [--torch-dtype bfloat16] \
    [--device cuda:0] \
    [--prepend-bos] \
    [--fold-activation-scale]

Required arguments:
  --scope                      gemma, llama, or both as comma-separated values
  --gemma-model-path           Required when gemma scope is selected
  --gemma-sae-path-template    Required when gemma scope is selected
  --llama-model-path           Required when llama scope is selected
  --llama-sae-path-template    Required when llama scope is selected

Optional arguments:
  --target                     Can be passed multiple times and/or comma-separated.
                               Default: all supported targets.
  --layers                     Comma-separated layer indices. If omitted, defaults are
                               llama=0-27 and gemma=0-25.
  --start-step                 Step number to start from (1-3).
EOF
}

ALL_TARGETS=("will" "can" "could" "may" "might" "must" "should" "would" "suppose" "know")
TARGETS=()
SCOPES=()

GEMMA_MODEL_PATH=""
GEMMA_SAE_PATH_TEMPLATE=""
LLAMA_MODEL_PATH=""
LLAMA_SAE_PATH_TEMPLATE=""

LAYERS=""
INPUT_JSONL="data/minimal_pairs_acceptability.jsonl"
OUTPUT_ROOT="outputs/scope-lingualens"
START_STEP="1"
DEV_RATIO="0.2"
TEST_RATIO="0.2"
SEED="42"
TOP_K="10"
BATCH_SIZE="64"
TORCH_DTYPE="bfloat16"
DEVICE=""
PREPEND_BOS="1"
FOLD_ACTIVATION_SCALE="0"
DEFAULT_GEMMA_LAYERS="$(seq -s, 0 25)"
DEFAULT_LLAMA_LAYERS="$(seq -s, 0 27)"

append_list_values() {
  local -n target_array="$1"
  local raw="$2"
  local part=""

  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="${part#"${part%%[![:space:]]*}"}"
    part="${part%"${part##*[![:space:]]}"}"
    if [[ -n "$part" ]]; then
      target_array+=("$part")
    fi
  done
}

require_arg_value() {
  local option="$1"
  local value="${2-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "Missing value for $option" >&2
    usage >&2
    exit 1
  fi
}

resolve_layers_for_scope() {
  local scope="$1"
  if [[ -n "$LAYERS" ]]; then
    echo "$LAYERS"
    return
  fi

  case "$scope" in
    gemma)
      echo "$DEFAULT_GEMMA_LAYERS"
      ;;
    llama)
      echo "$DEFAULT_LLAMA_LAYERS"
      ;;
    *)
      echo "Unsupported scope for layer resolution: $scope" >&2
      exit 1
      ;;
  esac
}

scope_output_subdir() {
  local scope="$1"
  case "$scope" in
    gemma)
      echo "gemma-3-1b-pt"
      ;;
    llama)
      echo "Qwen3-1.7B"
      ;;
    *)
      echo "Unsupported scope for output path: $scope" >&2
      exit 1
      ;;
  esac
}

is_scope_target_completed() {
  local target_slug="$1"
  local data_dir="$2"
  local crosslayer_dir="$3"
  local layers_csv="$4"
  local train_txt dev_txt dev_jsonl test_txt test_jsonl
  local feature_name layer output_path

  train_txt="$data_dir/${target_slug}_train.txt"
  dev_txt="$data_dir/${target_slug}_dev.txt"
  dev_jsonl="$data_dir/${target_slug}_dev.jsonl"
  test_txt="$data_dir/${target_slug}_test.txt"
  test_jsonl="$data_dir/${target_slug}_test.jsonl"

  if [[ ! -f "$train_txt" || ! -f "$dev_txt" || ! -f "$dev_jsonl" || ! -f "$test_txt" || ! -f "$test_jsonl" ]]; then
    return 1
  fi

  feature_name="${target_slug}_train"
  IFS=',' read -r -a layers_array <<< "$layers_csv"
  for layer in "${layers_array[@]}"; do
    output_path="$crosslayer_dir/${feature_name}_layer${layer}_evolution.json"
    if [[ ! -f "$output_path" ]]; then
      return 1
    fi
  done

  return 0
}

build_common_crosslayer_args() {
  COMMON_CROSSLAYER_ARGS=(
    --layers "$1"
    --feature-file "$2"
    --output-dir "$3"
    --top-k "$TOP_K"
    --batch-size "$BATCH_SIZE"
    --torch-dtype "$TORCH_DTYPE"
    --resume
  )
  if [[ "$PREPEND_BOS" == "1" ]]; then
    COMMON_CROSSLAYER_ARGS+=(--prepend-bos)
  else
    COMMON_CROSSLAYER_ARGS+=(--no-prepend-bos)
  fi
  if [[ "$FOLD_ACTIVATION_SCALE" == "1" ]]; then
    COMMON_CROSSLAYER_ARGS+=(--fold-activation-scale)
  fi
  if [[ -n "$DEVICE" ]]; then
    COMMON_CROSSLAYER_ARGS+=(--device "$DEVICE")
  fi
}

run_crosslayer_for_scope() {
  local scope="$1"
  local target_slug="$2"
  local train_txt="$3"

  local scope_dir script_path model_path sae_path_template layers_csv crosslayer_dir scope_subdir
  case "$scope" in
    gemma)
      script_path="crosslayer_gemmascope_lingualens.py"
      model_path="$GEMMA_MODEL_PATH"
      sae_path_template="$GEMMA_SAE_PATH_TEMPLATE"
      ;;
    llama)
      script_path="crosslayer_llamascope_lingualens.py"
      model_path="$LLAMA_MODEL_PATH"
      sae_path_template="$LLAMA_SAE_PATH_TEMPLATE"
      ;;
    *)
      echo "Unsupported scope: $scope" >&2
      exit 1
      ;;
  esac

  layers_csv="$(resolve_layers_for_scope "$scope")"
  scope_subdir="$(scope_output_subdir "$scope")"
  scope_dir="${OUTPUT_ROOT%/}/${scope_subdir}/${target_slug}"
  crosslayer_dir="$scope_dir/crosslayer"
  mkdir -p "$crosslayer_dir"

  if is_scope_target_completed "$target_slug" "${OUTPUT_ROOT%/}/data/${target_slug}" "$crosslayer_dir" "$layers_csv"; then
    echo "Skipping $scope target: $target_slug"
    echo "Reason: all expected outputs already exist."
    return 0
  fi

  echo "[3/3] Running $scope cross-layer analysis on layers: $layers_csv"
  echo "      output_dir: $crosslayer_dir"
  build_common_crosslayer_args "$layers_csv" "$train_txt" "$crosslayer_dir"
  python "$script_path" \
    --model-path "$model_path" \
    --sae-path-template "$sae_path_template" \
    "${COMMON_CROSSLAYER_ARGS[@]}"
}

run_for_target() {
  local target="$1"
  local target_slug data_dir train_txt dev_txt dev_jsonl test_txt test_jsonl scope

  target_slug="${target// /_}"
  data_dir="${OUTPUT_ROOT%/}/data/${target_slug}"
  mkdir -p "$data_dir"

  train_txt="$data_dir/${target_slug}_train.txt"
  dev_txt="$data_dir/${target_slug}_dev.txt"
  dev_jsonl="$data_dir/${target_slug}_dev.jsonl"
  test_txt="$data_dir/${target_slug}_test.txt"
  test_jsonl="$data_dir/${target_slug}_test.jsonl"

  echo "============================================================"
  echo "Processing target: $target"
  echo "Data directory: $data_dir"

  if (( START_STEP <= 1 )); then
    echo "[1/3] Generating train/dev/test data for target: $target"
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
    echo "[1/3] Skipped data generation (--start-step=$START_STEP)"
    if [[ ! -f "$train_txt" || ! -f "$dev_txt" || ! -f "$dev_jsonl" || ! -f "$test_txt" || ! -f "$test_jsonl" ]]; then
      echo "Missing generated data required to skip step 1." >&2
      exit 1
    fi
  fi

  echo "[2/3] Preparing scope-specific layer configuration"
  for scope in "${SCOPES[@]}"; do
    case "$scope" in
      gemma)
        echo "  gemma layers: $(resolve_layers_for_scope gemma)"
        ;;
      llama)
        echo "  llama layers: $(resolve_layers_for_scope llama)"
        ;;
    esac
  done

  if (( START_STEP <= 3 )); then
    for scope in "${SCOPES[@]}"; do
      run_crosslayer_for_scope "$scope" "$target_slug" "$train_txt"
    done
  else
    echo "[3/3] Skipped cross-layer analysis (--start-step=$START_STEP)"
  fi

  echo "Finished target: $target"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      require_arg_value "$1" "${2-}"
      append_list_values SCOPES "$2"
      shift 2
      ;;
    --scope=*)
      append_list_values SCOPES "${1#*=}"
      shift
      ;;
    --target)
      require_arg_value "$1" "${2-}"
      append_list_values TARGETS "$2"
      shift 2
      ;;
    --target=*)
      append_list_values TARGETS "${1#*=}"
      shift
      ;;
    --gemma-model-path)
      require_arg_value "$1" "${2-}"
      GEMMA_MODEL_PATH="$2"
      shift 2
      ;;
    --gemma-model-path=*)
      GEMMA_MODEL_PATH="${1#*=}"
      shift
      ;;
    --gemma-sae-path-template)
      require_arg_value "$1" "${2-}"
      GEMMA_SAE_PATH_TEMPLATE="$2"
      shift 2
      ;;
    --gemma-sae-path-template=*)
      GEMMA_SAE_PATH_TEMPLATE="${1#*=}"
      shift
      ;;
    --llama-model-path)
      require_arg_value "$1" "${2-}"
      LLAMA_MODEL_PATH="$2"
      shift 2
      ;;
    --llama-model-path=*)
      LLAMA_MODEL_PATH="${1#*=}"
      shift
      ;;
    --llama-sae-path-template)
      require_arg_value "$1" "${2-}"
      LLAMA_SAE_PATH_TEMPLATE="$2"
      shift 2
      ;;
    --llama-sae-path-template=*)
      LLAMA_SAE_PATH_TEMPLATE="${1#*=}"
      shift
      ;;
    --layers)
      require_arg_value "$1" "${2-}"
      LAYERS="$2"
      shift 2
      ;;
    --layers=*)
      LAYERS="${1#*=}"
      shift
      ;;
    --input-jsonl)
      require_arg_value "$1" "${2-}"
      INPUT_JSONL="$2"
      shift 2
      ;;
    --input-jsonl=*)
      INPUT_JSONL="${1#*=}"
      shift
      ;;
    --output-root)
      require_arg_value "$1" "${2-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT="${1#*=}"
      shift
      ;;
    --start-step)
      require_arg_value "$1" "${2-}"
      START_STEP="$2"
      shift 2
      ;;
    --start-step=*)
      START_STEP="${1#*=}"
      shift
      ;;
    --dev-ratio)
      require_arg_value "$1" "${2-}"
      DEV_RATIO="$2"
      shift 2
      ;;
    --dev-ratio=*)
      DEV_RATIO="${1#*=}"
      shift
      ;;
    --test-ratio)
      require_arg_value "$1" "${2-}"
      TEST_RATIO="$2"
      shift 2
      ;;
    --test-ratio=*)
      TEST_RATIO="${1#*=}"
      shift
      ;;
    --seed)
      require_arg_value "$1" "${2-}"
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --top-k)
      require_arg_value "$1" "${2-}"
      TOP_K="$2"
      shift 2
      ;;
    --top-k=*)
      TOP_K="${1#*=}"
      shift
      ;;
    --batch-size)
      require_arg_value "$1" "${2-}"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --torch-dtype)
      require_arg_value "$1" "${2-}"
      TORCH_DTYPE="$2"
      shift 2
      ;;
    --torch-dtype=*)
      TORCH_DTYPE="${1#*=}"
      shift
      ;;
    --device)
      require_arg_value "$1" "${2-}"
      DEVICE="$2"
      shift 2
      ;;
    --device=*)
      DEVICE="${1#*=}"
      shift
      ;;
    --prepend-bos)
      PREPEND_BOS="1"
      shift
      ;;
    --no-prepend-bos)
      PREPEND_BOS="0"
      shift
      ;;
    --fold-activation-scale)
      FOLD_ACTIVATION_SCALE="1"
      shift
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

if [[ ${#SCOPES[@]} -eq 0 ]]; then
  echo "Missing required argument: --scope" >&2
  usage >&2
  exit 1
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=("${ALL_TARGETS[@]}")
fi

if [[ ! "$START_STEP" =~ ^[1-3]$ ]]; then
  echo "--start-step must be one of: 1, 2, 3" >&2
  exit 1
fi

for scope in "${SCOPES[@]}"; do
  case "$scope" in
    gemma)
      if [[ -z "$GEMMA_MODEL_PATH" || -z "$GEMMA_SAE_PATH_TEMPLATE" ]]; then
        echo "gemma scope requires --gemma-model-path and --gemma-sae-path-template" >&2
        exit 1
      fi
      ;;
    llama)
      if [[ -z "$LLAMA_MODEL_PATH" || -z "$LLAMA_SAE_PATH_TEMPLATE" ]]; then
        echo "llama scope requires --llama-model-path and --llama-sae-path-template" >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported scope: $scope" >&2
      exit 1
      ;;
  esac
done

echo "[Pipeline] scopes: $(IFS=,; echo "${SCOPES[*]}")"
echo "[Pipeline] targets: $(IFS=,; echo "${TARGETS[*]}")"
echo "[Pipeline] output_root: $OUTPUT_ROOT"

for target in "${TARGETS[@]}"; do
  run_for_target "$target"
done
