"""
Export SAE activations for either:

1) modality CSVs with `Marked_Sentence_English` (markers like `*` removed), or
2) generic CSVs with a raw sentence column (no preprocessing).

Usage examples:

  - Modality CSV (default mode):

        python3 export_sae_activations.py \
            --layers 12 13 --n_d 16 --k 32 --nl Scalar \
            --ckpt 988240 --lr 1e-3 \
            --label experiment1 \
            --input-csv data/qp_modality_list_v1.csv \
            --output-dir data

    This removes `*` markers from `Marked_Sentence_English`, extracts SAE
    activations, and writes Parquet files. It also adds
    `Marked_Sentence_English_Unmarked` to outputs.

  - Generic CSV (no modality markers, custom sentence column):

        python3 export_sae_activations.py \
            --no-modality \
            --sentence-column sentence \
            --input-csv data/simple.csv \
            --layers 12 13 \
            --output-dir outputs

    This reads the `sentence` column as-is (no preprocessing) and exports SAE
    activations. No modality-specific columns are added.

The script loads the language model and trained sparse autoencoders, obtains
latent activations for each requested layer, and exports the original CSV
columns together with SAE features in layer-specific Parquet files. Pass
`--label` when the SAE was trained with a label prefix so the script looks in
the matching checkpoint directory and optionally appends the label to output
filenames.
"""

import argparse
import os
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SaeConfig, UsrConfig, return_save_dir
from model import SimpleHook, SparseAutoEncoder, normalize_activation

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - pandas is required at runtime
    raise RuntimeError(
        "pandas is required to run this script. Install pandas before executing."
    ) from exc

try:
    import pyarrow  # noqa: F401  # Needed for pandas.to_parquet with default engine
except Exception:
    pyarrow = None  # pragma: no cover - optional, pandas will raise if missing


if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        MODEL_DEVICE = torch.device("cuda:0")
        SAE_DEVICE = torch.device("cuda:1")
    else:
        MODEL_DEVICE = torch.device("cuda")
        SAE_DEVICE = torch.device("cuda")
else:
    MODEL_DEVICE = torch.device("cpu")
    SAE_DEVICE = torch.device("cpu")


def _remove_markers(text: str) -> str:
    """Clear modal markers from a sentence before tokenisation.

    Args:
        text: Sentence that may contain asterisks around modal verbs.

    Returns:
        The sentence with all asterisk characters removed.
    """
    return text.replace("*", "")


def _resolve_save_dir(args: argparse.Namespace, usr_cfg: UsrConfig) -> str:
    """Reconstruct the SAE checkpoint directory based on training hyperparameters.

    Args:
        args: Parsed CLI arguments that mirror the training configuration.
        usr_cfg: User configuration providing default save roots.

    Returns:
        Absolute path to the directory that stores the SAE weights.
    """
    root = args.sae_root or usr_cfg.sae_save_dir
    if args.label:
        root = args.label + root
    return return_save_dir(root, args.n_d, args.k, args.nl, args.ckpt, args.lr)


def _load_model(model_name_or_dir: str) -> AutoModelForCausalLM:
    """Load the base language model used when training the SAE.

    Args:
        model_name_or_dir: Hugging Face identifier or local path for the model.

    Returns:
        The model on the appropriate device, ready for inference.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_dir,
        torch_dtype=torch.bfloat16,
    ).to(MODEL_DEVICE)
    model.eval()
    return model


def _load_sae(layer: int, args: argparse.Namespace, save_dir: str) -> SparseAutoEncoder:
    """Instantiate and load SAE weights for the selected layer.

    Args:
        layer: Layer index that was used when the SAE was trained.
        args: CLI arguments containing hyperparameters for the SAE.
        save_dir: Directory containing the saved `sae_layer{layer}.pth` file.

    Returns:
        Sparse autoencoder initialised with the stored weights.

    Raises:
        FileNotFoundError: If the checkpoint for the requested layer is missing.
    """
    cfg = SaeConfig(expansion_factor=args.n_d, k=args.k)
    sae = SparseAutoEncoder(cfg).to(SAE_DEVICE)
    weight_name = f"sae_layer{layer}.pth"
    weight_path = os.path.join(save_dir, weight_name)
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"SAE checkpoint not found: {weight_path}. Make sure train.py was run."
        )
    state = torch.load(weight_path, map_location=SAE_DEVICE)
    sae.load_state_dict(state)
    sae.eval()
    return sae


def _build_hook(model, layer: int) -> SimpleHook:
    """Attach a forward hook to capture hidden states for the target layer.

    Args:
        model: Language model that provides hidden activations.
        layer: Layer index whose output should be intercepted.

    Returns:
        A `SimpleHook` object that records activations during the forward pass.
    """
    target = model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
    return SimpleHook(target)


def _batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    """Yield slices of `items` with at most `batch_size` elements per chunk.

    Args:
        items: Sequence to iterate over.
        batch_size: Maximum number of elements to return per chunk.

    Yields:
        Subsequences of `items` preserving order.
    """
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _extract_batch(
    model,
    sae,
    tokenizer,
    hook: SimpleHook,
    texts: Sequence[str],
    nl: str,
) -> List[Dict[str, Any]]:
    """Run the model and SAE on a batch of cleaned sentences.

    Args:
        model: Language model used to produce contextual activations.
        sae: Sparse autoencoder that projects activations to latent space.
        tokenizer: Tokenizer aligned with `model`.
        hook: Previously registered hook capturing the hidden states.
        texts: Batch of sentences (without markers) to process.
        nl: Normalisation strategy applied before feeding activations to the SAE.

    Returns:
        One dictionary per input sentence containing tokens, latent indices, and
        latent activations.
    """
    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"].to(MODEL_DEVICE)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(MODEL_DEVICE)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    acts = hook.output
    if isinstance(acts, tuple):
        acts = acts[0]
    if acts.dim() == 2:
        acts = acts.unsqueeze(0)
    acts = acts[:, 1:, :]
    mask = attention_mask[:, 1:]
    records: List[Dict[str, Any]] = []

    for idx in range(input_ids.size(0)):
        valid = int(mask[idx].sum().item())
        if valid <= 0:
            records.append(
                {
                    "tokens": [],
                    "latent_indices": [],
                    "latent_acts": [],
                }
            )
            continue

        token_ids = input_ids[idx, 1 : valid + 1]
        tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        tokens = [tok.replace("▁", " ") for tok in tokens]

        activation = acts[idx, :valid, :]
        activation = normalize_activation(activation, nl)
        activation = activation.to(SAE_DEVICE)

        with torch.no_grad():
            out = sae(activation)

        latent_indices = out.latent_indices.detach().cpu().tolist()
        latent_acts = out.latent_acts.detach().cpu().tolist()

        records.append(
            {
                "tokens": tokens,
                "latent_indices": latent_indices,
                "latent_acts": latent_acts,
            }
        )

    hook.output = None
    return records


def _collect_layer(
    layer: int,
    args: argparse.Namespace,
    base_df: "pd.DataFrame",
    texts: Sequence[str],
    model,
    tokenizer,
    save_dir: str,
) -> "pd.DataFrame":
    """Collect SAE activations for a single layer and return a result dataframe.

    Args:
        layer: Layer index to process.
        args: CLI arguments describing normalisation and batching behaviour.
        base_df: Original dataframe loaded from the CSV file.
        texts: Cleaned or raw sentences to process (depending on modality flag).
        model: Language model loaded once for all layers.
        tokenizer: Tokeniser aligned with the language model.
        save_dir: Directory containing the SAE checkpoints.

    Returns:
        Dataframe with original columns plus SAE activations for the given layer.
    """

    print(f"[INFO] loading SAE for layer {layer} from {save_dir}")
    sae = _load_sae(layer, args, save_dir)
    hook = _build_hook(model, layer)

    aggregated: List[Dict[str, Any]] = []
    for batch in _batched(texts, args.batch_size):
        batch_records = _extract_batch(
            model=model,
            sae=sae,
            tokenizer=tokenizer,
            hook=hook,
            texts=batch,
            nl=args.nl,
        )
        aggregated.extend(batch_records)

    hook.hook.remove()

    if len(aggregated) != len(base_df):
        raise RuntimeError("Mismatch between input rows and collected records.")

    result_df = base_df.copy()
    result_df["sae_tokens"] = [rec["tokens"] for rec in aggregated]
    result_df["sae_latent_indices"] = [rec["latent_indices"] for rec in aggregated]
    result_df["sae_latent_acts"] = [rec["latent_acts"] for rec in aggregated]

    return result_df


def run(args: argparse.Namespace) -> None:
    """Entry point that orchestrates loading resources and exporting activations.

    Args:
        args: Parsed command-line arguments describing data sources and model
            hyperparameters.
    """
    usr_cfg = UsrConfig()
    model_name = args.model_name_or_dir or usr_cfg.model_name_or_dir

    save_dir = _resolve_save_dir(args, usr_cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    if "{layer}" not in args.output_template:
        raise ValueError("--output-template must contain '{layer}' placeholder.")

    print(f"[INFO] loading LLM from {model_name}")
    model = _load_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv(args.input_csv)

    if args.modality:
        if "Marked_Sentence_English" not in df.columns:
            raise KeyError(
                "Column 'Marked_Sentence_English' not found in the CSV file (modality mode)."
            )
        sentences = df["Marked_Sentence_English"].astype(str).tolist()
        texts = [_remove_markers(text) for text in sentences]
        # Add unmarked column in modality mode so it gets saved with outputs
        df["Marked_Sentence_English_Unmarked"] = texts
    else:
        if args.sentence_column not in df.columns:
            raise KeyError(
                f"Column '{args.sentence_column}' not found in the CSV file (generic mode)."
            )
        texts = df[args.sentence_column].astype(str).tolist()

    label_suffix = f"_{args.label}" if args.label else ""
    template_kwargs_base = {
        "label": args.label or "",
        "label_suffix": label_suffix,
    }

    for layer in args.layers:
        layer_df = _collect_layer(
            layer=layer,
            args=args,
            base_df=df,
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            save_dir=save_dir,
        )
        template_kwargs = dict(template_kwargs_base, layer=layer)
        layer_filename = args.output_template.format(**template_kwargs)
        layer_path = os.path.join(args.output_dir, layer_filename)
        layer_df.to_parquet(layer_path, index=False)
        print(f"[INFO] saved {len(layer_df)} rows to {layer_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SAE export utility.

    Returns:
        Structured namespace containing all CLI options and defaults.
    """
    parser = argparse.ArgumentParser(
        description="Compute SAE activations for Marked_Sentence_English and export to Parquet.",
    )
    parser.add_argument(
        "--input-csv",
        default="data/qp_modality_list_v1.csv",
        dest="input_csv",
        help="Path to the modality CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        dest="output_dir",
        help="Directory where layer-specific Parquet files will be saved.",
    )
    parser.add_argument(
        "--output-template",
        default="qp_modality_sae{label_suffix}_layer{layer}.parquet",
        help=(
            "Filename template for each layer. Must contain '{layer}'. Optional"
            " placeholders '{label}' and '{label_suffix}' are replaced when a"
            " label is provided."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[12],
        help="One or more layer indices used for SAE training.",
    )
    parser.add_argument("--n_d", type=int, default=16, help="Expansion ratio n/d used during training.")
    parser.add_argument("--k", type=int, default=32, help="Sparsity level k used during training.")
    parser.add_argument(
        "--nl",
        type=str,
        default="Scalar",
        help="Normalization method used during training (Standardization, Scalar, None).",
    )
    parser.add_argument("--ckpt", type=int, default=988240, help="Checkpoint id used during training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used during training.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of sentences processed at once.",
    )
    parser.add_argument(
        "--sae-root",
        type=str,
        default=None,
        help="Root directory that contains saved SAE checkpoints (defaults to config).",
    )
    parser.add_argument(
        "--model_name_or_dir",
        "--model-name-or-dir",
        type=str,
        default=None,
        dest="model_name_or_dir",
        help="Model name or path to load (defaults to config).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label prefix used during training to disambiguate outputs.",
    )
    # Modality vs generic input handling
    parser.add_argument(
        "--modality",
        dest="modality",
        action="store_true",
        help=(
            "Treat CSV as modality dataset with 'Marked_Sentence_English' and remove markers."
        ),
    )
    parser.add_argument(
        "--no-modality",
        dest="modality",
        action="store_false",
        help="Treat CSV as generic; read sentences from --sentence-column with no preprocessing.",
    )
    parser.set_defaults(modality=True)
    parser.add_argument(
        "--sentence-column",
        type=str,
        default="sentence",
        help="Column name that holds raw sentences in generic mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
