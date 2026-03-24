"""
LinguisticAnalyzer extension for Gemma Scope sparse dictionaries.

This module mirrors llamascope_analyzer_lingualens.py but is intended for
Gemma Scope / SAELens-format sparse dictionaries loaded through lm-saes.
"""

from __future__ import annotations

import argparse

from analyzer_lingualens import _parse_layers, _parse_torch_dtype
from llamascope_analyzer_lingualens import LlamaScopeLinguisticAnalyzer


class GemmaScopeLinguisticAnalyzer(LlamaScopeLinguisticAnalyzer):
    """
    LinguisticAnalyzer compatible with Gemma Scope sparse dictionaries.

    `sae_path_template` should resolve to a SAELens-compatible identifier such as:
        gemma-scope-2-1b-pt-res-all:layer_{}_width_16k_l0_small
    """


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-analyze feature text files with GemmaScopeLinguisticAnalyzer."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Base model name for TransformerLens HookedTransformer.from_pretrained().",
    )
    parser.add_argument(
        "--sae-path-template",
        required=True,
        help="SAELens-compatible template, e.g. gemma-scope-...:layer_{}_width_16k_l0_small",
    )
    parser.add_argument(
        "--feature-files",
        nargs="+",
        required=True,
        help="One or more feature text files (alternating original/minimal pairs).",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices, e.g. 0,7,15,19,25,31",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save batch analysis results.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top K base vectors per layer.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for analysis.")
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype placeholder kept for CLI compatibility.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. cuda:0 or cpu). Default: auto.",
    )
    parser.add_argument(
        "--prepend-bos",
        dest="prepend_bos",
        action="store_true",
        help="Prepend BOS before running the hooked model.",
    )
    parser.add_argument(
        "--no-prepend-bos",
        dest="prepend_bos",
        action="store_false",
        help="Disable BOS prepending before running the hooked model.",
    )
    parser.add_argument(
        "--fold-activation-scale",
        action="store_true",
        help="Fold dataset activation scale into SparseDictionary weights.",
    )
    parser.set_defaults(prepend_bos=True)
    args = parser.parse_args()

    layers = _parse_layers(args.layers)
    analyzer = GemmaScopeLinguisticAnalyzer(
        model_path=args.model_path,
        sae_path_template=args.sae_path_template,
        device=args.device,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
        prepend_bos=args.prepend_bos,
        fold_activation_scale=args.fold_activation_scale,
    )

    try:
        results = analyzer.batch_analyze_features(
            feature_files=args.feature_files,
            layers=layers,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )
        print("Batch analysis complete.")
        print(f"Output directory: {args.output_dir}")
        print(
            f"Successful: {results.get('successful_analyses', 0)}, "
            f"Failed: {results.get('failed_analyses', 0)}"
        )
    finally:
        analyzer.clear_cache()


if __name__ == "__main__":
    main()
