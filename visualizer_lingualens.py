"""
Visualization module for linguistic feature analysis with train.py SAEs.
"""

from __future__ import annotations

import html
import os
from typing import Any, Dict, List, Optional

import torch

from analyzer_lingualens import TrainSaeLinguisticAnalyzer
from LinguaLens.lingualens.metrics import compute_layer_stats, get_top_features_by_frc
from LinguaLens.lingualens.utils import load_text_data, save_json_results
from model import normalize_activation


class Visualizer:
    """
    Visualizer compatible with train.py SAE checkpoints.
    """

    def __init__(
        self,
        model_path: str,
        sae_path_template: str = "/path/to/sae/sae_layer{}.pth",
        device: Optional[str] = None,
        k: int = 32,
        normalization: str = "Scalar",
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.sae_path_template = sae_path_template
        self.analyzer = TrainSaeLinguisticAnalyzer(
            model_path=model_path,
            sae_path_template=sae_path_template,
            device=device,
            k=k,
            normalization=normalization,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
        )
        self.device = self.analyzer.device
        self.tokenizer = self.analyzer.tokenizer

    def _extract_structured_data(self, lines: List[str], layer_idx: int) -> List[Dict[str, Any]]:
        runtime = self.analyzer._get_sae_model(layer_idx)
        model = self.analyzer._load_base_model()
        sae = runtime["sae"]
        hook = runtime["hook"]

        structured_data: List[Dict[str, Any]] = []
        batch_size = self.analyzer.batch_size

        for start in range(0, len(lines), batch_size):
            batch_lines = lines[start : start + batch_size]
            enc = self.tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            acts = hook.output
            if isinstance(acts, tuple):
                acts = acts[0]
            if acts.dim() == 2:
                acts = acts.unsqueeze(0)
            acts = acts[:, 1:, :]
            token_mask = attention_mask[:, 1:]

            for i in range(input_ids.size(0)):
                valid_len = int(token_mask[i].sum().item())
                token_ids = input_ids[i, 1 : valid_len + 1]
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
                tokens = [tok.replace("▁", " ") for tok in tokens]

                sentence_tokens: List[Dict[str, Any]] = []
                if valid_len > 0:
                    activation = acts[i, :valid_len, :]
                    activation = normalize_activation(activation, self.analyzer.normalization)
                    activation = activation.to(self.device)

                    with torch.no_grad():
                        out = sae(activation)

                    top_indices = out.latent_indices.detach().cpu().tolist()
                    top_acts = out.latent_acts.detach().cpu().tolist()

                    for token, idxs, vals in zip(tokens, top_indices, top_acts):
                        token_activations = []
                        for base_vector, value in zip(idxs, vals):
                            if value > 0:
                                token_activations.append(
                                    {
                                        "base_vector": int(base_vector),
                                        "activation": float(value),
                                    }
                                )
                        sentence_tokens.append({"token": token, "activations": token_activations})

                structured_data.append(
                    {
                        "sentence_id": start + i + 1,
                        "tokens": sentence_tokens,
                    }
                )

            hook.output = None

        return structured_data

    def _write_simple_html(
        self,
        structured_data: List[Dict[str, Any]],
        base_vector_indices: List[int],
        output_html: str,
        title: str,
    ) -> None:
        output_dir = os.path.dirname(output_html)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        bv_set = set(base_vector_indices)
        rows = []
        for sent in structured_data:
            token_chunks = []
            for tok in sent["tokens"]:
                matched = [
                    (a["base_vector"], a["activation"])
                    for a in tok["activations"]
                    if a["base_vector"] in bv_set
                ]
                if matched:
                    max_act = max(v for _, v in matched)
                    alpha = min(0.85, 0.15 + max_act / 10.0)
                    details = ", ".join([f"BV{b}:{v:.3f}" for b, v in matched])
                    token_chunks.append(
                        f"<span class='tok hot' style='--a:{alpha:.3f}' title='{html.escape(details)}'>"
                        f"{html.escape(tok['token'])}</span>"
                    )
                else:
                    token_chunks.append(f"<span class='tok'>{html.escape(tok['token'])}</span>")

            rows.append(
                f"<div class='sent'><div class='sid'>#{sent['sentence_id']}</div>"
                f"<div class='tokens'>{' '.join(token_chunks)}</div></div>"
            )

        html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 1.3rem; }}
    .legend {{ margin-bottom: 16px; color: #444; font-size: 0.95rem; }}
    .sent {{ margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
    .sid {{ color: #555; font-size: 0.85rem; margin-bottom: 4px; }}
    .tokens {{ line-height: 2.0; }}
    .tok {{ display: inline-block; padding: 1px 4px; border-radius: 4px; }}
    .tok.hot {{ background: rgba(255, 128, 0, var(--a)); }}
    code {{ background: #f7f7f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class=\"legend\">Highlighted tokens activate selected base vectors: <code>{html.escape(str(base_vector_indices))}</code></div>
  {''.join(rows)}
</body>
</html>"""

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_doc)

    def generate_html_report(
        self,
        feature_file: str,
        layer_idx: int,
        output_html: str,
        top_k: int = 10,
        manual_base_vectors: Optional[List[int]] = None,
        analysis_mode: str = "FRC",
    ) -> Dict[str, Any]:
        lines = load_text_data(feature_file)
        print(f"[INPUT TEXT] {len(lines)} examples from {feature_file}")

        structured_data = self._extract_structured_data(lines, layer_idx)
        if analysis_mode == "FRC":
            analysis_results = self._analyze_frc_mode(structured_data, top_k)
        else:
            analysis_results = self._analyze_frequency_mode(structured_data, top_k)

        base_vector_indices = []
        if analysis_results.get("top_k_results"):
            base_vector_indices = [result[0] for result in analysis_results["top_k_results"]]
        if manual_base_vectors:
            base_vector_indices.extend(manual_base_vectors)
        base_vector_indices = sorted(set(base_vector_indices))

        if base_vector_indices:
            title = f"{os.path.basename(feature_file)} | layer {layer_idx}"
            self._write_simple_html(structured_data, base_vector_indices, output_html, title)
            print(f"[VISUALIZE] Visualization saved to '{output_html}'")
        else:
            print("[VISUALIZE] No valid base vectors to visualize.")

        return {
            "feature_file": feature_file,
            "layer": layer_idx,
            "analysis_mode": analysis_mode,
            "total_examples": len(lines),
            "base_vectors_visualized": base_vector_indices,
            "output_html": output_html,
            "analysis_results": analysis_results,
        }

    def _analyze_frc_mode(self, structured_data: List[Dict], top_k: int) -> Dict[str, Any]:
        layer_stats = compute_layer_stats(structured_data)
        top_features = get_top_features_by_frc(layer_stats, top_k)
        return {
            "mode": "FRC",
            "top_k_results": top_features,
            "full_stats": layer_stats,
            "total_base_vectors": len(layer_stats),
        }

    def _analyze_frequency_mode(self, structured_data: List[Dict], top_k: int) -> Dict[str, Any]:
        base_vector_counts: Dict[int, int] = {}
        for sentence in structured_data:
            for token in sentence["tokens"]:
                for activation in token["activations"]:
                    base_vec = activation["base_vector"]
                    base_vector_counts[base_vec] = base_vector_counts.get(base_vec, 0) + 1

        frequency_results = sorted(
            base_vector_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return {
            "mode": "frequency",
            "top_k_results": frequency_results,
            "total_base_vectors": len(base_vector_counts),
            "frequency_distribution": dict(frequency_results),
        }

    def batch_visualize(
        self,
        feature_files: List[str],
        layer_idx: int,
        output_dir: str,
        top_k: int = 10,
        analysis_mode: str = "FRC",
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        batch_results = {
            "layer": layer_idx,
            "analysis_mode": analysis_mode,
            "total_features": len(feature_files),
            "successful_visualizations": 0,
            "failed_visualizations": 0,
            "feature_results": {},
        }

        for feature_file in feature_files:
            try:
                feature_name = os.path.splitext(os.path.basename(feature_file))[0]
                output_html = os.path.join(output_dir, f"{feature_name}.html")
                results = self.generate_html_report(
                    feature_file,
                    layer_idx,
                    output_html,
                    top_k,
                    analysis_mode=analysis_mode,
                )
                batch_results["feature_results"][feature_name] = results
                batch_results["successful_visualizations"] += 1
                print(f"Generated visualization for {feature_name}")
            except Exception as e:
                print(f"Failed to visualize {feature_file}: {e}")
                batch_results["failed_visualizations"] += 1

        summary_file = os.path.join(output_dir, "visualization_summary.json")
        save_json_results(batch_results, summary_file)
        print(f"\nBatch visualization complete. Summary saved to {summary_file}")
        return batch_results

    def compare_layers_visualization(
        self,
        feature_file: str,
        layers: List[int],
        output_dir: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        feature_name = os.path.splitext(os.path.basename(feature_file))[0]

        comparison_results = {
            "feature": feature_name,
            "layers": layers,
            "layer_visualizations": {},
            "cross_layer_summary": {},
        }

        for layer_idx in layers:
            try:
                output_html = os.path.join(output_dir, f"{feature_name}_layer_{layer_idx:02d}.html")
                results = self.generate_html_report(
                    feature_file,
                    layer_idx,
                    output_html,
                    top_k,
                )
                comparison_results["layer_visualizations"][layer_idx] = results
                print(f"Generated layer {layer_idx} visualization")
            except Exception as e:
                print(f"Failed layer {layer_idx}: {e}")

        summary_file = os.path.join(output_dir, f"{feature_name}_layer_comparison.json")
        save_json_results(comparison_results, summary_file)
        return comparison_results

    def clear_cache(self):
        self.analyzer.clear_cache()
