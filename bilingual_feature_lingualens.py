"""
Bilingual/generalized analysis module for feature-group comparison.

This module can compare any two feature groups (not only EN vs ZH)
using layer-wise base-vector overlaps.
"""

import os
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from LinguaLens.lingualens.utils import save_json_results, ProgressLogger


class BilingualAnalyzer:
    """
    Analyzer for bilingual linguistic feature comparison.
    
    This class provides functionality to compare linguistic features
    across languages, particularly English and Chinese.
    """
    
    def __init__(self, vector_data_dir: Optional[str] = None):
        """
        Initialize the bilingual analyzer.
        
        Args:
            vector_data_dir: Directory containing vector data files
        """
        self.vector_data_dir = vector_data_dir or "data/vectors"
    
    def load_vector_data(self, filepath: str) -> List[List[str]]:
        """
        Load vector data from file.
        
        Each line contains space-separated base vector IDs for that layer.
        
        Args:
            filepath: Path to vector data file
            
        Returns:
            List of sets, one per layer, containing base vector IDs
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vector file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        
        return [set(line.split()) for line in lines]
    
    def compute_overlap_score(self, set_en: set, set_cn: set) -> float:
        """
        Compute overlap score between two sets of base vectors.
        
        Args:
            set_en: English base vector set
            set_cn: Chinese base vector set
            
        Returns:
            Overlap ratio (intersection size / English set size)
        """
        if not set_en:
            return 0.0
        return len(set_en & set_cn) / len(set_en)
    
    def _normalize_feature_pair(
        self,
        pair: Union[
            Tuple[List[str], str],
            Tuple[List[str], List[str]],
            Dict[str, Any],
        ],
    ) -> Tuple[List[str], List[str], str, str]:
        """
        Normalize pair specs into two feature groups.

        Supported input forms:
        - (group_a_features: List[str], group_b_feature: str)
        - (group_a_features: List[str], group_b_features: List[str])
        - {"group_a": [...], "group_b": [...], "label_a": "...", "label_b": "..."}
        """
        if isinstance(pair, dict):
            group_a = list(pair.get("group_a", []))
            group_b = list(pair.get("group_b", []))
            label_a = pair.get("label_a", "Group A")
            label_b = pair.get("label_b", "Group B")
            if not group_a or not group_b:
                raise ValueError("Dict pair must include non-empty 'group_a' and 'group_b'.")
            return group_a, group_b, label_a, label_b

        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Unsupported pair format: {pair}")

        group_a = list(pair[0])
        rhs = pair[1]
        if isinstance(rhs, str):
            group_b = [rhs]
        else:
            group_b = list(rhs)

        return group_a, group_b, "Group A", "Group B"

    def analyze_bilingual_feature_similarity(
        self,
        feature_pairs: List[
            Union[
                Tuple[List[str], str],
                Tuple[List[str], List[str]],
                Dict[str, Any],
            ]
        ],
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Analyze similarity between two feature groups.

        Args:
            feature_pairs: Pair specs (tuple or dict) normalized by _normalize_feature_pair
            output_path: Path to save results

        Returns:
            Feature-group similarity analysis results
        """
        results = {
            "total_pairs": len(feature_pairs),
            "pair_results": [],
            "layer_statistics": {},
            "overall_statistics": {}
        }
        
        progress = ProgressLogger(len(feature_pairs), "Analyzing feature-group similarities")
        
        all_layer_scores = {}  # layer_idx -> list of scores
        
        for pair in feature_pairs:
            try:
                group_a, group_b, label_a, label_b = self._normalize_feature_pair(pair)

                group_a_vectors_list = []
                for feat in group_a:
                    path = os.path.join(self.vector_data_dir, f"{feat}.txt")
                    group_a_vectors_list.append(self.load_vector_data(path))

                group_b_vectors_list = []
                for feat in group_b:
                    path = os.path.join(self.vector_data_dir, f"{feat}.txt")
                    group_b_vectors_list.append(self.load_vector_data(path))

                # Compute layer-wise similarity
                num_layers = max(
                    max(len(v) for v in group_a_vectors_list) if group_a_vectors_list else 0,
                    max(len(v) for v in group_b_vectors_list) if group_b_vectors_list else 0,
                )
                layer_scores = []
                
                for layer_idx in range(num_layers):
                    group_a_union = set()
                    for vecs in group_a_vectors_list:
                        if layer_idx < len(vecs):
                            group_a_union |= vecs[layer_idx]

                    group_b_union = set()
                    for vecs in group_b_vectors_list:
                        if layer_idx < len(vecs):
                            group_b_union |= vecs[layer_idx]

                    overlap_score = self.compute_overlap_score(group_a_union, group_b_union)
                    layer_scores.append(overlap_score)
                    
                    # Collect for overall statistics
                    if layer_idx not in all_layer_scores:
                        all_layer_scores[layer_idx] = []
                    all_layer_scores[layer_idx].append(overlap_score)
                
                # Store pair results
                avg_similarity = sum(layer_scores) / len(layer_scores) if layer_scores else 0.0
                max_similarity = max(layer_scores) if layer_scores else 0.0
                best_layer = layer_scores.index(max_similarity) if layer_scores else None

                pair_result = {
                    "group_a_features": group_a,
                    "group_b_features": group_b,
                    "group_a_label": label_a,
                    "group_b_label": label_b,
                    # compatibility keys
                    "english_features": group_a,
                    "chinese_feature": "/".join(group_b),
                    "layer_scores": layer_scores,
                    "average_similarity": avg_similarity,
                    "max_similarity": max_similarity,
                    "best_layer": best_layer,
                }
                
                results["pair_results"].append(pair_result)
                progress.update()
                
            except Exception as e:
                print(f"Warning: Skipped pair {pair} due to {e}")
                progress.update()
                continue
        
        progress.finish()
        
        # Compute layer statistics
        for layer_idx, scores in all_layer_scores.items():
            results["layer_statistics"][layer_idx] = {
                "mean_similarity": np.mean(scores),
                "std_similarity": np.std(scores),
                "max_similarity": np.max(scores),
                "min_similarity": np.min(scores)
            }
        
        # Compute overall statistics
        all_scores = [score for scores in all_layer_scores.values() for score in scores]
        if all_scores:
            results["overall_statistics"] = {
                "mean_similarity": np.mean(all_scores),
                "std_similarity": np.std(all_scores),
                "best_performing_layers": sorted(
                    results["layer_statistics"].keys(),
                    key=lambda l: results["layer_statistics"][l]["mean_similarity"],
                    reverse=True
                )[:5]
            }
        else:
            results["overall_statistics"] = {
                "mean_similarity": 0.0,
                "std_similarity": 0.0,
                "best_performing_layers": []
            }
        
        # Save results
        save_json_results(results, output_path)
        print(f"Feature-group analysis complete. Results saved to {output_path}")
        
        return results
    
    def load_default_feature_pairs(self) -> List[Tuple[List[str], str]]:
        """
        Load default English-Chinese feature pairs.
        
        Returns:
            List of default feature pairs for analysis
        """
        # Default feature pairs based on linguistic correspondence
        pairs = [
            (['2-noun_plural-Morphology'], '110-们_复数后缀-形态学'),
            (['3-agentive_suffix-Morphology'], '109-化_动词性后缀-形态学'),
            (['9-nominal_suffix-Morphology'], '108-性_抽象名词后缀-形态学'),
            (['21-intransitive_verb-Syntax'], '112-不及物动词-句法学&语义学'),
            (['22-transitive_verb-Syntax'], '113-及物动词-句法学&语义学'),
            (['23-linking_verb-Syntax'], '114-系动词-句法学'),
            (['13-possessive_form-Morphology&Syntax'], '115-属格-句法学&语义学'),
            (['25-subject_auxiliary_inversion-Syntax', '26-subject_verb_inversion-Syntax'], '116-逆向结构-句法学&语义学'),
            (['27-passive_voice-Syntax&Semantics', '75-passive-Semantics&Syntax'], '117-被动语态-句法学&语义学'),
            (['24-anaphor-Syntax&Pragmatics'], '119-回指-句法学&语义学&语用学'),
            (['30-indirect_speech-Syntax&Pragmatics'], '120-间接引语-句法学&语用学'),
            (['31-elliptical_sentences-Syntax'], '121-省略句-句法学&语用学'),
            (['33-appositives-Syntax'], '122-同位结构-句法学'),
            (['38-imperative_sentence-Syntax&Pragmatics'], '125-祈使句-句法学&语用学'),
            (['17-comparative-Morphology&Semantics', '91-comparative-Semantics'], '143-比较-语义学'),
            (['47-universal_quantifiers-Syntax&Semantics', '48-existential_quantifiers-Syntax&Semantics'], '156-数量词-句法学&语义学'),
            (['96-conditional-Syntax&Semantics'], '131-条件句-句法学&语义学'),
            (['28-subjunctive_mood-Syntax&Semantics', '100-subjunctive-Syntax&Semantics'], '133-情态-语义学&语用学'),
            (['65-past-Semantics', '66-future-Semantics', '67-present_progressive-Semantics',
              '68-present_perfect-Semantics', '69-past_progressive-Semantics', '70-past_perfect-Semantics',
              '71-future_progressive-Semantics', '72-future_perfect-Semantics'], '134-时体标记-形态学&语义学'),
            (['52-intensifiers-Semantics&Pragmatics'], '126-语气助词-形态学&语义学&语用学'),
            (['74-deontic-Semantics&Pragmatics'], '140-应该_义务情态-语义学&语用学'),
            (['93-transitional-Semantics&Pragmatics'], '148-递进-语义学&语用学'),
            (['101-deixis-Pragmatics&Semantics'], '149-指示-语义学&语用学'),
            (['102-turn_taking-Pragmatics'], '150-话轮转换-语用学'),
            (['103-euphemism-Pragmatics&Semantics'], '151-委婉语-语用学&语义学'),
            (['104-personification-Semantics&Pragmatics'], '152-拟人-语义学&语用学'),
            (['105-hyperbole-Semantics&Pragmatics'], '153-夸张-语义学&语用学'),
            (['106-discourse_markers-Pragmatics'], '154-话语标记-语用学'),
            (['107-politeness-Pragmatics'], '155-礼貌-语用学')
        ]
        
        return pairs
    
    def generate_similarity_heatmap(
        self,
        results: Dict[str, Any],
        output_path: str,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Generate a heatmap showing feature-group similarity across layers.
        
        Args:
            results: Results from analyze_bilingual_feature_similarity
            output_path: Path to save the heatmap
            figsize: Figure size tuple
        """
        # Prepare data for heatmap
        pair_names = []
        layer_similarities = []
        
        for pair_result in results["pair_results"]:
            group_a = pair_result.get("group_a_features", pair_result.get("english_features", []))
            group_b = pair_result.get("group_b_features", [pair_result.get("chinese_feature", "")])
            label_a = pair_result.get("group_a_label", "Group A")
            label_b = pair_result.get("group_b_label", "Group B")

            group_a_str = "/".join(group_a)
            group_b_str = "/".join(group_b)
            pair_name = f"{label_a}: {group_a_str} vs {label_b}: {group_b_str}"
            
            # Truncate long names for display
            if len(pair_name) > 60:
                pair_name = pair_name[:57] + "..."
            
            pair_names.append(pair_name)
            layer_similarities.append(pair_result["layer_scores"])
        
        if not layer_similarities:
            print("No pair results available for heatmap.")
            return

        # Create heatmap
        plt.figure(figsize=figsize)
        
        similarity_matrix = np.array(layer_similarities)
        
        sns.heatmap(
            similarity_matrix,
            yticklabels=pair_names,
            xticklabels=[f"L{i:02d}" for i in range(similarity_matrix.shape[1])],
            cmap="viridis",
            cbar_kws={"label": "Overlap Similarity"},
            annot=False
        )
        
        plt.title("Feature-Group Similarity Across Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("Feature Pairs")
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Similarity heatmap saved to {output_path}")
    
    def export_similarity_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Export a detailed similarity report.
        
        Args:
            results: Results from analyze_bilingual_feature_similarity
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Feature-Group Similarity Report\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n")
            overall = results["overall_statistics"]
            f.write(f"- Mean Similarity: {overall['mean_similarity']:.4f}\n")
            f.write(f"- Standard Deviation: {overall['std_similarity']:.4f}\n")
            f.write(f"- Best Performing Layers: {', '.join(map(str, overall['best_performing_layers']))}\n\n")
            
            # Layer statistics (if available)
            if "layer_statistics" in results:
                f.write("## Layer Statistics\n")
                for layer_idx in sorted(results["layer_statistics"].keys()):
                    stats = results["layer_statistics"][layer_idx]
                    f.write(f"### Layer {layer_idx:02d}\n")
                    f.write(f"- Mean: {stats['mean_similarity']:.4f}\n")
                    f.write(f"- Std: {stats['std_similarity']:.4f}\n")
                    f.write(f"- Range: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]\n\n")
            
            # Feature pair details
            f.write("## Feature Pair Details\n")
            sorted_pairs = sorted(results["pair_results"], 
                                key=lambda x: x["average_similarity"], reverse=True)
            
            for pair in sorted_pairs:
                group_a = pair.get("group_a_features", pair.get("english_features", []))
                group_b = pair.get("group_b_features", [pair.get("chinese_feature", "")])
                label_a = pair.get("group_a_label", "Group A")
                label_b = pair.get("group_b_label", "Group B")
                f.write(f"### {label_a}: {'/'.join(group_a)} vs {label_b}: {'/'.join(group_b)}\n")
                f.write(f"- Average Similarity: {pair['average_similarity']:.4f}\n")
                f.write(f"- Max Similarity: {pair['max_similarity']:.4f} (Layer {pair['best_layer']})\n")
                f.write(f"- Layer Scores: {', '.join([f'{s:.3f}' for s in pair['layer_scores']])}\n\n")
        
        print(f"Detailed report saved to {output_path}")
