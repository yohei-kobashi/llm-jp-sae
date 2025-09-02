import argparse
import json
import math
import os

import torch
from config import EvalConfig, SaeConfig, TrainConfig, UsrConfig, return_save_dir
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import ActivationRecord, CustomWikiDataset, FeatureRecord
import heapq
from typing import Dict, List, Tuple
from model import SimpleHook, SparseAutoEncoder, normalize_activation

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


def collect_feature_pattern(
    dl_test,
    model_name_or_dir,
    layers,
    n_d,
    k,
    nl,
    ckpt,
    lr,
    save_dir,
    top_n,
):
    # Load model and tokenizer once
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_dir,
        torch_dtype=torch.bfloat16,
    ).to(MODEL_DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)

    # Prepare per-layer SAE, hooks, and heaps
    saes: Dict[int, SparseAutoEncoder] = {}
    hooks_model: Dict[int, SimpleHook] = {}
    hooks_sae: Dict[int, SimpleHook] = {}
    num_features_per_layer: Dict[int, int] = {}
    # For each layer, maintain a list of min-heaps per feature id
    # Each heap stores tuples: (score, counter, ActivationRecord)
    heaps_per_layer: Dict[int, List[List[Tuple[float, int, ActivationRecord]]]] = {}
    counter = 0  # global tie-breaker

    for layer in layers:
        # SAE per layer
        sae_config = SaeConfig(expansion_factor=n_d, k=k)
        sae = SparseAutoEncoder(sae_config).to(SAE_DEVICE)
        sae.eval()
        sae_path = os.path.join(save_dir, f"sae_layer{layer}.pth")
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE weight not found for layer {layer}: {sae_path}")
        sae.load_state_dict(torch.load(sae_path))
        saes[layer] = sae
        num_features_per_layer[layer] = sae.num_latents
        hooks_sae[layer] = SimpleHook(sae.encoder)

        # Hook target layer of the model
        target = model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
        hooks_model[layer] = SimpleHook(target)

        # Init heaps per feature
        heaps_per_layer[layer] = [[] for _ in range(sae.num_latents)]

    # Single pass over the dataset; each batch computed once for all layers
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dl_test, desc="collect(top-N)")):
            batch = batch.to(MODEL_DEVICE)
            _ = model(batch, use_cache=False)

            # tokens for this batch samples will be computed lazily per sample
            tokens_cache: Dict[int, List[str]] = {}

            for layer in layers:
                hook = hooks_model[layer]
                sae = saes[layer]
                hook_sae = hooks_sae[layer]

                out = hook.output
                activation = out[0] if isinstance(out, tuple) else out
                activation = activation[:, 1:, :]
                shape = activation.shape  # bs, seq, d
                activation = activation.flatten(0, 1)
                activation = normalize_activation(activation, nl).to(SAE_DEVICE)

                out_sae = sae(activation)
                latent_indices = out_sae.latent_indices.view(shape[0], shape[1], k)
                latent_acts = out_sae.latent_acts.view(shape[0], shape[1], k)
                sae_activation = hook_sae.output.view(shape[0], shape[1], -1)

                # For each sample in batch, compute per-feature max activation (from sparse top-k)
                for b in range(shape[0]):
                    # Build per-feature max score using sparse indices/acts
                    flat_idx = latent_indices[b].reshape(-1)
                    flat_act = latent_acts[b].reshape(-1)
                    # Accumulate maxima in a python dict to avoid giant dense ops
                    feat_max: Dict[int, float] = {}
                    idx_list = flat_idx.tolist()
                    act_list = flat_act.tolist()
                    for idx_i, act_i in zip(idx_list, act_list):
                        # act_i is a python float already if dtype is fp16/bf16; ensure float
                        prev = feat_max.get(idx_i)
                        if prev is None or act_i > prev:
                            feat_max[idx_i] = float(act_i)

                    if not feat_max:
                        continue

                    # Compute tokens lazily only if we are going to insert something
                    # We don't know yet, so we prepare a quick reference to heaps to compare
                    # We'll compute tokens only when pushing
                    for idx_i, score in feat_max.items():
                        heap = heaps_per_layer[layer][idx_i]
                        # Quick skip if heap is full and score is not better than current min
                        if len(heap) >= top_n and score <= heap[0][0]:
                            continue

                        # Prepare tokens for this sample only once
                        if b not in tokens_cache:
                            tokens_cache[b] = [
                                w.replace("▁", " ")
                                for w in tokenizer.convert_ids_to_tokens(batch[b][1:].tolist())
                            ]
                        tokens = tokens_cache[b]
                        # Extract per-token activation values for this feature from SAE encoder output
                        act_values = sae_activation[b, :, idx_i].tolist()
                        record = ActivationRecord(tokens=tokens, act_values=act_values)

                        counter += 1
                        item = (float(max(0.0, max(act_values))), counter, record)

                        if len(heap) < top_n:
                            heapq.heappush(heap, item)
                        else:
                            # Replace smallest if current is larger
                            if item[0] > heap[0][0]:
                                heapq.heapreplace(heap, item)

    # Save per-layer features
    for layer in layers:
        features_dir = os.path.join(save_dir, f"features_layer{layer}")
        os.makedirs(features_dir, exist_ok=True)
        num_features = num_features_per_layer[layer]
        heaps = heaps_per_layer[layer]

        for feat_id in tqdm(range(num_features), desc=f"L{layer} save"):
            heap = heaps[feat_id]
            if not heap:
                continue
            # Sort descending by score
            items_sorted = sorted(heap, key=lambda x: x[0], reverse=True)
            fr = FeatureRecord(feature_id=feat_id)
            for _, _, rec in items_sorted:
                fr.act_patterns.append(rec)
            save_token_act(fr, features_dir)


def _format_activation_record(activation_record, max_act):
    tokens = activation_record.tokens
    acts = activation_record.act_values
    token_act_list = []

    for token, act in zip(tokens, acts):
        if act < 0:
            act = 0
        else:
            act = math.ceil(act * 10 / max_act)
        token_act_list.append([token, act])

    return token_act_list


def format_example(activation_records, max_act):
    token_act_list = []
    for idx_sample, activation_record in enumerate(activation_records):
        token_act = _format_activation_record(activation_record, max_act)
        token_act_list.append(token_act)

    return token_act_list


def save_token_act(
    feature_record,
    features_dir,
):
    len_data = len(feature_record.act_patterns)
    max_act = max(
        [max(feature_record.act_patterns[i].act_values) for i in range(len_data)]
    )

    token_act_list = format_example(
        feature_record.act_patterns,
        max_act,
    )

    with open(
        os.path.join(features_dir, f"{feature_record.feature_id}.json"), "w"
    ) as f:
        json.dump(
            {
                "token_act": token_act_list,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=int, default=988240, help="Checkpoint (for save_dir naming)")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None, help="Layer indices to extract activations"
    )
    parser.add_argument("--n_d", type=int, default=16, help="Expansion ratio (n/d)")
    parser.add_argument(
        "--k", type=int, default=32, help="K parameter for SAE (sparsity)"
    )
    parser.add_argument(
        "--nl",
        type=str,
        default="Scalar",
        help="normalization method: Standardization, Scalar, None",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--top_n", type=int, default=None, help="Top-N examples per feature (overrides thresholding)")
    parser.add_argument(
        "--model_name_or_dir",
        type=str,
        default=None,
        help="Model name or path (overrides UsrConfig.model_name_or_dir)",
    )
    args = parser.parse_args()

    usr_cfg = UsrConfig()
    eval_cfg = EvalConfig()
    train_cfg = TrainConfig()

    test_data_pth = os.path.join(usr_cfg.tokenized_data_dir, "test_data.pt")
    dataset_test = CustomWikiDataset(test_data_pth)
    dl_test = DataLoader(
        dataset_test,
        batch_size=train_cfg.batch_size * train_cfg.inf_bs_expansion,
        shuffle=False,
    )

    # Resolve model path/name
    model_name_or_dir = usr_cfg.model_name_or_dir
    if args.model_name_or_dir:
        model_name_or_dir = args.model_name_or_dir

    # Resolve layers
    layers = args.layers or [12]

    save_dir = return_save_dir(
        usr_cfg.sae_save_dir,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
    )
    top_n = args.top_n if args.top_n is not None else eval_cfg.num_examples

    collect_feature_pattern(
        dl_test,
        model_name_or_dir,
        layers,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
        save_dir,
        top_n,
    )


if __name__ == "__main__":
    main()
