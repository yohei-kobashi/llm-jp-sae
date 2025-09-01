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
    num_examples,
    act_threshold_p,
):
    # Load model and tokenizer once
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_dir,
        torch_dtype=torch.bfloat16,
    ).to(MODEL_DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)

    for layer in layers:
        # Load per-layer SAE
        sae_config = SaeConfig(expansion_factor=n_d, k=k)
        sae = SparseAutoEncoder(sae_config).to(SAE_DEVICE)
        sae.eval()
        sae_path = os.path.join(save_dir, f"sae_layer{layer}.pth")
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE weight not found for layer {layer}: {sae_path}")
        sae.load_state_dict(torch.load(sae_path))

        # Prepare output dir per-layer
        features_dir = os.path.join(save_dir, f"features_layer{layer}")
        os.makedirs(features_dir, exist_ok=True)

        # Hooks
        hook_layer = model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
        hook = SimpleHook(hook_layer)
        hook_sae = SimpleHook(sae.encoder)

        num_features = sae.num_latents

        with torch.no_grad():
            # Pass 1: compute per-feature activation thresholds
            max_act_values = torch.zeros(num_features, dtype=torch.bfloat16, device=SAE_DEVICE)
            for step, batch in tqdm(enumerate(dl_test), desc=f"L{layer} pass1"):
                _ = model(batch.to(MODEL_DEVICE), use_cache=False)
                out = hook.output
                activation = out[0] if isinstance(out, tuple) else out
                activation = activation[:, 1:, :]
                shape = activation.shape  # bs, seq, d
                activation = activation.flatten(0, 1)
                activation = normalize_activation(activation, nl).to(SAE_DEVICE)
                _ = sae(activation)
                sae_activation = hook_sae.output.view(shape[0], shape[1], -1)  # bs, seq, num_latents
                max_act_values = torch.max(max_act_values, sae_activation.flatten(0, 1).max(dim=0)[0])

            act_thresholds = max_act_values * act_threshold_p

            # Pass 2: collect examples exceeding thresholds
            cnt_full = 0
            feature_records = [FeatureRecord(feature_id=i) for i in range(num_features)]
            feature_notfull = torch.ones(num_features, dtype=torch.bool, device=SAE_DEVICE)
            feature_cnt = torch.zeros(num_features, dtype=torch.int32, device=SAE_DEVICE)

            for step, batch in enumerate(tqdm(dl_test, desc=f"L{layer} pass2")):
                _ = model(batch.to(MODEL_DEVICE), use_cache=False)
                out = hook.output
                activation = out[0] if isinstance(out, tuple) else out
                activation = activation[:, 1:, :]
                shape = activation.shape
                activation = activation.flatten(0, 1)
                activation = normalize_activation(activation, nl).to(SAE_DEVICE)
                out_sae = sae(activation)
                latent_indices = out_sae.latent_indices.view(shape[0], shape[1], k)
                latent_acts = out_sae.latent_acts.view(shape[0], shape[1], k)
                sae_activation = hook_sae.output.view(shape[0], shape[1], -1)
                exceed_position = act_thresholds[latent_indices] < latent_acts

                for batch_idx in range(shape[0]):
                    if not feature_notfull.any():
                        break
                    indices = latent_indices[batch_idx][exceed_position[batch_idx]].unique()
                    if indices.numel() == 0:
                        continue
                    allowed = set(torch.nonzero(feature_notfull, as_tuple=False).squeeze(-1).tolist())
                    tokens = [
                        w.replace("▁", " ")
                        for w in tokenizer.convert_ids_to_tokens(batch[batch_idx][1:])
                    ]
                    for idx in indices.tolist():
                        if idx in allowed:
                            act_values = sae_activation[batch_idx, :, idx].tolist()
                            feature_records[idx].act_patterns.append(ActivationRecord(tokens=tokens, act_values=act_values))
                            feature_cnt[idx] += 1
                            if feature_cnt[idx].item() == num_examples:
                                feature_notfull[idx] = False
                                save_token_act(feature_records[idx], features_dir)
                                feature_records[idx] = None
                                cnt_full += 1
                            elif feature_cnt[idx].item() > num_examples:
                                raise ValueError("Feature count exceeds num_examples")

            for feature_record in tqdm(feature_records, desc=f"L{layer} save remaining"):
                if feature_record is not None and len(feature_record.act_patterns) > 0:
                    save_token_act(feature_record, features_dir)


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
        eval_cfg.num_examples,
        eval_cfg.act_threshold_p,
    )


if __name__ == "__main__":
    main()
