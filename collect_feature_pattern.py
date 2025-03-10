import argparse
import os

import torch
from configs import EvalConfig, SaeConfig, TrainConfig, UsrConfig, return_save_dir
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from dataset import CustomWikiDataset, FeatureRecord
from model import SimpleHook, SparseAutoEncoder, normalize_activation
from tqdm import tqdm

DEVICE = torch.device("cuda:0")


def collect_feature_pattern(
    dl_test,
    train_cfg,
    model_dir,
    layer,
    n_d,
    k,
    nl,
    ckpt,
    lr,
    save_dir,
    num_examples,
    act_threshold_p,
):
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, f"iter_{str(ckpt).zfill(7)}"),
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    model.eval()
    sae_config = SaeConfig(n_d=n_d, k=k)
    sae = SparseAutoEncoder(sae_config).to(DEVICE)
    sae.eval()
    sae.load_state_dict(torch.load(os.path.join(save_dir, "sae.pt")))

    hook_layer = (
        model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
    )
    hook = SimpleHook(hook_layer)
    hook_sae = SimpleHook(sae.encoder)

    num_features = sae_config.d_in * n_d

    with torch.no_grad():
        # Find the activation threshold (act_threshold) for each feature
        max_act_values = torch.zeros(num_features, dtype=torch.bfloat16).to(DEVICE)
        for step, batch in enumerate(dl_test):
            _ = model(batch.to(DEVICE), use_cache=False)
            activation = hook.output if layer == 0 else hook.output[0]
            activation = activation[:, 1:, :]
            shape = activation.shape  # bs, seq, d
            activation = activation.flatten(0, 1)
            activation = normalize_activation(activation, nl)
            _ = sae(activation)
            sae_activation = hook_sae.output.view(
                shape[0], shape[1], -1
            )  # bs, seq, d_in * n_d
            max_act_values = torch.max(
                max_act_values, sae_activation.flatten(0, 1).max(dim=0)[0]
            )

        act_thresholds = max_act_values * act_threshold_p

        cnt_full = 0
        feature_records = [FeatureRecord(feature_id=i) for i in range(num_features)]

        with tqdm(dl_test) as pbar:
            for step, batch in enumerate(pbar):
                pbar.set_postfix(cnt_full=cnt_full)
                _ = model(batch.to(DEVICE), use_cache=False)
                activation = hook.output if layer == 0 else hook.output[0]
                activation = activation[:, 1:, :]
                shape = activation.shape
                activation = activation.flatten(0, 1)
                activation = normalize_activation(activation, nl)
                out = sae(activation)
                latent_indices = out.latent_indices.view(shape[0], shape[1], k)
                latent_acts = out.latent_acts.view(shape[0], shape[1], k)
                sae_activation = hook_sae.output.view(
                    shape[0], shape[1], -1
                )
                

        # Collect feature patterns


# def auto_lang_trend():
#     pass


# def manual_gran_trend():
#     pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=int, default=988240, help="Checkpoint")
    parser.add_argument(
        "--layer", type=int, default=12, help="Layer index to extract activations"
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

    save_dir = return_save_dir(
        usr_cfg.model_save_dir,
        args.layer,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
    )
    collect_feature_pattern(
        dl_test,
        train_cfg,
        usr_cfg.llmjp_model_dir,
        args.layer,
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
