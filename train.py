import argparse
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_constant_schedule_with_warmup

from config import SaeConfig, TrainConfig, UsrConfig, return_save_dir
from dataset import CustomWikiDataset
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


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5) -> Tensor:
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        distances = torch.norm(points - guess, dim=1)
        # Avoid division by zero
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


def train(
    dl_train, dl_val, train_cfg, model_dir, layer, n_d, k, nl, ckpt, lr, save_dir
):
    # load the language model to extract activations
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, f"iter_{str(ckpt).zfill(7)}"),
        torch_dtype=torch.bfloat16,
    ).to(MODEL_DEVICE)
    model.eval()

    # initialize the sparse autoencoder
    sae_config = SaeConfig(n_d=n_d, k=k)
    sae = SparseAutoEncoder(sae_config).to(SAE_DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, eps=6.25e-10)
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=train_cfg.lr_warmup_steps
    )

    # setup hook
    hook_layer = (
        model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
    )
    hook = SimpleHook(hook_layer)

    global_step = 0
    loss_sum = 0.0

    for batch in tqdm(dl_train, desc="Training"):
        with torch.inference_mode():
            _ = model(batch.to(MODEL_DEVICE), use_cache=False)
            activation = hook.output if layer == 0 else hook.output[0]
            # remove sos token
            activation = activation[:, 1:, :]
            # (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, hidden_size)
            activation = activation.flatten(0, 1)
            # normalize activations
            activation = normalize_activation(activation, nl)

        # split the activations into chunks
        for chunk in torch.chunk(activation, train_cfg.inf_bs_expansion, dim=0):
            # Initialize decoder bias with the geometric median of the chunk
            if global_step == 0:
                median = geometric_median(chunk.to(SAE_DEVICE))
                sae.b_dec.data = median.to(sae.dtype)
            # make sure the decoder weights are unit norm
            sae.set_decoder_norm_to_unit_norm()
            optimizer.zero_grad()
            out = sae(chunk.to(SAE_DEVICE))
            loss = out.loss
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            lr_scheduler.step()

            if global_step % train_cfg.logging_step == 0 and global_step > 0:
                print(f"Step: {global_step}, Loss: {loss_sum / train_cfg.logging_step}")
                loss_sum = 0.0
            global_step += 1

    # save the trained SAE
    torch.save(sae.state_dict(), os.path.join(save_dir, "sae.pth"))

    # evaluation
    del dl_train, optimizer, lr_scheduler
    sae.eval()
    loss_eval = 0

    with torch.no_grad():
        for batch in tqdm(dl_val):
            _ = model(batch.to(MODEL_DEVICE), use_cache=False)
            activation = hook.output if layer == 0 else hook.output[0]
            activation = activation[:, 1:, :]
            activation = activation.flatten(0, 1)
            activation = normalize_activation(activation, nl)
            for chunk in torch.chunk(activation, train_cfg.inf_bs_expansion, dim=0):
                out = sae(chunk.to(SAE_DEVICE))
                loss_eval += out.loss.item()
        print(f"Validation Loss: {loss_eval / len(dl_val)}")

    print("Training finished.")


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
    train_cfg = TrainConfig()

    train_data_pth = os.path.join(usr_cfg.tokenized_data_dir, "train_data.pt")
    val_data_pth = os.path.join(usr_cfg.tokenized_data_dir, "val_data.pt")

    dataset_train = CustomWikiDataset(train_data_pth)
    dataset_val = CustomWikiDataset(val_data_pth)
    dl_train = DataLoader(
        dataset_train,
        batch_size=train_cfg.batch_size * train_cfg.inf_bs_expansion,
        shuffle=True,
    )
    dl_val = DataLoader(
        dataset_val,
        batch_size=train_cfg.batch_size * train_cfg.inf_bs_expansion,
        shuffle=False,
    )

    print(
        f"Layer: {args.layer}, n/d: {args.n_d}, k: {args.k}, nl: {args.nl}, ckpt: {args.ckpt}, lr: {args.lr}"
    )
    save_dir = return_save_dir(
        usr_cfg.sae_save_dir,
        args.layer,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
    )
    if os.path.exists(os.path.join(save_dir, "sae.pth")):
        print(f"Already exists at: {save_dir}")
        if input("Overwrite? (y/n): ").lower() != "y":
            exit()
    os.makedirs(save_dir, exist_ok=True)

    train(
        dl_train,
        dl_val,
        train_cfg,
        usr_cfg.llmjp_model_dir,
        args.layer,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
        save_dir,
    )


if __name__ == "__main__":
    main()
