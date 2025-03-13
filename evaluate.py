import argparse
import json
import os
import re

from rich.console import Console
from rich.text import Text
from tqdm import tqdm

from config import UsrConfig, return_save_dir


def get_color(value, target_color=(255, 120, 0), exp=0.8):
    if value < 0:
        value = 0
    else:
        value = value**exp
    white = (255, 255, 255)
    rgb_int = [int((1 - value) * white[i] + value * target_color[i]) for i in range(3)]
    rgb_str = f"rgb({rgb_int[0]},{rgb_int[1]},{rgb_int[2]})"
    return rgb_str


def show_token_acts(token_acts):
    console = Console()
    text = Text()
    for sentences in token_acts:
        for token, act in sentences:
            bg_color = get_color(act / 10)
            text.append(token, style=f"black on {bg_color}")
        text.append("\n")
    console.print(text)


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="auto_language",
        help="auto_language or manual_granularity",
    )
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

    save_dir = return_save_dir(
        usr_cfg.sae_save_dir,
        args.layer,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
    )
    features_dir = os.path.join(save_dir, "features")

    if args.eval_mode == "auto_language":
        auto_language(features_dir)
    elif args.eval_mode == "manual_granularity":
        manual_granularity(features_dir)
    else:
        raise ValueError(f"Invalid eval_mode: {args.eval_mode}")


def detect_language(token_act, idx):
    token = token_act[idx][0].strip()
    if re.match(r"^[a-zA-Z\s\"\'\(\),\.\-]+$", token):
        return "en"
    elif re.match(r"^[0-9]+$", token):
        return detect_language(token_act, idx - 1)
    else:
        return "ja"


def auto_language(features_dir, threshold=0.9):
    feature_pths = os.listdir(features_dir)
    feature_langs = [0, 0, 0]  # en, ja, mix
    for feature_pth in tqdm(feature_pths):
        if not feature_pth.endswith(".json"):
            continue
        with open(os.path.join(features_dir, feature_pth), "r") as f:
            data = json.load(f)
        en_ja_cnt = [0, 0]
        token_acts = data["token_act"]
        for token_act in token_acts:
            max_index = max(range(len(token_act)), key=lambda i: token_act[i][1])
            language = detect_language(token_act, max_index)
            if language == "en":
                en_ja_cnt[0] += 1
            elif language == "ja":
                en_ja_cnt[1] += 1
            else:
                raise ValueError(f"Invalid language: {language}")
        data["en_ja_cnt"] = en_ja_cnt
        with open(os.path.join(features_dir, feature_pth), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        if en_ja_cnt[0] / len(token_acts) > threshold:
            feature_langs[0] += 1
        elif en_ja_cnt[1] / len(token_acts) > threshold:
            feature_langs[1] += 1
        else:
            feature_langs[2] += 1

    print(f"en: {feature_langs[0]}, ja: {feature_langs[1]}, mix: {feature_langs[2]}")


def manual_granularity(features_dir, check_num=100):
    feature_pths = os.listdir(features_dir)
    gran_list = [0, 0, 0, 0]
    for feature_pth in tqdm(feature_pths):
        if not feature_pth.endswith(".json"):
            continue
        with open(os.path.join(features_dir, feature_pth), "r") as f:
            data = json.load(f)["token_act"]
        token_acts = data["token_act"]
        # 0:Token-Level, 1:Concept-level(synonym), 2:Concept-level(semantic sim.), 3:Uninterpretable
        show_token_acts(token_acts)
        gran = int(input("Enter granularity: "))
        data["granularity"] = gran
        gran_list[gran] += 1
        with open(os.path.join(features_dir, feature_pth), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(
        f"Token-Level: {gran_list[0]}, Concept-level(synonym): {gran_list[1]}, Concept-level(semantic sim.): {gran_list[2]}, Uninterpretable: {gran_list[3]}"
    )


if __name__ == "__main__":
    evaluate()
