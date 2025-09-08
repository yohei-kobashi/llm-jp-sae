import argparse
import json
import math
import os

import torch
import concurrent.futures as _futures
import os as _os
from config import EvalConfig, SaeConfig, TrainConfig, UsrConfig, return_save_dir
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import ActivationRecord, CustomWikiDataset, FeatureRecord
import heapq
from typing import Dict, List, Tuple, Optional
import time
import pickle
import numpy as np

def _read_proc_rss_kb() -> Optional[int]:
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])  # kB
    except Exception:
        return None
    return None

def _get_rss_bytes() -> Optional[int]:
    # Try psutil first if available
    try:
        import psutil  # type: ignore
        p = psutil.Process()
        return int(p.memory_info().rss)
    except Exception:
        kb = _read_proc_rss_kb()
        return kb * 1024 if kb is not None else None

def _bytes_to_str(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"

def _print_mem(tag: str):
    cpu = _bytes_to_str(_get_rss_bytes())
    if torch.cuda.is_available():
        try:
            alloc = _bytes_to_str(torch.cuda.memory_allocated())
            reserved = _bytes_to_str(torch.cuda.memory_reserved())
            peak = _bytes_to_str(torch.cuda.max_memory_allocated())
            print(f"[MEM] {tag} | CPU RSS={cpu} | GPU alloc={alloc} reserved={reserved} peak={peak}", flush=True)
        except Exception as e:
            print(f"[MEM] {tag} | CPU RSS={cpu} | GPU n/a ({e})", flush=True)
    else:
        print(f"[MEM] {tag} | CPU RSS={cpu} | GPU=CPU only", flush=True)
from model import SimpleHook, SparseAutoEncoder, normalize_activation

def _log(msg: str):
    try:
        tqdm.write(msg)
    except Exception:
        print(msg, flush=True)

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


# Optional fast JSON
try:
    import orjson as _orjson  # type: ignore
    _HAS_ORJSON = True
except Exception:
    _HAS_ORJSON = False

# --- Binary intermediate record helpers ---
def _bin_append_record(path: str, obj: dict):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    # Write length-prefixed record to allow streaming append
    with open(path, "ab") as f:
        f.write(len(data).to_bytes(8, "little"))
        f.write(data)


def _bin_iter_records(path: str):
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if not hdr or len(hdr) < 8:
                break
            n = int.from_bytes(hdr, "little")
            buf = f.read(n)
            if not buf or len(buf) < n:
                break
            yield pickle.loads(buf)


def _bin_append_many(path: str, objs: List[dict]):
    if not objs:
        return
    with open(path, "ab") as f:
        for obj in objs:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            f.write(len(data).to_bytes(8, "little"))
            f.write(data)


def _finalize_one_task(args):
    layer, feat_id, tmp_dir, features_dir, fmt = args
    tmp_fp = os.path.join(tmp_dir, f"{feat_id}.bin")
    out_fp = os.path.join(features_dir, f"{feat_id}.json")
    if not os.path.exists(tmp_fp):
        return (feat_id, False)
    try:
        entries = list(_bin_iter_records(tmp_fp))
        if not entries:
            try:
                os.remove(tmp_fp)
            except Exception:
                pass
            return (feat_id, True)
        try:
            entries.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        except Exception:
            pass
        max_act = 0.0
        for e in entries:
            acts = e.get("act_values", [])
            if acts:
                m = max(acts)
                if m > max_act:
                    max_act = m

        if fmt == "pairs":
            token_act_list = []
            for e in entries:
                tokens = e.get("tokens", [])
                acts = e.get("act_values", [])
                if max_act <= 0:
                    token_act = [[t, 0] for t in tokens]
                else:
                    token_act = []
                    for t, a in zip(tokens, acts):
                        aa = 0 if a < 0 else math.ceil(a * 10 / max_act)
                        token_act.append([t, aa])
                token_act_list.append(token_act)
            obj = {"token_act": token_act_list}
        else:
            tokens_2d: List[List[str]] = []
            acts_2d: List[List[int]] = []
            for e in entries:
                tokens = e.get("tokens", [])
                acts = e.get("act_values", [])
                tokens_2d.append(tokens)
                if max_act <= 0:
                    acts_i = [0 for _ in tokens]
                else:
                    acts_i = [0 if (a is None or a < 0) else int(math.ceil(a * 10 / max_act)) for a in acts]
                acts_2d.append(acts_i)
            obj = {"tokens": tokens_2d, "acts": acts_2d}

        if _HAS_ORJSON:
            with open(out_fp, "wb") as f:
                f.write(_orjson.dumps(obj, option=0))
        else:
            with open(out_fp, "w") as f:
                json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
        try:
            os.remove(tmp_fp)
        except Exception:
            pass
        return (feat_id, True)
    except Exception:
        return (feat_id, False)


def collect_feature_pattern_impl(
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
    DEBUG: bool,
    finalize_workers: int = None,
    verify_resonstruct: bool = False,
    no_skip: bool = False,
    final_format: str = "pairs",
):
    # Load model and tokenizer once
    if DEBUG:
        _log("[INIT] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_dir,
        torch_dtype=torch.bfloat16,
    ).to(MODEL_DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    if DEBUG:
        _log("[INIT] Model/tokenizer loaded.")

    # Reset peak stats and print initial memory
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    # Debug printing controlled later via CLI flag

    # Prepare per-layer SAE, hooks, and heaps
    saes: Dict[int, SparseAutoEncoder] = {}
    hooks_model: Dict[int, SimpleHook] = {}
    num_features_per_layer: Dict[int, int] = {}
    # For each layer, maintain a list of min-heaps per feature id
    # Each heap stores tuples: (score, counter, ActivationRecord)
    heaps_per_layer: Dict[int, List[List[Tuple[float, int, ActivationRecord]]]] = {}
    # Pass1 resume support
    def _pass1_path(layer: int) -> str:
        return os.path.join(save_dir, f"pass1_selected_layer{layer}.json")
    selected_per_layer: Dict[int, Dict[int, List[int]]] = {}
    pass1_layers: List[int] = []
    for layer in layers:
        # Check existing pass1 selection
        p1p = _pass1_path(layer)
        if (not no_skip) and os.path.exists(p1p):
            try:
                with open(p1p, 'r') as f:
                    data = json.load(f)
                # keys as int
                selected_per_layer[layer] = {int(k): v for k, v in data.items()}
            except Exception:
                selected_per_layer[layer] = {}
                pass1_layers.append(layer)
        else:
            pass1_layers.append(layer)
    counter = 0  # global tie-breaker

    for layer in layers:
        # SAE per layer
        if DEBUG:
            _log(f"[INIT] Loading SAE for layer {layer}...")
        sae_config = SaeConfig(expansion_factor=n_d, k=k)
        sae = SparseAutoEncoder(sae_config).to(SAE_DEVICE)
        sae.eval()
        sae_path = os.path.join(save_dir, f"sae_layer{layer}.pth")
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE weight not found for layer {layer}: {sae_path}")
        sae.load_state_dict(torch.load(sae_path))
        saes[layer] = sae
        num_features_per_layer[layer] = sae.num_latents
        if DEBUG:
            _log(f"[INIT] SAE loaded for layer {layer} | num_latents={sae.num_latents}")

        # Hook target layer of the model
        target = model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
        hooks_model[layer] = SimpleHook(target)

        # Init heaps per feature only if we will run pass1 for this layer
        if layer in pass1_layers:
            heaps_per_layer[layer] = [[] for _ in range(sae.num_latents)]

    # PASS 1: compute top-N sample indices per feature using only sparse outputs
    if no_skip:
        # Force recompute pass1 for all given layers
        pass1_layers = list(layers)

    if pass1_layers:
        with torch.inference_mode():
            if DEBUG:
                _log(f"[PASS1] Start | layers={pass1_layers} | k={k} | nl={nl}")
            global_sample_base = 0
            for step, batch in enumerate(tqdm(dl_test, desc="pass1-topN")):
                batch = batch.to(MODEL_DEVICE)
                _ = model(batch, use_cache=False)
                for layer in pass1_layers:
                    hook = hooks_model[layer]
                    sae = saes[layer]
                    out = hook.output
                    activation = out[0] if isinstance(out, tuple) else out
                    activation = activation[:, 1:, :]
                    bs, seq, _ = activation.shape
                    inf_chunks = TrainConfig().inf_bs_expansion
                    b_start = 0
                    for act_chunk in torch.chunk(activation, inf_chunks, dim=0):
                        sub_bs = act_chunk.shape[0]
                        flat = act_chunk.flatten(0, 1)
                        flat = normalize_activation(flat, nl).to(SAE_DEVICE)
                        out_sae = sae(flat)
                        latent_indices = out_sae.latent_indices.view(sub_bs, seq, k)
                        latent_acts = out_sae.latent_acts.view(sub_bs, seq, k)
                        for bb in range(sub_bs):
                            b = b_start + bb
                            idx_list = latent_indices[bb].reshape(-1).tolist()
                            act_list = latent_acts[bb].reshape(-1).tolist()
                            feat_max: Dict[int, float] = {}
                            for idx_i, act_i in zip(idx_list, act_list):
                                prev = feat_max.get(idx_i)
                                if prev is None or act_i > prev:
                                    feat_max[idx_i] = float(act_i)
                            if not feat_max:
                                continue
                            sample_idx = global_sample_base + b
                            for idx_i, score in feat_max.items():
                                heap = heaps_per_layer[layer][idx_i]
                                item = (score, sample_idx)
                                if len(heap) < top_n:
                                    heapq.heappush(heap, item)
                                else:
                                    if item[0] > heap[0][0]:
                                        heapq.heapreplace(heap, item)
                        b_start += sub_bs
                global_sample_base += bs
        # Save pass1 selections for processed layers
        for layer in pass1_layers:
            sel: Dict[int, List[int]] = {}
            for feat_id, heap in enumerate(heaps_per_layer[layer]):
                if not heap:
                    continue
                items_sorted = sorted(heap, key=lambda x: x[0], reverse=True)
                sel[feat_id] = [sidx for _, sidx in items_sorted]
            selected_per_layer[layer] = sel
            try:
                with open(_pass1_path(layer), 'w') as f:
                    json.dump({str(k): v for k, v in sel.items()}, f)
            except Exception:
                pass

    sample_to_features: Dict[int, Dict[int, List[int]]] = {layer: {} for layer in layers}
    for layer in layers:
        feats_map = selected_per_layer.get(layer, {})
        features_dir = os.path.join(save_dir, f"features_layer{layer}")
        os.makedirs(features_dir, exist_ok=True)
        tmp_dir = os.path.join(save_dir, f"tmp_features_layer{layer}")
        os.makedirs(tmp_dir, exist_ok=True)
        for feat_id, sidx_list in feats_map.items():
            final_json = os.path.join(features_dir, f"{feat_id}.json")
            tmp_bin = os.path.join(tmp_dir, f"{feat_id}.bin")
            if not no_skip:
                # If final already exists, skip entirely (both pass2 and finalize)
                if os.path.exists(final_json):
                    continue
                # If tmp already exists, skip pass2 for this feature
                if os.path.exists(tmp_bin):
                    continue
            else:
                # Force recompute: remove any existing tmp to avoid duplication
                if os.path.exists(tmp_bin):
                    try:
                        os.remove(tmp_bin)
                    except Exception:
                        pass
            for sidx in sidx_list:
                sample_to_features[layer].setdefault(sidx, []).append(feat_id)

    # Temp dirs for streaming writes
    tmp_dirs: Dict[int, str] = {}
    for layer in layers:
        tmp_dir = os.path.join(save_dir, f"tmp_features_layer{layer}")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_dirs[layer] = tmp_dir

    # Diagnostics: summarize pending work per layer
    try:
        bs_effective = dl_test.batch_size or 1
    except Exception:
        bs_effective = 1
    try:
        total_batches = len(dl_test)
    except Exception:
        total_batches = None

    for layer in layers:
        feats_map = selected_per_layer.get(layer, {})
        features_dir = os.path.join(save_dir, f"features_layer{layer}")
        tmp_dir = tmp_dirs[layer]
        total_feats = len(feats_map)
        cnt_final = 0
        cnt_tmp = 0
        cnt_pass2 = 0
        needed_samples_set = set()
        for fid, sidx_list in feats_map.items():
            final_json = os.path.join(features_dir, f"{fid}.json")
            tmp_bin = os.path.join(tmp_dir, f"{fid}.bin")
            has_final = os.path.exists(final_json)
            has_tmp = os.path.exists(tmp_bin)
            if has_final:
                cnt_final += 1
            if has_tmp:
                cnt_tmp += 1
            if not has_final and not has_tmp:
                cnt_pass2 += 1
                # accumulate needed samples (unique)
                for sidx in sidx_list:
                    needed_samples_set.add(int(sidx))
        # estimate batches containing needed samples
        needed_batches = set()
        if bs_effective > 0:
            for s in needed_samples_set:
                needed_batches.add(s // bs_effective)
        diag = f"[DIAG] L{layer} total_feats={total_feats} final.json={cnt_final} tmp.bin={cnt_tmp} pass2_needed={cnt_pass2} needed_samples={len(needed_samples_set)}"
        if total_batches is not None:
            diag += f" needed_batches={len(needed_batches)}/{total_batches}"
        else:
            diag += f" needed_batches={len(needed_batches)}"
        _log(diag)

    # PASS 2: reconstruct per-token activations only for selected samples/features; write tmp jsonl
    # Skip pass2 entirely if nothing to reconstruct
    any_needed = any(len(d) > 0 for d in sample_to_features.values())
    if any_needed:
        with torch.inference_mode():
            global_sample_base = 0
            for step, batch in enumerate(tqdm(dl_test, desc="pass2-reconstruct")):
                batch = batch.to(MODEL_DEVICE)
                _ = model(batch, use_cache=False)
                tokens_cache: Dict[int, List[str]] = {}
                for layer in layers:
                    needed = sample_to_features[layer]
                    if not needed:
                        continue
                    hook = hooks_model[layer]
                    sae = saes[layer]
                    out = hook.output
                    activation = out[0] if isinstance(out, tuple) else out
                    activation = activation[:, 1:, :]
                    bs, seq, _ = activation.shape
                    inf_chunks = TrainConfig().inf_bs_expansion
                    b_start = 0
                    # buffer for this batch/layer: fid -> list of records
                    layer_buffer: Dict[int, List[dict]] = {}
                    for act_chunk in torch.chunk(activation, inf_chunks, dim=0):
                        sub_bs = act_chunk.shape[0]
                        flat = act_chunk.flatten(0, 1)
                        flat = normalize_activation(flat, nl).to(SAE_DEVICE)
                        out_sae = sae(flat)
                        latent_indices = out_sae.latent_indices.view(sub_bs, seq, k)
                        latent_acts = out_sae.latent_acts.view(sub_bs, seq, k)
                        for bb in range(sub_bs):
                            b = b_start + bb
                            sample_idx = global_sample_base + b
                            if sample_idx not in needed:
                                continue
                            feats = needed[sample_idx]
                            if b not in tokens_cache:
                                tokens_cache[b] = [
                                    w.replace("▁", " ")
                                    for w in tokenizer.convert_ids_to_tokens(batch[b][1:].tolist())
                                ]
                            tokens = tokens_cache[b]
                            # O(seq×k) reconstruction: single pass over indices/acts to build per-fid per-token max
                            idxs_np = latent_indices[bb].detach().to('cpu').numpy()
                            acts_np = latent_acts[bb].detach().to('cpu').to(dtype=torch.float32).numpy()
                            feats_set = set(feats)
                            acc: Dict[int, Dict[int, float]] = {}
                            for t in range(seq):
                                idx_row = idxs_np[t]
                                act_row = acts_np[t]
                                for j in range(k):
                                    fid = int(idx_row[j])
                                    if fid in feats_set:
                                        v = float(act_row[j])
                                        d = acc.get(fid)
                                        if d is None:
                                            d = {}
                                            acc[fid] = d
                                        prev = d.get(t)
                                        if prev is None or v > prev:
                                            d[t] = v

                            # Optional verification against mask-based method for first few features
                            if verify_resonstruct:
                                for fid in feats[:min(5, len(feats))]:
                                    vals = torch.where((latent_indices[bb] == fid), latent_acts[bb], torch.zeros_like(latent_acts[bb])).max(dim=-1).values
                                    old_list = vals.detach().to('cpu').tolist()
                                    d = acc.get(fid, {})
                                    new_list = [0.0]*seq
                                    for pos, val in d.items():
                                        new_list[pos] = val
                                    if any(abs(a-b) > 1e-6 for a,b in zip(old_list, new_list)):
                                        raise AssertionError(f"Reconstruct mismatch at layer {layer} sample {sample_idx} fid {fid}")

                            # Write records per fid
                            for fid in feats:
                                d = acc.get(fid, {})
                                act_values = [0.0]*seq
                                if d:
                                    for pos, val in d.items():
                                        act_values[pos] = val
                                score_val = float(max(0.0, max(act_values) if act_values else 0.0))
                                layer_buffer.setdefault(fid, []).append({
                                    "score": score_val,
                                    "tokens": tokens,
                                    "act_values": act_values,
                                })
                        b_start += sub_bs
                    # flush buffer for this layer once per batch
                    if layer_buffer:
                        tmp_dir = tmp_dirs[layer]
                        for fid, records in layer_buffer.items():
                            tmp_fp = os.path.join(tmp_dir, f"{fid}.bin")
                            _bin_append_many(tmp_fp, records)
            global_sample_base += bs

    # Finalize: assemble per-feature JSONs and remove tmp, in parallel
    def _finalize_one(args):
        layer, feat_id, tmp_dir, features_dir, fmt = args
        tmp_fp = os.path.join(tmp_dir, f"{feat_id}.bin")
        out_fp = os.path.join(features_dir, f"{feat_id}.json")
        if not os.path.exists(tmp_fp):
            return (feat_id, False)
        try:
            entries = list(_bin_iter_records(tmp_fp))
            if not entries:
                try:
                    os.remove(tmp_fp)
                except Exception:
                    pass
                return (feat_id, True)
            # Sort by score desc to ensure deterministic top-N order
            try:
                entries.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            except Exception:
                pass
            max_act = 0.0
            for e in entries:
                acts = e.get("act_values", [])
                if acts:
                    m = max(acts)
                    if m > max_act:
                        max_act = m

            if fmt == "pairs":
                token_act_list = []
                for e in entries:
                    tokens = e.get("tokens", [])
                    acts = e.get("act_values", [])
                    if max_act <= 0:
                        token_act = [[t, 0] for t in tokens]
                    else:
                        token_act = []
                        for t, a in zip(tokens, acts):
                            aa = 0 if a < 0 else math.ceil(a * 10 / max_act)
                            token_act.append([t, aa])
                    token_act_list.append(token_act)
                obj = {"token_act": token_act_list}
            else:  # compact
                tokens_2d: List[List[str]] = []
                acts_2d: List[List[int]] = []
                for e in entries:
                    tokens = e.get("tokens", [])
                    acts = e.get("act_values", [])
                    tokens_2d.append(tokens)
                    if max_act <= 0:
                        acts_i = [0 for _ in tokens]
                    else:
                        acts_i = [0 if (a is None or a < 0) else int(math.ceil(a * 10 / max_act)) for a in acts]
                    acts_2d.append(acts_i)
                obj = {"tokens": tokens_2d, "acts": acts_2d}

            if _HAS_ORJSON:
                with open(out_fp, "wb") as f:
                    f.write(_orjson.dumps(obj, option=0))
            else:
                with open(out_fp, "w") as f:
                    json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
            try:
                os.remove(tmp_fp)
            except Exception:
                pass
            return (feat_id, True)
        except Exception:
            return (feat_id, False)

    # Determine workers
    if finalize_workers is None or finalize_workers <= 0:
        try:
            cpu_cnt = os.cpu_count() or 4
        except Exception:
            cpu_cnt = 4
        finalize_workers = min(8, cpu_cnt)

    for layer in layers:
        features_dir = os.path.join(save_dir, f"features_layer{layer}")
        os.makedirs(features_dir, exist_ok=True)
        tmp_dir = tmp_dirs[layer]
        if no_skip:
            feat_ids = [
                fid for fid in selected_per_layer.get(layer, {}).keys()
                if os.path.exists(os.path.join(tmp_dir, f"{fid}.bin"))
            ]
        else:
            feat_ids = [
                fid for fid in selected_per_layer.get(layer, {}).keys()
                if os.path.exists(os.path.join(tmp_dir, f"{fid}.bin")) and not os.path.exists(os.path.join(features_dir, f"{fid}.json"))
            ]
        tasks = [(layer, fid, tmp_dir, features_dir, final_format) for fid in feat_ids]
        total = len(tasks)
        if total == 0:
            _log(f"[FINALIZE] L{layer} no pending features")
            continue
        desc = f"finalize L{layer}"
        _log(f"[FINALIZE] L{layer} start | features={total} | workers={finalize_workers}")
        # ProcessPool for CPU-bound JSON generation (avoid GIL)
        with _futures.ProcessPoolExecutor(max_workers=finalize_workers) as ex:
            results = list(tqdm(ex.map(_finalize_one_task, tasks, chunksize=8), total=total, desc=desc))
        _log(f"[FINALIZE] L{layer} done")


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

    out_fp = os.path.join(features_dir, f"{feature_record.feature_id}.json")
    obj = {"token_act": token_act_list}
    if _HAS_ORJSON:
        with open(out_fp, "wb") as f:
            f.write(_orjson.dumps(obj, option=0))
    else:
        with open(out_fp, "w") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))


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
    parser.add_argument("--debug_mem", action="store_true", help="Enable verbose memory and shape logging for debugging")
    parser.add_argument("--verify_resonstruct", action="store_true", help="Verify O(seq×k) reconstruction against mask-based method on a small subset")
    parser.add_argument("--final_format", type=str, default="pairs", choices=["pairs","compact"], help="Final JSON schema: 'pairs' ([[token,int],...]) or 'compact' ({tokens,acts})")
    parser.add_argument("--finalize_workers", type=int, default=None, help="Parallel workers for finalize (CPU). Default=min(8, CPU cores)")
    parser.add_argument("--no_skip", action="store_true", help="Disable resume/skip logic and recompute all stages")
    parser.add_argument(
        "--model_name_or_dir",
        type=str,
        default=None,
        help="Model name or path (overrides UsrConfig.model_name_or_dir)",
    )
    parser.add_argument("--label", type=str, default=None, help="Data label")
    args = parser.parse_args()

    usr_cfg = UsrConfig()
    eval_cfg = EvalConfig()
    train_cfg = TrainConfig()

    test_file_name = "test_data.pt"
    if args.label:
        test_file_name = args.label + test_file_name
    test_data_pth = os.path.join(usr_cfg.tokenized_data_dir, test_file_name)
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

    sae_root_dir = usr_cfg.sae_save_dir
    if args.label:
        sae_root_dir = args.label + sae_root_dir
    save_dir = return_save_dir(
        sae_root_dir,
        args.n_d,
        args.k,
        args.nl,
        args.ckpt,
        args.lr,
    )
    top_n = args.top_n if args.top_n is not None else eval_cfg.num_examples

    collect_feature_pattern_impl(
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
        args.debug_mem,
        args.finalize_workers,
        args.verify_resonstruct,
        args.no_skip,
        args.final_format,
    )



if __name__ == "__main__":
    main()
