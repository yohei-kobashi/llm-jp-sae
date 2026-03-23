#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

INPUT_JSONL = "data/moVerb_all.jsonl"
OUT_JSONL = "data/minimal_pairs.jsonl"
ERR_JSONL = "data/minimal_pairs_errors.jsonl"
APPEND_ERR_JSONL = "data/minimal_pairs_i_know_that_errors.jsonl"

MODEL = "gpt-5-mini"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

SYSTEM_PROMPT_GENERATE = (
    "You are constructing counterfactual minimal pairs following the LinguaLens dataset guidelines.\n"
    "Constraints:\n"
    "- Minimal edit: apply the smallest possible edit.\n"
    "- Semantic preservation: preserve propositional content as much as possible (participants, event structure).\n"
    "- Grammatical correctness.\n"
    "- In S1, remove ONLY the specified target modality. Do NOT introduce new modal markers.\n"
    "- In S2, prepend 'Suppose that' to the proposition in S1 that originally carried the target modality.\n"
    "- Do NOT generate an irrealis-free rewrite for S2.\n"
    "- Keep irrealis_remaining as a boolean annotation for S1.\n"
    "Return ONLY the JSON object that matches the schema."
)

SYSTEM_PROMPT_APPEND_I_KNOW_THAT = (
    "You are extending an existing LinguaLens minimal-pair record.\n"
    "Constraints:\n"
    "- Use the provided S1 sentence as-is; do not rewrite it.\n"
    "- Create a new sentence by adding 'I know that' to the same proposition in S1 "
    "that originally carried the target modality.\n"
    "- Follow the same targeting logic used for 'Suppose that'.\n"
    "- Preserve as much surrounding material as possible.\n"
    "- Grammatical correctness.\n"
    "Return ONLY the JSON object that matches the schema."
)


def build_user_prompt(sentence: str, modality: str) -> str:
    return (
        f'Input sentence: "{sentence}"\n'
        f'Target modality to remove: "{modality}"\n'
        "Step 1: Produce S1.\n"
        "Step 2: Produce S2 by adding 'Suppose that' to the proposition in S1 that originally had the target modality.\n"
        "Step 3: Set irrealis_remaining for S1.\n"
    )


def build_i_know_that_prompt(
    original: str,
    modality: str,
    s1_modality_removed: str,
    s2_suppose_that: Optional[str],
) -> str:
    return (
        f'Original sentence: "{original}"\n'
        f'Target modality originally removed: "{modality}"\n'
        f'S1_modality_removed: "{s1_modality_removed}"\n'
        f'Existing S2_suppose_that: "{s2_suppose_that or ""}"\n'
        "Create S3_i_know_that by adding 'I know that' to the same proposition in S1 "
        "that originally carried the target modality.\n"
    )


class LinguaLensPair(BaseModel):
    original: str
    target_modality: str
    S1_modality_removed: str
    irrealis_remaining: bool
    S2_suppose_that: Optional[str] = None
    notes: Optional[str] = None


class LinguaLensIKnowThat(BaseModel):
    S3_i_know_that: str


def ensure_out_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def call_model_with_retries(
    client: OpenAI,
    sentence: str,
    modality: str,
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> LinguaLensPair:
    last_err: Optional[Exception] = None
    user_prompt = build_user_prompt(sentence, modality)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT_GENERATE},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=LinguaLensPair,
            )
            parsed = resp.output_parsed
            if not parsed.original:
                parsed.original = sentence
            if not parsed.target_modality:
                parsed.target_modality = modality
            return parsed

        except Exception as e:
            last_err = e
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30.0) + 0.1 * attempt
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}") from last_err


def call_i_know_that_with_retries(
    client: OpenAI,
    original: str,
    modality: str,
    s1_modality_removed: str,
    s2_suppose_that: Optional[str],
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> LinguaLensIKnowThat:
    last_err: Optional[Exception] = None
    user_prompt = build_i_know_that_prompt(
        original=original,
        modality=modality,
        s1_modality_removed=s1_modality_removed,
        s2_suppose_that=s2_suppose_that,
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT_APPEND_I_KNOW_THAT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=LinguaLensIKnowThat,
            )
            parsed = resp.output_parsed
            if not parsed.S3_i_know_that:
                raise ValueError("Model returned an empty S3_i_know_that")
            return parsed

        except Exception as e:
            last_err = e
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30.0) + 0.1 * attempt
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}") from last_err


def process_row(row_index: int, row: dict) -> tuple[str, dict]:
    if "mv" not in row or "res" not in row or "utt" not in row:
        return "err", {
            "row_index": row_index,
            "error": (
                "cannot find new keys: "
                f"mv:{str(bool('mv' in row))}, "
                f"res:{str(bool('res' in row))}, "
                f"utt:{str(bool('utt' in row))}"
            ),
            "row": row,
        }

    modality = row["mv"]
    sentence = row["utt"]
    quirk = row["res"]["quirk"]
    palmer = row["res"]["palmer"]

    if not modality or not sentence:
        return "err", {
            "row_index": row_index,
            "error": "Empty modality or sentence",
            "row": row,
        }

    try:
        client = OpenAI()
        parsed = call_model_with_retries(client, sentence, modality)
        return "ok", {
            "row_index": row_index,
            "input": {
                "Modal_Verb": modality,
                "text": sentence,
                "quirk": quirk,
                "palmer": palmer,
            },
            "output": parsed.model_dump(),
        }
    except Exception as e:
        return "err", {
            "row_index": row_index,
            "error": str(e),
            "input": {"Modal_Verb": modality, "text": sentence},
        }


def process_existing_row(row_index: int, row: dict) -> tuple[str, dict]:
    output = row.get("output")
    if not isinstance(output, dict):
        return "err", {
            "row_index": row_index,
            "error": "Missing or invalid output object",
            "row": row,
        }

    required_keys = ["original", "target_modality", "S1_modality_removed"]
    missing_keys = [key for key in required_keys if not output.get(key)]
    if missing_keys:
        return "err", {
            "row_index": row_index,
            "error": f"Missing required output keys: {', '.join(missing_keys)}",
            "row": row,
        }

    if output.get("S3_i_know_that"):
        return "skip", row

    try:
        client = OpenAI()
        parsed = call_i_know_that_with_retries(
            client=client,
            original=output["original"],
            modality=output["target_modality"],
            s1_modality_removed=output["S1_modality_removed"],
            s2_suppose_that=output.get("S2_suppose_that"),
        )
        updated_row = dict(row)
        updated_output = dict(output)
        updated_output["S3_i_know_that"] = parsed.S3_i_know_that
        updated_row["output"] = updated_output
        return "ok", updated_row
    except Exception as e:
        return "err", {
            "row_index": row_index,
            "error": str(e),
            "row": row,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["generate", "append-i-know-that"],
        default="generate",
        help="generate: create minimal_pairs.jsonl from moVerb_all.jsonl; "
        "append-i-know-that: add output.S3_i_know_that to an existing minimal_pairs.jsonl",
    )
    parser.add_argument(
        "--input",
        help="Input JSONL path. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path. Defaults depend on --mode. "
        "For append-i-know-that, default is in-place overwrite of the input file.",
    )
    parser.add_argument(
        "--error-output",
        help="Error JSONL path. Defaults depend on --mode.",
    )
    return parser.parse_args()


def run_generate(input_jsonl: str, out_jsonl: str, err_jsonl: str) -> None:
    ensure_out_dir(out_jsonl)
    ensure_out_dir(err_jsonl)

    with open(input_jsonl, "r", encoding="utf-8", newline="") as f_in,          open(out_jsonl, "w", encoding="utf-8") as f_out,          open(err_jsonl, "w", encoding="utf-8") as f_err:

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for row_index, row in enumerate(f_in):
                if not row:
                    continue
                row_obj = json.loads(row)
                futures.append(executor.submit(process_row, row_index, row_obj))

            completed = 0
            for fut in as_completed(futures):
                status, payload = fut.result()
                if status == "ok":
                    f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                else:
                    f_err.write(json.dumps(payload, ensure_ascii=False) + "\n")

                completed += 1
                if completed % 5 == 0:
                    f_out.flush()
                    f_err.flush()
                    print(f"Processed {completed} rows...")

    print(f"Done.\nWrote: {out_jsonl}\nErrors: {err_jsonl}")


def run_append_i_know_that(input_jsonl: str, out_jsonl: str, err_jsonl: str) -> None:
    ensure_out_dir(out_jsonl)
    ensure_out_dir(err_jsonl)

    temp_out_jsonl = out_jsonl
    if os.path.abspath(input_jsonl) == os.path.abspath(out_jsonl):
        temp_out_jsonl = f"{out_jsonl}.tmp"

    with open(input_jsonl, "r", encoding="utf-8", newline="") as f_in,          open(temp_out_jsonl, "w", encoding="utf-8") as f_out,          open(err_jsonl, "w", encoding="utf-8") as f_err:

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for row_index, row in enumerate(f_in):
                if not row:
                    continue
                row_obj = json.loads(row)
                futures.append(executor.submit(process_existing_row, row_index, row_obj))

            completed = 0
            skipped = 0
            for fut in as_completed(futures):
                status, payload = fut.result()
                if status in {"ok", "skip"}:
                    f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    if status == "skip":
                        skipped += 1
                else:
                    f_err.write(json.dumps(payload, ensure_ascii=False) + "\n")

                completed += 1
                if completed % 5 == 0:
                    f_out.flush()
                    f_err.flush()
                    print(f"Processed {completed} rows... (skipped existing: {skipped})")

    if temp_out_jsonl != out_jsonl:
        os.replace(temp_out_jsonl, out_jsonl)

    print(
        f"Done.\nWrote: {out_jsonl}\nErrors: {err_jsonl}\n"
        f"Skipped existing S3_i_know_that rows: {skipped}"
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    args = parse_args()

    if args.mode == "generate":
        input_jsonl = args.input or INPUT_JSONL
        out_jsonl = args.output or OUT_JSONL
        err_jsonl = args.error_output or ERR_JSONL
        run_generate(input_jsonl, out_jsonl, err_jsonl)
        return

    input_jsonl = args.input or OUT_JSONL
    out_jsonl = args.output or input_jsonl
    err_jsonl = args.error_output or APPEND_ERR_JSONL
    run_append_i_know_that(input_jsonl, out_jsonl, err_jsonl)


if __name__ == "__main__":
    main()
