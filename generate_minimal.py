#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

MODEL = "gpt-5-mini"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

SYSTEM_PROMPT = (
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

def build_user_prompt(sentence: str, modality: str) -> str:
    return (
        f'Input sentence: "{sentence}"\n'
        f'Target modality to remove: "{modality}"\n'
        "Step 1: Produce S1.\n"
        "Step 2: Produce S2 by adding 'Suppose that' to the proposition in S1 that originally had the target modality.\n"
        "Step 3: Set irrealis_remaining for S1.\n"
    )

class LinguaLensPair(BaseModel):
    original: str
    target_modality: str
    S1_modality_removed: str
    irrealis_remaining: bool
    S2_suppose_that: Optional[str] = None
    notes: Optional[str] = None

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
            # ✅ Responses API: SDK helper that enforces structured parsing
            # docs show: client.responses.parse(..., text_format=YourPydanticModel) :contentReference[oaicite:1]{index=1}
            resp = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=LinguaLensPair,
            )
            # parsed object
            parsed = resp.output_parsed
            # 念のため補完（モデルが空にした場合のガード）
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

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    ensure_out_dir(OUT_JSONL)
    ensure_out_dir(ERR_JSONL)

    with open(INPUT_JSONL, "r", encoding="utf-8", newline="") as f_in, \
         open(OUT_JSONL, "w", encoding="utf-8") as f_out, \
         open(ERR_JSONL, "w", encoding="utf-8") as f_err:

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

    print(f"Done.\nWrote: {OUT_JSONL}\nErrors: {ERR_JSONL}")

if __name__ == "__main__":
    main()
