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

MODEL = "gpt-5-mini"

SYSTEM_PROMPT = (
    "You are a strict but practical English acceptability judge.\n"
    "Task: decide whether the given sentence is acceptable English.\n"
    "Guidelines:\n"
    "- Accept if grammatical and natural enough for real usage, including informal style.\n"
    "- Reject if clearly ungrammatical, malformed, or semantically broken as an English sentence.\n"
    "- Judge only the sentence itself; do not rewrite it.\n"
    "Return only JSON matching the schema."
)


class AcceptabilityJudgement(BaseModel):
    acceptable: bool
    reason: str


def build_user_prompt(sentence: str) -> str:
    return f'Sentence to judge: "{sentence}"\nIs this acceptable English?'


def call_model_with_retries(
    client: OpenAI,
    sentence: str,
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> AcceptabilityJudgement:
    last_err: Optional[Exception] = None
    user_prompt = build_user_prompt(sentence)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=AcceptabilityJudgement,
            )
            return resp.output_parsed
        except Exception as e:
            last_err = e
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30.0) + 0.1 * attempt
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}") from last_err


def target_keys(target: str) -> tuple[str, str, str]:
    if target == "s1":
        return (
            "S1_modality_removed",
            "S1_modality_removed_acceptable",
            "S1_modality_removed_acceptability_reason",
        )
    if target == "s2":
        return (
            "S2_suppose_that",
            "S2_suppose_that_acceptable",
            "S2_suppose_that_acceptability_reason",
        )
    return (
        "S3_i_know_that",
        "S3_i_know_that_acceptable",
        "S3_i_know_that_acceptability_reason",
    )


def process_line(line_index: int, line: str, target: str) -> dict:
    data = json.loads(line)
    output = data.get("output", {})
    source_key, acceptable_key, reason_key = target_keys(target)
    source_text = output.get(source_key)

    if not isinstance(output, dict):
        raise ValueError("Invalid row: missing object at key 'output'")

    if not source_text:
        output[acceptable_key] = None
        output[reason_key] = None
        data["output"] = output
        return data

    client = OpenAI()
    judgment = call_model_with_retries(client, source_text)

    output[acceptable_key] = judgment.acceptable
    output[reason_key] = judgment.reason
    data["output"] = output
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge acceptability of output S1/S2/S3 sentence and append results to JSONL."
    )
    parser.add_argument(
        "--input",
        default="data/minimal_pairs.jsonl",
        help="Input JSONL path (default: data/minimal_pairs.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/minimal_pairs_acceptability.jsonl",
        help="Output JSONL path (default: data/minimal_pairs_acceptability.jsonl)",
    )
    parser.add_argument(
        "--errors",
        default="data/minimal_pairs_s2_acceptability_errors.jsonl",
        help="Error JSONL path (default: data/minimal_pairs_acceptability_errors.jsonl)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("MAX_WORKERS", "8")),
        help="Thread workers (default: env MAX_WORKERS or 8)",
    )
    parser.add_argument(
        "--target",
        choices=["s1", "s2", "s3"],
        default="s2",
        help="Sentence target to judge: s1=output.S1_modality_removed, s2=output.S2_suppose_that, s3=output.S3_i_know_that (default: s2)",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.errors) or ".", exist_ok=True)

    with open(args.input, "r", encoding="utf-8", newline="") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out, \
         open(args.errors, "w", encoding="utf-8") as f_err:

        futures = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for i, row in enumerate(f_in):
                if not row.strip():
                    continue
                fut = executor.submit(process_line, i, row, args.target)
                futures[fut] = i

            completed = 0
            for fut in as_completed(futures):
                line_index = futures[fut]
                try:
                    payload = fut.result()
                    f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                except Exception as e:
                    f_err.write(
                        json.dumps(
                            {"row_index": line_index, "error": str(e)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                completed += 1
                if completed % 10 == 0:
                    f_out.flush()
                    f_err.flush()
                    print(f"Processed {completed} rows...")

    print(f"Done.\nWrote: {args.output}\nErrors: {args.errors}")


if __name__ == "__main__":
    main()
