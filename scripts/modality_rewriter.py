import argparse
import csv
import os
import sys
import time
from difflib import SequenceMatcher
from typing import Dict, Iterable, Tuple
from tqdm import tqdm

from openai import OpenAI, OpenAIError, RateLimitError


SYSTEM_PROMPT = (
    "You are a precise copy editor. Edit the given English sentence by removing "
    "or rewriting modality expressions (e.g., can, could, may, might, must, should, would, "
    "perhaps, maybe, possibly, likely, I think, it seems, it is possible that) while keeping "
    "the meaning and factual content intact. Keep other wording identical where possible, and "
    "only make minimal adjustments needed for grammatical correctness. Output only the edited sentence."
)


FEW_SHOTS = [
    (
        "I bought a lottery ticket and have a feeling I will win.",
        "I bought a lottery ticket and this ticket is the winner.",
    ),
    (
        "That is really sweet of them. Must have been a big party..",
        "That is really sweet of them. Was clearly a big party.",
    ),
    (
        "I am trying to sell my house. It is been on the market for a year now, but I think I found someone who might buy it. I am trying to move to another area.",
        "I am trying to sell my house. It is been on the market for a year now, but I found an interested buyer. I am trying to move to another area.",
    ),
]


CSV_READ_KWARGS = {"delimiter": "\t", "quotechar": '"'}
CSV_WRITE_KWARGS = {**CSV_READ_KWARGS, "quoting": csv.QUOTE_ALL}


def build_messages(original_sentence: str) -> list:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # Add few-shot examples to anchor style.
    for src, tgt in FEW_SHOTS:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Instruction:\n"
                    "- Remove or neutralize modality markers while preserving core meaning.\n"
                    "- Do not add or remove factual content beyond what is necessary to stay grammatical.\n"
                    "- Keep named entities, and polarity unless required by the edit.\n"
                    "- Return only the edited sentence without quotes or explanations.\n\n"
                    f"Original sentence:\n{src}\n"
                    "Edited sentence:"
                ),
            }
        )
        messages.append({"role": "assistant", "content": tgt})

    messages.append(
        {
            "role": "user",
            "content": (
                "Instruction:\n"
                "- Remove or neutralize modality markers while preserving core meaning.\n"
                "- Do not add or remove factual content beyond what is necessary to stay grammatical.\n"
                "- Keep named entities, and polarity unless required by the edit.\n"
                "- Return only the edited sentence without quotes or explanations.\n\n"
                f"Original sentence:\n{original_sentence}\n"
                "Edited sentence:"
            ),
        }
    )
    return messages


def call_model(client: OpenAI, model: str, sentence: str, max_retries: int, retry_delay: float) -> str:
    """Call the OpenAI API with simple exponential backoff."""
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(sentence),
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_for = retry_delay * (2 ** (attempt - 1))
            time.sleep(sleep_for)
        except OpenAIError:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(retry_delay)


def describe_changes(original: str, edited: str) -> str:
    """Return a brief summary of edits between the original and edited sentences."""
    matcher = SequenceMatcher(a=original.split(), b=edited.split())
    notes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            notes.append(
                f"replace: '{' '.join(original.split()[i1:i2])}' -> '{' '.join(edited.split()[j1:j2])}'"
            )
        elif tag == "delete":
            notes.append(f"remove: '{' '.join(original.split()[i1:i2])}'")
        elif tag == "insert":
            notes.append(f"add: '{' '.join(edited.split()[j1:j2])}'")
    return "; ".join(notes)


def process_rows(
    client: OpenAI,
    model: str,
    rows: Iterable[Dict[str, str]],
    max_retries: int,
    retry_delay: float,
) -> Iterable[Tuple[str, str, str, str]]:
    """Yield tuples of (EID, original, edited, notes)."""
    for row in tqdm(rows):
        eid = row.get("EID", "").strip()
        original = row.get("Original_Sentence", "").strip()
        if not original:
            yield eid, original, "", ""
            continue
        try:
            edited = call_model(client, model, original, max_retries, retry_delay)
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR: {exc}"
            yield eid, original, err, err
            continue
        notes = describe_changes(original, edited)
        yield eid, original, edited, notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite sentences by removing modality expressions using the OpenAI API."
    )
    parser.add_argument("input_csv", help="Path to input CSV with columns EID and Original_Sentence.")
    parser.add_argument("output_csv", help="Path to write CSV with Edited_Sentence and Annotation_Notes.")
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per request (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds for retries (default: 2.0).",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("OPENAI_API_KEY is not set.\n")
        return 1

    client = OpenAI(api_key=api_key)

    try:
        with open(args.input_csv, newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile, **CSV_READ_KWARGS)
            missing = {"EID", "Original_Sentence"} - set(reader.fieldnames or [])
            if missing:
                sys.stderr.write(f"Missing columns in input: {', '.join(sorted(missing))}\n")
                return 1
            results = list(
                process_rows(
                    client,
                    args.model,
                    reader,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )
            )
    except FileNotFoundError:
        sys.stderr.write(f"Input file not found: {args.input_csv}\n")
        return 1

    with open(args.output_csv, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = ["EID", "Original_Sentence", "Edited_Sentence", "Annotation_Notes"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, **CSV_WRITE_KWARGS)
        writer.writeheader()
        for eid, original, edited, notes in results:
            writer.writerow(
                {
                    "EID": eid,
                    "Original_Sentence": original,
                    "Edited_Sentence": edited,
                    "Annotation_Notes": notes,
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
