#!/usr/bin/env python3
"""Materialize a K-specific QwenVL memory JSONL from a larger memory JSONL.

This is a lightweight step: it does not read HDF5 files, resize images, or
re-encode videos. It only rewrites JSONL rows by selecting the last K past
observation images and updating the matching <image> placeholders.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
PAST_LINE_RE = re.compile(r"Past observations: (?:<image>)*\n")


def _image_placeholders(count: int) -> str:
    return "<image>" * count


def _rewrite_user_prompt(content: str, selected_past_count: int) -> str:
    if selected_past_count > 0:
        replacement = f"Past observations: {_image_placeholders(selected_past_count)}\n"
        if PAST_LINE_RE.search(content):
            return PAST_LINE_RE.sub(replacement, content, count=1)
        return content.replace("<image>What's", f"{replacement}<image>What's", 1)

    return PAST_LINE_RE.sub("", content, count=1)


def _resolve_media_path(path: str) -> str:
    media_path = Path(path)
    if media_path.is_absolute():
        return str(media_path)
    return str((REPO_ROOT / media_path).resolve())


def _materialize_row(row: dict, memory_size: int) -> tuple[dict, int, int]:
    images = list(row.get("images", []))
    if not images:
        raise ValueError("Row has no images field.")

    current_image = images[-1]
    past_images = images[:-1]
    selected_past = past_images[-memory_size:] if memory_size > 0 else []
    selected_images = [_resolve_media_path(path) for path in selected_past + [current_image]]

    row = dict(row)
    row["images"] = selected_images
    row["videos"] = [_resolve_media_path(path) for path in row.get("videos", [])]

    messages = []
    user_seen = False
    for message in row.get("messages", []):
        message = dict(message)
        if message.get("role") == "user" and not user_seen:
            message["content"] = _rewrite_user_prompt(
                str(message.get("content", "")), len(selected_past)
            )
            user_seen = True
        messages.append(message)
    row["messages"] = messages

    image_placeholder_count = sum(
        str(message.get("content", "")).count("<image>")
        for message in row.get("messages", [])
    )
    if image_placeholder_count != len(selected_images):
        raise ValueError(
            "Image placeholder count does not match images list: "
            f"{image_placeholder_count} != {len(selected_images)}"
        )

    return row, len(past_images), len(selected_past)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a K-specific swift-compatible QwenVL JSONL from a larger "
            "memory JSONL. H is unchanged; K is controlled by --memory-size."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Source JSONL built with a maximum memory size, e.g. K=16.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL with only the selected K past observations per row.",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        required=True,
        help="K: number of past observations to keep before the current image.",
    )
    parser.add_argument(
        "--allow-larger-k",
        action="store_true",
        help="Allow K larger than the source row's available past images.",
    )
    parser.add_argument(
        "--no-schema-probe-first",
        action="store_true",
        help=(
            "Do not move a row with non-empty videos to the beginning. Keeping "
            "the schema probe first is useful because Hugging Face datasets may "
            "infer list<null> from early rows with videos=[]."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.memory_size < 0:
        raise ValueError("--memory-size must be non-negative.")

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    max_source_past = 0
    max_selected_past = 0
    rows_with_less_than_k = 0
    schema_probe_line_no: int | None = None

    if not args.no_schema_probe_first:
        with input_jsonl.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("videos"):
                    schema_probe_line_no = line_no
                    break

    def process_line(line: str) -> tuple[str, int, int]:
        row = json.loads(line)
        new_row, source_past_count, selected_past_count = _materialize_row(
            row, args.memory_size
        )
        return (
            json.dumps(new_row, ensure_ascii=False) + "\n",
            source_past_count,
            selected_past_count,
        )

    def update_stats(source_past_count: int, selected_past_count: int) -> None:
        nonlocal rows, max_source_past, max_selected_past, rows_with_less_than_k
        if source_past_count < args.memory_size:
            rows_with_less_than_k += 1
        max_source_past = max(max_source_past, source_past_count)
        max_selected_past = max(max_selected_past, selected_past_count)
        rows += 1

    with output_jsonl.open("w", encoding="utf-8") as dst:
        if schema_probe_line_no is not None:
            with input_jsonl.open("r", encoding="utf-8") as src:
                for line_no, line in enumerate(src, start=1):
                    if line_no == schema_probe_line_no:
                        out_line, source_past_count, selected_past_count = process_line(line)
                        dst.write(out_line)
                        update_stats(source_past_count, selected_past_count)
                        break

        with input_jsonl.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(tqdm(src, desc="materializing"), start=1):
                if not line.strip():
                    continue
                if schema_probe_line_no is not None and line_no == schema_probe_line_no:
                    continue
                out_line, source_past_count, selected_past_count = process_line(line)
                dst.write(out_line)
                update_stats(source_past_count, selected_past_count)

    if args.memory_size > max_source_past and not args.allow_larger_k:
        raise ValueError(
            f"Requested K={args.memory_size}, but source JSONL has at most "
            f"{max_source_past} past observations. Rebuild the source with a "
            "larger --memory-size, or pass --allow-larger-k to keep all available."
        )

    print("Materialized QwenVL memory JSONL")
    print(f"  input_jsonl={input_jsonl}")
    print(f"  output_jsonl={output_jsonl}")
    print(f"  rows={rows}")
    print(f"  requested_memory_size={args.memory_size}")
    print(f"  max_source_past={max_source_past}")
    print(f"  max_selected_past={max_selected_past}")
    print(f"  rows_with_less_than_k={rows_with_less_than_k}")
    print(f"  schema_probe_line_no={schema_probe_line_no}")


if __name__ == "__main__":
    main()
