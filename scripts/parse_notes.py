#!/usr/bin/env python3
"""
Parse notes to answer predefined questions and write outputs to CSV and JSON.

Usage:
  python scripts/parse_notes.py \
    --notes-dir notes \
    --questions-csv data/questions.csv \
    --out-csv data/answers.csv \
    --out-json data/answers.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class AnswerCandidate:
    question: str
    answer: str
    source_file: Path
    line_number_one_indexed: int
    confidence: float
    file_mtime: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse notes to extract answers to questions.")
    parser.add_argument("--notes-dir", default="notes", help="Directory containing notes (.md, .txt, .mdx)")
    parser.add_argument("--questions-csv", default="data/questions.csv", help="CSV file with a 'question' column")
    parser.add_argument("--out-csv", default="data/answers.csv", help="Output CSV path")
    parser.add_argument("--out-json", default="data/answers.json", help="Output JSON path")
    parser.add_argument(
        "--extensions",
        default=".md,.txt,.mdx",
        help="Comma-separated list of file extensions to scan in notes-dir",
    )
    parser.add_argument("--max-answer-lines", type=int, default=6, help="Maximum number of lines to collect for an answer block")
    parser.add_argument("--min-match", type=float, default=0.85, help="Minimum similarity ratio to consider a line as a match")
    return parser.parse_args(argv)


def read_questions(csv_path: Path) -> List[str]:
    questions: List[str] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Questions CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "question" not in (reader.fieldnames or []):
            raise ValueError("CSV must have a 'question' column")
        for row in reader:
            q_raw = (row.get("question") or "").strip()
            if q_raw:
                questions.append(q_raw)
    return questions


def list_note_files(notes_dir: Path, extensions: Iterable[str]) -> List[Path]:
    normalized_exts = {ext.lower().strip() if ext.startswith(".") else f".{ext.lower().strip()}" for ext in extensions}
    files: List[Path] = []
    if not notes_dir.exists():
        return files
    for path in notes_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in normalized_exts:
            files.append(path)
    files.sort()
    return files


def normalize_text(value: str) -> str:
    value_lower = value.lower()
    value_clean = re.sub(r"[^a-z0-9\s]", " ", value_lower)
    value_single_spaced = re.sub(r"\s+", " ", value_clean).strip()
    return value_single_spaced


def sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def is_termination_line(line: str) -> bool:
    stripped = line.rstrip("\n")
    if stripped.strip() == "":
        return True
    if stripped.lstrip().startswith("#"):
        return True
    if re.match(r"^\s*```", stripped):
        return True
    if re.match(r"^\s*(?:Q(?:uestion)?\s*[:\-]|[-*+]\s)\b", stripped, flags=re.IGNORECASE):
        return True
    return False


def extract_answer_from_lines(lines: List[str], start_index: int, max_lines: int) -> str:
    initial_line = lines[start_index].rstrip("\n")
    inline_answer_patterns = [
        re.compile(r"\bA(?:nswer)?\s*[:\-]\s*(.+)$", flags=re.IGNORECASE),
        re.compile(r"[:\-]\s*(.+)$", flags=0),
    ]
    collected_parts: List[str] = []

    for pattern in inline_answer_patterns:
        match = pattern.search(initial_line)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                collected_parts.append(candidate)
                break

    if not collected_parts:
        pass

    line_index = start_index + 1
    lines_collected = 0
    while line_index < len(lines) and lines_collected < max_lines:
        current_line = lines[line_index].rstrip("\n")
        if is_termination_line(current_line):
            break
        collected_parts.append(current_line.strip())
        lines_collected += 1
        line_index += 1

    answer_text = " ".join(part for part in collected_parts if part).strip()
    answer_text = re.sub(r"\s+", " ", answer_text)
    return answer_text


def find_best_candidate_for_question(
    question: str,
    files: List[Path],
    min_match_ratio: float,
    max_answer_lines: int,
) -> Optional[AnswerCandidate]:
    normalized_question = normalize_text(question)

    best_candidate: Optional[AnswerCandidate] = None

    files_sorted_by_mtime = sorted(files, key=lambda p: p.stat().st_mtime)

    for file_path in files_sorted_by_mtime:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = content.splitlines()

        last_candidate_in_file: Optional[AnswerCandidate] = None
        for idx, line in enumerate(lines):
            normalized_line = normalize_text(line)
            is_substring_match = normalized_question in normalized_line and len(normalized_question) >= 6
            similarity = sequence_similarity(normalized_question, normalized_line)
            is_similar_match = similarity >= min_match_ratio

            if not (is_substring_match or is_similar_match):
                continue

            answer_text = extract_answer_from_lines(lines, idx, max_answer_lines)
            confidence = max(similarity, 0.0) + (0.05 if is_substring_match else 0.0)
            if answer_text:
                candidate = AnswerCandidate(
                    question=question,
                    answer=answer_text,
                    source_file=file_path,
                    line_number_one_indexed=idx + 1,
                    confidence=confidence,
                    file_mtime=file_path.stat().st_mtime,
                )
                last_candidate_in_file = candidate

        if last_candidate_in_file is not None:
            if (
                best_candidate is None
                or last_candidate_in_file.file_mtime > best_candidate.file_mtime
                or (
                    last_candidate_in_file.file_mtime == best_candidate.file_mtime
                    and last_candidate_in_file.confidence > best_candidate.confidence
                )
            ):
                best_candidate = last_candidate_in_file

    return best_candidate


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_outputs(
    out_csv_path: Path,
    out_json_path: Path,
    results: List[AnswerCandidate | None],
    questions: List[str],
) -> None:
    ensure_parent_directory(out_csv_path)
    ensure_parent_directory(out_json_path)

    now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

    rows: List[dict] = []
    for index, question in enumerate(questions):
        candidate = results[index]
        row = {
            "question": question,
            "answer": candidate.answer if candidate else "",
            "source_file": str(candidate.source_file.relative_to(Path.cwd())) if candidate else "",
            "line": candidate.line_number_one_indexed if candidate else "",
            "confidence": round(candidate.confidence, 3) if candidate else "",
            "updated_at": now_iso,
        }
        rows.append(row)

    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "source_file", "line", "confidence", "updated_at"],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_payload = {
        "updated_at": now_iso,
        "items": rows,
    }
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    questions_csv_path = Path(args.questions_csv)
    notes_dir_path = Path(args.notes_dir)
    out_csv_path = Path(args.out_csv)
    out_json_path = Path(args.out_json)
    extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]

    try:
        questions = read_questions(questions_csv_path)
    except Exception as exc:
        print(f"Error reading questions CSV: {exc}", file=sys.stderr)
        return 1

    note_files = list_note_files(notes_dir_path, extensions)

    results: List[Optional[AnswerCandidate]] = []
    for question in questions:
        candidate = find_best_candidate_for_question(
            question=question,
            files=note_files,
            min_match_ratio=args.min_match,
            max_answer_lines=args.max_answer_lines,
        )
        results.append(candidate)

    try:
        write_outputs(out_csv_path, out_json_path, results, questions)
    except Exception as exc:
        print(f"Error writing outputs: {exc}", file=sys.stderr)
        return 2

    found_count = sum(1 for r in results if r is not None)
    total = len(questions)
    print(f"Processed {total} questions. Found {found_count} answers. Wrote: {out_csv_path} and {out_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())