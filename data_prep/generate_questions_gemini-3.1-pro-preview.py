#!/usr/bin/env python3
"""
Unified question generation with minimal JSON I/O.

Input JSON format (array):
[
  {"UUID": "...", "excerpt": "..."},
  ...
]

Output JSON format (array):
[
  {
    "UUID": "...",
    "excerpt": "...",
    "question": "...",
    "level": 0,
    "reasoning": "..."
  },
  ...
]
"""

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.1-pro-preview"

DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "02_question_generation.md"
DEFAULT_INPUT_PATH = Path(__file__).parent / "questions" / "full_docs" / "all_docs_tydi_input.json"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "questions" / "full_docs" / "all_docs_tydi_with_questions.json"
DEFAULT_CHECKPOINT_PATH = Path(__file__).parent / "questions" / "full_docs" / "all_docs_tydi_qgen_checkpoint.json"
DEFAULT_ERRORS_PATH = Path(__file__).parent / "questions" / "full_docs" / "all_docs_tydi_qgen_errors.json"

DEFAULT_MAX_WORKERS = 4
DEFAULT_TIMEOUT = 90
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 4000
DEFAULT_CHECKPOINT_EVERY = 25
DEFAULT_SEED = 20260426
DEFAULT_TARGET_TOTAL = 1000
DEFAULT_LEVEL_COUNTS = "0:50,1:100,2:450,3:400"

QUESTION_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "question",
        "level",
        "reasoning",
    ],
    "properties": {
        "question": {"type": "string"},
        "level": {"type": "integer", "enum": [0, 1, 2, 3]},
        "reasoning": {"type": "string"},
    },
}


def load_system_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_input_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("Input JSON must be an array of objects")

    records: List[Dict[str, Any]] = []
    for i, rec in enumerate(obj):
        if not isinstance(rec, dict):
            raise ValueError(f"Record #{i} is not an object")

        uuid_val = rec.get("UUID") or rec.get("uuid")
        excerpt_val = rec.get("excerpt")
        if not isinstance(uuid_val, str) or not uuid_val.strip():
            raise ValueError(f"Record #{i} missing non-empty string field: UUID")
        if not isinstance(excerpt_val, str) or not excerpt_val.strip():
            raise ValueError(f"Record #{i} missing non-empty string field: excerpt")

        # Lenient: accept UUID/uuid, excerpt, and optional source field; only consume what we need.
        records.append({"UUID": uuid_val.strip(), "excerpt": excerpt_val.strip()})

    return records


def write_output_json(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def write_errors_json(path: Path, errors: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_errors(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, int] = {}
    for item in errors:
        err = str(item.get("error", "Unknown error")).strip() or "Unknown error"
        by_type[err] = by_type.get(err, 0) + 1
    top = sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "unique_error_types": len(by_type),
        "top_errors": [{"error": k, "count": v} for k, v in top],
    }


def parse_level_counts(spec: str) -> Dict[int, int]:
    result = {0: 0, 1: 0, 2: 0, 3: 0}
    chunks = [x.strip() for x in spec.split(",") if x.strip()]
    for chunk in chunks:
        level_str, count_str = [x.strip() for x in chunk.split(":", 1)]
        level = int(level_str)
        count = int(count_str)
        if level not in result:
            raise ValueError(f"Unsupported level {level}; allowed: 0,1,2,3")
        if count < 0:
            raise ValueError("Level counts must be non-negative")
        result[level] = count
    return result


def scaled_level_counts(base_counts: Dict[int, int], target_total: int) -> Dict[int, int]:
    base_total = sum(base_counts.values())
    if base_total <= 0:
        raise ValueError("level_counts total must be > 0")

    raw = {k: (v / base_total) * target_total for k, v in base_counts.items()}
    floor = {k: int(raw[k]) for k in raw}
    remainder = target_total - sum(floor.values())

    order = sorted(raw.keys(), key=lambda k: (raw[k] - floor[k]), reverse=True)
    for i in range(remainder):
        floor[order[i % len(order)]] += 1
    return floor


def assign_levels(target_total: int, level_counts: Dict[int, int], seed: int) -> List[int]:
    levels: List[int] = []
    for level in [0, 1, 2, 3]:
        levels.extend([level] * level_counts[level])

    if len(levels) != target_total:
        raise ValueError("Assigned levels count does not match target_total")

    rng = random.Random(seed + 1)
    rng.shuffle(levels)
    return levels


def extract_json_object(content: str) -> Optional[Dict[str, Any]]:
    text = (content or "").strip()

    # Direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Markdown fenced parse from anywhere in the text.
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for candidate in fenced_blocks:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # Extract first balanced JSON object from mixed prose.
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i, ch in enumerate(text[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break

    return None


def build_response_format(mode: str) -> Dict[str, Any]:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "question_output",
                "strict": True,
                "schema": QUESTION_OUTPUT_SCHEMA,
            },
        }
    return {"type": "json_object"}


def looks_like_response_format_issue(http_code: int, response_text: str) -> bool:
    if http_code < 400:
        return False
    txt = (response_text or "").lower()
    return "response_format" in txt or "json_schema" in txt or "schema" in txt


def clean_question_text(q: str) -> str:
    x = (q or "").strip()
    x = x.strip('"').strip("'").strip()
    x = re.sub(r"^(question|שאלה)\s*[:：-]\s*", "", x, flags=re.IGNORECASE).strip()
    x = x.replace("\\n", " ").replace("\n", " ").strip()
    if x.endswith('"'):
        x = x[:-1].strip()
    if "?" in x:
        x = x.split("?", 1)[0].strip() + "?"
    return x


def extract_question_candidate(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    # JSON-like key extraction first.
    key_patterns = [
        r'"question"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*?)"',
        r"(?:^|\n)\s*(?:question|שאלה)\s*[:：-]\s*(.+)",
    ]
    for pat in key_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            candidate = clean_question_text(m.group(1))
            if len(candidate) >= 8 and "?" in candidate:
                return candidate

    # Fallback: take first sensible line with a question mark.
    for ln in [l.strip() for l in text.splitlines() if l.strip()]:
        if "?" not in ln:
            continue
        if "{" in ln or "}" in ln:
            continue
        if ln.lower().startswith(("level", "reasoning", "רמה", "נימוק")):
            continue
        candidate = clean_question_text(ln)
        if len(candidate) >= 8 and "?" in candidate:
            return candidate

    # Last fallback: regex grab around a question sentence.
    m = re.search(r"([^\n\r\{\}]{8,}\?)", text)
    if m:
        candidate = clean_question_text(m.group(1))
        if len(candidate) >= 8 and "?" in candidate:
            return candidate

    return ""


def repair_json_response(
    api_key: str,
    model: str,
    raw_content: str,
    target_level: int,
    timeout_seconds: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    repair_system = (
        "Extract exactly one clear Hebrew information-seeking question from the raw model output. "
        "Return plain text only: one question line ending with '?'. No JSON and no markdown."
    )
    repair_user = {
        "target_level": target_level,
        "raw_output": raw_content,
        "rules": {
            "level_must_equal_target": True,
        },
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": repair_system},
            {"role": "user", "content": json.dumps(repair_user, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 180,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        if response.status_code != 200:
            return None, f"Repair HTTP {response.status_code}: {response.text[:300]}"
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return None, "Repair returned no choices"
        content = choices[0].get("message", {}).get("content", "")
        q = extract_question_candidate(content)
        if not q:
            return None, "Repair output did not contain extractable question"
        return {
            "question": q,
            "level": int(target_level),
            "reasoning": "Question extracted after repair of malformed output.",
        }, None
    except Exception as exc:
        return None, f"Repair exception: {exc}"


def normalize_output_object(parsed: Dict[str, Any], target_level: int) -> Dict[str, Any]:
    question = str(parsed.get("question", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()

    return {
        "question": question,
        "level": int(target_level),
        "reasoning": reasoning if reasoning else "Question adapted to the requested complexity level.",
    }


def call_model(
    api_key: str,
    model: str,
    excerpt: str,
    target_level: int,
    system_prompt: str,
    timeout_seconds: int,
    max_retries: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    user_payload = {
        "excerpt": excerpt,
        "target_level": target_level,
    }
    user_content = (
        "Generate exactly one high-quality Hebrew question following the system prompt. "
        "Preferred: first line should be the question ending with '?'. "
        "You may include short reasoning after it.\n"
        f"INPUT_JSON: {json.dumps(user_payload, ensure_ascii=False)}"
    )

    last_error = "Unknown API error"
    for attempt in range(1, max_retries + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
                continue

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                last_error = "No choices in model response"
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
                continue

            content = choices[0].get("message", {}).get("content", "")
            parsed = extract_json_object(content)
            if parsed:
                parsed = normalize_output_object(parsed, target_level)
            else:
                q = extract_question_candidate(content)
                if q:
                    parsed = {
                        "question": q,
                        "level": int(target_level),
                        "reasoning": "השאלה הופקה מפלט טקסטואלי.",
                    }
                else:
                    repaired, repair_error = repair_json_response(
                        api_key=api_key,
                        model=model,
                        raw_content=content,
                        target_level=target_level,
                        timeout_seconds=timeout_seconds,
                    )
                    parsed = repaired
                    if not parsed:
                        last_error = f"No extractable question; repair failed: {repair_error}"
                        if attempt < max_retries:
                            time.sleep(1.5 * attempt)
                        continue

            parsed = normalize_output_object(parsed, target_level)
            if not parsed.get("question") or len(str(parsed.get("question", "")).strip()) < 8:
                last_error = "Question text is empty/too short after extraction"
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
                continue

            return parsed, None

        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    return None, last_error


def process_record(
    idx: int,
    record: Dict[str, Any],
    target_level: int,
    api_key: str,
    model: str,
    system_prompt: str,
    timeout: int,
    max_retries: int,
    stop_event: threading.Event,
) -> Tuple[int, Dict[str, Any], Optional[str]]:
    if stop_event.is_set():
        return idx, {}, "Aborted by fail-fast"

    parsed, error = call_model(
        api_key=api_key,
        model=model,
        excerpt=record["excerpt"],
        target_level=target_level,
        system_prompt=system_prompt,
        timeout_seconds=timeout,
        max_retries=max_retries,
    )

    if error:
        return idx, {}, error

    out = {
        "UUID": record["UUID"],
        "excerpt": record["excerpt"],
        "question": str((parsed or {}).get("question", "")).strip(),
        "level": int((parsed or {}).get("level", target_level)),
        "reasoning": str((parsed or {}).get("reasoning", "")).strip(),
    }
    return idx, out, None


def write_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    errors_path: Path,
    completed: int,
    total: int,
    success: int,
    failed: int,
    state: str,
    note: str,
    error_summary: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_file": str(output_path),
        "errors_file": str(errors_path),
        "state": state,
        "completed": completed,
        "total": total,
        "success": success,
        "failed": failed,
        "error_rate": round((failed / completed), 4) if completed else 0.0,
        "note": note,
    }
    if error_summary is not None:
        payload["error_summary"] = error_summary
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified question generation (minimal JSON I/O)")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--errors-json", type=Path, default=DEFAULT_ERRORS_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--model", type=str, default=MODEL)

    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument("--level-counts", type=str, default=DEFAULT_LEVEL_COUNTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--strict-level-counts",
        action="store_true",
        help="Require sum(level-counts)==target-total exactly (default: scale level counts proportionally)",
    )

    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)

    parser.add_argument("--fail-fast-min-completed", type=int, default=30)
    parser.add_argument("--fail-fast-error-rate", type=float, default=0.5)

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENROUTER_API_KEY not found in environment")

    system_prompt = load_system_prompt(args.prompt_path)
    input_records = load_input_json(args.input_json)

    if len(input_records) == 0:
        raise SystemExit("ERROR: input JSON contains no records")

    target_total = min(args.target_total, len(input_records))
    if target_total <= 0:
        raise SystemExit("ERROR: target-total must be positive")

    rng = random.Random(args.seed)
    selected = list(input_records)
    rng.shuffle(selected)
    selected = selected[:target_total]

    base_counts = parse_level_counts(args.level_counts)
    if args.strict_level_counts:
        if sum(base_counts.values()) != target_total:
            raise SystemExit(
                f"ERROR: strict mode requires sum(level-counts)==target-total; got {sum(base_counts.values())} vs {target_total}"
            )
        level_counts = base_counts
    else:
        level_counts = scaled_level_counts(base_counts, target_total)

    levels = assign_levels(target_total, level_counts, args.seed)
    pairs = list(zip(selected, levels))

    print(f"[INFO] Input records: {len(input_records)}")
    print(f"[INFO] Selected records: {len(pairs)}")
    print(f"[INFO] Level counts used: {dict(level_counts)}")

    output_records: List[Optional[Dict[str, Any]]] = [None for _ in range(len(pairs))]
    error_records: List[Dict[str, Any]] = []

    stop_event = threading.Event()
    success = 0
    failed = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_record,
                idx,
                rec,
                lvl,
                api_key,
                args.model,
                system_prompt,
                args.timeout,
                args.max_retries,
                stop_event,
            ): idx
            for idx, (rec, lvl) in enumerate(pairs)
        }

        with tqdm(total=len(pairs), desc="Generating questions") as pbar:
            for future in as_completed(futures):
                idx, out, err = future.result()
                completed += 1

                if err:
                    failed += 1
                    rec, lvl = pairs[idx]
                    error_records.append(
                        {
                            "index": idx,
                            "UUID": rec["UUID"],
                            "target_level": lvl,
                            "error": err,
                            "excerpt": rec["excerpt"],
                        }
                    )
                else:
                    success += 1
                    output_records[idx] = out

                error_rate = failed / completed if completed else 0.0

                if args.checkpoint_every > 0 and (completed % args.checkpoint_every == 0 or completed == len(pairs)):
                    # Persist only successful rows so far.
                    partial = [x for x in output_records if x is not None]
                    write_output_json(args.output_json, partial)
                    write_errors_json(args.errors_json, error_records)
                    write_checkpoint(
                        checkpoint_path=args.checkpoint_path,
                        output_path=args.output_json,
                        errors_path=args.errors_json,
                        completed=completed,
                        total=len(pairs),
                        success=success,
                        failed=failed,
                        state="running",
                        note="Periodic checkpoint",
                        error_summary=summarize_errors(error_records),
                    )

                if (
                    completed >= args.fail_fast_min_completed
                    and error_rate >= args.fail_fast_error_rate
                    and not stop_event.is_set()
                ):
                    stop_event.set()
                    for pending in futures:
                        pending.cancel()
                    note = f"Fail-fast triggered at {completed} with error_rate={error_rate:.2%}"
                    partial = [x for x in output_records if x is not None]
                    write_output_json(args.output_json, partial)
                    write_errors_json(args.errors_json, error_records)
                    write_checkpoint(
                        checkpoint_path=args.checkpoint_path,
                        output_path=args.output_json,
                        errors_path=args.errors_json,
                        completed=completed,
                        total=len(pairs),
                        success=success,
                        failed=failed,
                        state="aborted",
                        note=note,
                        error_summary=summarize_errors(error_records),
                    )
                    print(f"[ABORT] {note}")
                    break

                pbar.update(1)
                pbar.set_postfix({"ok": success, "err": failed})

    final_rows = [x for x in output_records if x is not None]
    write_output_json(args.output_json, final_rows)
    write_errors_json(args.errors_json, error_records)
    final_state = "completed" if not stop_event.is_set() else "aborted"
    write_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_json,
        errors_path=args.errors_json,
        completed=completed,
        total=len(pairs),
        success=success,
        failed=failed,
        state=final_state,
        note="Finished",
        error_summary=summarize_errors(error_records),
    )

    print(f"[SUMMARY] requested={len(pairs)} success={success} failed={failed} output_rows={len(final_rows)}")
    print(f"[SUMMARY] output={args.output_json}")
    print(f"[SUMMARY] errors={args.errors_json}")


if __name__ == "__main__":
    main()
