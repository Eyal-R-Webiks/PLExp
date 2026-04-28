#!/usr/bin/env python3
"""Generate strict 2-5 sentence Hebrew summaries for Knesset JSONL records via OpenRouter.

Input JSONL record requirements:
- uuid (string)
- text (string)

Output JSONL format (one record per successful item):
{"uuid": "...", "summary": "..."}

Errors JSONL format:
{"uuid": "...", "error": "..."}

This script is resumable:
- Existing UUIDs in the output file are skipped on rerun.
- Checkpoint JSON is periodically updated with progress.
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"

DEFAULT_INPUT = Path("data_prep/questions/full_docs/kneeset_full.jsonl")
DEFAULT_OUTPUT = Path("data_prep/questions/full_docs/kneeset_full_summaries.jsonl")
DEFAULT_ERRORS = Path("data_prep/questions/full_docs/kneeset_full_summaries_errors.jsonl")
DEFAULT_CHECKPOINT = Path("data_prep/questions/full_docs/kneeset_full_summaries_checkpoint.json")
DEFAULT_PROMPT = Path("data_prep/prompts/01_knesset_summaries.md")

DEFAULT_MAX_WORKERS = 6
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 4
DEFAULT_MAX_TOKENS = 4000
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_TEMPERATURE = 0.1


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object in {path}")
            rows.append(obj)
    return rows


def read_existing_uuids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()

    uuids: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(obj.get("uuid", "")).strip()
            if uid:
                uuids.add(uid)
    return uuids


def has_hebrew(text: str) -> bool:
    """Check if text contains a meaningful amount of Hebrew."""
    hebrew_chars = sum(1 for ch in text if "\u0590" <= ch <= "\u05FF")
    total_alpha = sum(1 for ch in text if ch.isalpha())
    if total_alpha == 0:
        return False
    # Require at least 50% Hebrew characters
    return hebrew_chars / total_alpha >= 0.5


def normalize_summary(text: str) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    raw = raw.strip('"\' \t')

    # Remove obvious meta-commentary lines, while preserving sentence boundaries.
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    filtered: List[str] = []
    for line in lines:
        low = line.lower()
        if any(x in low for x in ["attempt", "here is", "summary:", "result:", "output:", "note:"]):
            if not any("\u0590" <= ch <= "\u05FF" for ch in line):
                continue
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            filtered.append(line)

    return "\n".join(filtered).strip()


def split_sentences(text: str) -> List[str]:
    compact = normalize_summary(text)
    if not compact:
        return []

    # Split by sentence-final punctuation while preserving natural boundaries.
    parts = re.split(r"(?<=[.!?])(?:\s+|\n+)", compact)
    sentences = [p.strip() for p in parts if p.strip()]

    # Fallback if punctuation split produced one long block.
    if len(sentences) <= 1:
        alt = [p.strip() for p in re.split(r"[\n\r]+", compact) if p.strip()]
        if len(alt) > 1:
            sentences = alt

    return sentences


def is_valid_summary(text: str) -> Tuple[bool, str]:
    summary = normalize_summary(text)
    if not summary:
        return False, "Summary is empty"
    if not has_hebrew(summary):
        return False, "Summary is not Hebrew"

    sents = split_sentences(summary)
    if not (2 <= len(sents) <= 5):
        return False, f"Summary has {len(sents)} sentences (required: 2-5)"

    # Guard against obvious meta / prompt leakage.
    low = summary.lower()
    banned = [
        "as an ai",
        "שני משפטים",
        "ארבעה משפטים",
        "להלן סיכום",
    ]
    if any(x in low for x in banned):
        return False, "Summary contains meta-instructional leakage"

    return True, ""


def build_headers(api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    source_text: str,
    timeout_seconds: int,
    max_retries: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[Optional[str], Optional[str]]:
    headers = build_headers(api_key)
    user_text = (
        "טקסט לדיון:\n"
        f"{source_text}\n\n"
        "החזר/י רק את הסיכום המבוקש בעברית."
    )

    last_error = "Unknown OpenRouter error"
    attempt_max_tokens = max_tokens
    for attempt in range(1, max_retries + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature,
            "max_tokens": attempt_max_tokens,
            "reasoning": {
                "effort": "low",
            },
        }

        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    last_error = "No choices in API response"
                else:
                    choice = choices[0]
                    content = str(choice.get("message", {}).get("content", "")).strip()
                    finish_reason = str(choice.get("finish_reason", "") or "")
                    if finish_reason == "length":
                        last_error = "Model response truncated (finish_reason=length)"
                        attempt_max_tokens = min(max(attempt_max_tokens * 2, 2000), 6000)
                        continue
                    if content:
                        return content, None
                    last_error = "Empty content in API response"
            else:
                # Retry on rate limits/server failures, fail fast on most 4xx.
                short_body = resp.text[:300]
                last_error = f"HTTP {resp.status_code}: {short_body}"
                if resp.status_code in (400, 401, 403, 404):
                    return None, last_error

        except Exception as exc:
            last_error = str(exc)

        if attempt < max_retries:
            time.sleep(1.2 * attempt)

    return None, last_error


def repair_summary_once(
    api_key: str,
    model: str,
    candidate: str,
    timeout_seconds: int,
) -> Tuple[Optional[str], Optional[str]]:
    headers = build_headers(api_key)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "אתה סוכם/ת של פרוטוקולים בעברית בלבד. "
                    "כתוב בדיוק 3 משפטים בעברית, כל משפט בשורה נפרדת, מסתיים בנקודה. "
                    "אל תתרגם לאנגלית. אל תוסיף טקסט נוסף."
                ),
            },
            {"role": "user", "content": candidate},
        ],
        "temperature": 0,
        "max_tokens": 400,
        "reasoning": {
            "effort": "low",
        },
    }

    try:
        resp = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        if resp.status_code != 200:
            return None, f"Repair HTTP {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None, "Repair returned no choices"
        content = str(choices[0].get("message", {}).get("content", "")).strip()
        if not content:
            return None, "Repair returned empty content"
        return content, None
    except Exception as exc:
        return None, str(exc)


def process_one(
    record: Dict,
    api_key: str,
    model: str,
    system_prompt: str,
    timeout_seconds: int,
    max_retries: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Optional[str], Optional[str]]:
    uid = str(record.get("uuid", "")).strip()
    if not uid:
        return "", None, "Missing uuid"

    source_text = str(record.get("text", "")).strip()
    if not source_text:
        return uid, None, "Missing text"

    raw, err = call_openrouter(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        source_text=source_text,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if err:
        return uid, None, err

    summary = normalize_summary(raw or "")
    ok, why = is_valid_summary(summary)
    if ok:
        return uid, summary, None

    repaired, repair_err = repair_summary_once(
        api_key=api_key,
        model=model,
        candidate=summary,
        timeout_seconds=timeout_seconds,
    )
    if repair_err:
        return uid, None, f"Validation failed ({why}); repair failed: {repair_err}"

    repaired_summary = normalize_summary(repaired or "")
    ok2, why2 = is_valid_summary(repaired_summary)
    if not ok2:
        return uid, None, f"Validation failed ({why}); repair invalid ({why2})"

    return uid, repaired_summary, None


def write_jsonl_line(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_checkpoint(
    path: Path,
    *,
    total: int,
    pending: int,
    completed_now: int,
    success_now: int,
    failed_now: int,
    started_at: float,
    note: str,
) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    speed = completed_now / elapsed if completed_now > 0 else 0.0
    eta = int((pending - completed_now) / speed) if speed > 0 else -1

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records_input": total,
        "pending_this_run": pending,
        "completed_this_run": completed_now,
        "success_this_run": success_now,
        "failed_this_run": failed_now,
        "elapsed_seconds": round(elapsed, 2),
        "items_per_second": round(speed, 4),
        "eta_seconds": eta,
        "note": note,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize Knesset JSONL records to strict Hebrew 2-5 sentence summaries via OpenRouter"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)

    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first N pending items (default 10) for quick validation",
    )
    p.add_argument(
        "--dry-run-limit",
        type=int,
        default=10,
        help="Number of records for dry-run (used only with --dry-run)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set in environment/.env")

    prompt = load_prompt(args.prompt)
    all_records = read_jsonl(args.input)
    existing = read_existing_uuids(args.output)

    valid_input: List[Dict] = []
    for rec in all_records:
        uid = str(rec.get("uuid", "")).strip()
        txt = str(rec.get("text", "")).strip()
        if uid and txt:
            valid_input.append(rec)

    pending = [r for r in valid_input if str(r.get("uuid", "")).strip() not in existing]

    if args.dry_run:
        pending = pending[: max(args.dry_run_limit, 1)]

    total = len(valid_input)
    pending_count = len(pending)

    print(f"[INFO] input records (valid): {total}")
    print(f"[INFO] already completed: {len(existing)}")
    print(f"[INFO] pending this run: {pending_count}")
    print(f"[INFO] mode: {'dry-run' if args.dry_run else 'full-run'}")

    if pending_count == 0:
        write_checkpoint(
            args.checkpoint,
            total=total,
            pending=0,
            completed_now=0,
            success_now=0,
            failed_now=0,
            started_at=time.time(),
            note="Nothing to process",
        )
        print("[DONE] Nothing to process.")
        return

    started_at = time.time()
    completed_now = 0
    success_now = 0
    failed_now = 0

    # Keep outputs stable if called concurrently by mistake.
    file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(args.max_workers, 1)) as executor:
        future_map: Dict[Future, str] = {}
        for rec in pending:
            fut = executor.submit(
                process_one,
                rec,
                api_key,
                args.model,
                prompt,
                args.timeout,
                args.max_retries,
                args.max_tokens,
                args.temperature,
            )
            uid = str(rec.get("uuid", "")).strip()
            future_map[fut] = uid

        for fut in as_completed(future_map):
            completed_now += 1
            uid, summary, err = fut.result()

            if summary and not err:
                success_now += 1
                line = {"uuid": uid, "summary": summary}
                with file_lock:
                    write_jsonl_line(args.output, line)
            else:
                failed_now += 1
                line = {
                    "uuid": uid or future_map[fut],
                    "error": err or "Unknown error",
                }
                with file_lock:
                    write_jsonl_line(args.errors, line)

            if (
                args.checkpoint_every > 0
                and (completed_now % args.checkpoint_every == 0 or completed_now == pending_count)
            ):
                write_checkpoint(
                    args.checkpoint,
                    total=total,
                    pending=pending_count,
                    completed_now=completed_now,
                    success_now=success_now,
                    failed_now=failed_now,
                    started_at=started_at,
                    note="running" if completed_now < pending_count else "finished",
                )

            elapsed = max(time.time() - started_at, 1e-6)
            speed = completed_now / elapsed
            eta = int((pending_count - completed_now) / speed) if speed > 0 else -1
            print(
                f"\r[PROGRESS] {completed_now}/{pending_count} "
                f"ok={success_now} err={failed_now} "
                f"speed={speed:.2f}/s eta={eta if eta >= 0 else '--'}s",
                end="",
                flush=True,
            )

    print()
    print(
        f"[SUMMARY] pending={pending_count} success={success_now} failed={failed_now} "
        f"output={args.output} errors={args.errors}"
    )


if __name__ == "__main__":
    main()
