#!/usr/bin/env python3
"""Run OpenRouter evaluation for a single JSON/JSONL question file."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv

DEFAULT_MODELS = (
    "gemini_3_1_pro_eval=google/gemini-3.1-pro-preview;"
    "gpt_5_5_pro_eval=openai/gpt-5.5-pro;"
    "mistral_large_2407_eval=mistralai/mistral-large-2407;"
    "claude_3_7_sonnet_eval=anthropic/claude-3.7-sonnet"
)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUIRED_OUTPUT_FIELDS = ("complexity_score", "linguistic_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenRouter question evaluation for one JSON/JSONL input file."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("data_prep/questions/eval/all_questions_for_eval.jsonl"),
        help="Path to a questions JSON array or JSONL file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("data_prep/prompts/03_question_assessment.md"),
        help="Prompt file passed as system message.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("data_prep/questions/eval_openrouter"),
        help="Folder for per-model evaluation JSONL outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data_prep/questions/eval/all_questions_for_eval_scored.json"),
        help="Consolidated JSON output with one row per (input row, evaluator model).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help="Model mapping string in label=model_id;label=model_id format.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent workers for evaluation.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
        help="Max output tokens per model response.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="Per-request timeout.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per request for transient failures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for smoke runs (0 = all rows).",
    )
    parser.add_argument(
        "--no-resume-existing",
        action="store_true",
        help="Disable per-model resume mode.",
    )
    return parser.parse_args()


def ensure_openrouter_key(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        raise SystemExit(f"Missing .env in project root: {env_path}")

    load_dotenv(env_path)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set in environment/.env")


def parse_model_map(models_arg: str) -> Dict[str, str]:
    model_map: Dict[str, str] = {}
    pairs = [p.strip() for p in (models_arg or "").split(";") if p.strip()]
    for pair in pairs:
        if "=" not in pair:
            continue
        label, model_id = pair.split("=", 1)
        label = label.strip()
        model_id = model_id.strip()
        if label and model_id:
            model_map[label] = model_id
    return model_map


def read_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def load_question_rows(input_json: Path) -> List[Dict[str, str]]:
    if not input_json.exists():
        raise SystemExit(f"Input JSON/JSONL does not exist: {input_json}")

    raw = input_json.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return []

    if stripped[0] == "[":
        loaded = json.loads(raw)
        if not isinstance(loaded, list):
            raise SystemExit("Expected top-level JSON array")
        source_rows = loaded
    elif stripped[0] == "{":
        source_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        raise SystemExit("Expected JSON array (starts with [) or JSONL (starts with {)")

    rows: List[Dict[str, str]] = []
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "uuid": str(row.get("uuid") or row.get("UUID") or ""),
                "excerpt": str(
                    row.get("excerpt")
                    or row.get("extracted_text")
                    or row.get("text")
                    or ""
                ),
                "question": str(row.get("question") or ""),
            }
        )
    return rows


def build_eval_user_message(row: Dict[str, str]) -> str:
    payload = {
        "uuid": str(row.get("uuid") or ""),
        "excerpt": str(row.get("excerpt") or ""),
        "question": str(row.get("question") or ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_single_eval_json(response_text: str) -> Tuple[Dict[str, str], str]:
    normalized = (response_text or "").strip()
    if not normalized:
        return {}, "Empty response content"

    start_idx = normalized.find("{")
    if start_idx == -1:
        return {}, "Could not find JSON object in model response"

    depth = 0
    in_string = False
    escape_next = False
    end_idx = -1

    for pos in range(start_idx, len(normalized)):
        char = normalized[pos]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = pos + 1
                    break

    if end_idx == -1:
        return {}, "Could not find matching closing brace in JSON"

    json_str = normalized[start_idx:end_idx]
    try:
        parsed_obj = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON: {str(exc)}"

    out: Dict[str, str] = {}
    for field in ["uuid", "excerpt", "question"]:
        if parsed_obj.get(field) is not None:
            out[field] = str(parsed_obj[field])

    if "complexity_score" in parsed_obj:
        complexity_raw = parsed_obj["complexity_score"]
    elif "complexity_level" in parsed_obj:
        complexity_raw = parsed_obj["complexity_level"]
    else:
        return {}, "Missing complexity_score field"

    try:
        complexity = int(complexity_raw)
    except (ValueError, TypeError):
        return {}, f"Invalid complexity_score value: {complexity_raw}"
    if not (0 <= complexity <= 3):
        return {}, f"complexity_score out of range: {complexity}"
    out["complexity_score"] = str(complexity)

    if "linguistic_score" in parsed_obj:
        linguistic_raw = parsed_obj["linguistic_score"]
    elif "linguistic_correctness_naturalness" in parsed_obj:
        linguistic_raw = parsed_obj["linguistic_correctness_naturalness"]
    else:
        return {}, "Missing linguistic_score field"

    try:
        linguistic = int(linguistic_raw)
    except (ValueError, TypeError):
        return {}, f"Invalid linguistic_score value: {linguistic_raw}"
    if not (0 <= linguistic <= 4):
        return {}, f"linguistic_score out of range: {linguistic}"
    out["linguistic_score"] = str(linguistic)

    out["reasoning"] = str(parsed_obj.get("reasoning", "") or "").strip()
    return out, ""


def _extract_content_from_choice(choice: Dict[str, object]) -> str:
    message = (choice.get("message") or {}) if isinstance(choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_val = item.get("text")
                if text_val is not None:
                    chunks.append(str(text_val))
            elif isinstance(item, str):
                chunks.append(item)
        return "".join(chunks)

    return str(content or "")


def call_openrouter_eval(
    api_key: str,
    model_id: str,
    prompt_text: str,
    row: Dict[str, str],
    timeout_seconds: int,
    max_retries: int,
    max_output_tokens: int,
) -> Tuple[Dict[str, str], str]:
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

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": build_eval_user_message(row)},
        ],
        "temperature": 0,
        "max_tokens": max_output_tokens,
    }

    last_error = "Unknown OpenRouter error"
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            else:
                try:
                    data = response.json()
                except json.JSONDecodeError as exc:
                    last_error = f"Non-JSON OpenRouter response: {exc}; raw={response.text[:500]}"
                    data = {}

                choices = data.get("choices", []) if isinstance(data, dict) else []
                first_choice = choices[0] if choices else {}
                finish_reason = ""
                if isinstance(first_choice, dict):
                    finish_reason = str(first_choice.get("finish_reason") or "")

                content = _extract_content_from_choice(first_choice if isinstance(first_choice, dict) else {})
                if not content.strip():
                    suffix = f"; finish_reason={finish_reason}" if finish_reason else ""
                    last_error = f"Empty response content{suffix}"
                else:
                    parsed, parse_error = parse_single_eval_json(content)
                    if parse_error:
                        suffix = f"; finish_reason={finish_reason}" if finish_reason else ""
                        last_error = (
                            f"Invalid JSON output: {parse_error}{suffix}; raw_content={content[:500]}"
                        )
                    else:
                        return parsed, ""
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        if attempt < max_retries:
            time.sleep(1.5 * attempt)

    return {}, f"ERROR: {last_error}"


def _worker_eval(args: Tuple) -> Tuple[int, Dict[str, str], str]:
    idx, api_key, model_id, prompt_text, row, timeout_seconds, max_retries, max_output_tokens = args
    parsed, err = call_openrouter_eval(
        api_key=api_key,
        model_id=model_id,
        prompt_text=prompt_text,
        row=row,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
    )
    return idx, parsed, err


def _format_eta_hms(total_seconds: int) -> str:
    if total_seconds < 0:
        return "--:--:--"
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _model_progress_line(completed: int, total: int, start_time: float) -> str:
    if total <= 0:
        return "  evaluating 0/0"
    ratio = min(max(completed / total, 0.0), 1.0)
    width = 28
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - start_time, 1e-6)
    speed = completed / elapsed if completed else 0.0
    eta = int((total - completed) / speed) if speed > 0 else -1
    eta_text = _format_eta_hms(eta)
    return f"  evaluating [{bar}] {completed}/{total} ({ratio * 100:5.1f}%) ETA {eta_text}"


def row_needs_evaluation(row: Dict[str, str]) -> bool:
    if (row.get("evaluation_error") or "").strip():
        return True
    return any(not (row.get(field) or "").strip() for field in REQUIRED_OUTPUT_FIELDS)


def load_resume_rows_jsonl(
    output_path: Path,
    input_rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[int]]:
    if not output_path.exists():
        return [dict(row) for row in input_rows], list(range(len(input_rows)))

    existing_rows: List[Dict[str, str]] = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                existing_rows.append(json.loads(line))

    if len(existing_rows) != len(input_rows):
        print(
            f"[RESUME] Existing file row count mismatch for {output_path.name} "
            f"({len(existing_rows)} vs {len(input_rows)}). Re-evaluating all rows."
        )
        return [dict(row) for row in input_rows], list(range(len(input_rows)))

    pending_indices = [i for i, row in enumerate(existing_rows) if row_needs_evaluation(row)]
    print(
        f"[RESUME] {output_path.name}: reusing {len(existing_rows) - len(pending_indices)} completed rows, "
        f"retrying {len(pending_indices)} rows"
    )
    return existing_rows, pending_indices


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _base_output_row(input_row: Dict[str, str], model_label: str, model_id: str) -> Dict[str, str]:
    return {
        "uuid": str(input_row.get("uuid") or ""),
        "excerpt": str(input_row.get("excerpt") or ""),
        "question": str(input_row.get("question") or ""),
        "complexity_score": "",
        "linguistic_score": "",
        "reasoning": "",
        "evaluator_model_name": model_label,
        "evaluator_model_id": model_id,
        "evaluation_error": "",
    }


def evaluate_rows_for_input(args: argparse.Namespace) -> int:
    if not args.input_json.exists():
        raise SystemExit(f"Input JSON/JSONL not found: {args.input_json}")
    if not args.prompt_file.exists():
        raise SystemExit(f"Prompt file not found: {args.prompt_file}")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    rows = load_question_rows(args.input_json)
    if not rows:
        raise SystemExit(f"Input has no rows: {args.input_json}")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
        print(f"[INFO] Using first {len(rows)} rows due to --limit")

    model_map = parse_model_map(args.models)
    if not model_map:
        raise SystemExit("No models configured for evaluation")

    prompt_text = read_prompt(args.prompt_file)
    base_name = args.input_json.stem

    for model_label, model_id in model_map.items():
        print(f"\n[MODEL] {model_label} ({model_id})")
        print(f"[MODEL] rows={len(rows)}, workers={args.max_workers}")

        output_path = args.output_folder / f"{base_name}_eval_{model_label}.jsonl"

        if not args.no_resume_existing:
            result_rows, pending_indices = load_resume_rows_jsonl(output_path, rows)
            for i, r in enumerate(result_rows):
                r.setdefault("evaluator_model_name", model_label)
                r.setdefault("evaluator_model_id", model_id)
                if "excerpt" not in r:
                    r["excerpt"] = str(rows[i].get("excerpt") or "")
                if "uuid" not in r:
                    r["uuid"] = str(rows[i].get("uuid") or "")
                if "question" not in r:
                    r["question"] = str(rows[i].get("question") or "")
                r.setdefault("reasoning", "")
                r.setdefault("evaluation_error", "")
                for field in REQUIRED_OUTPUT_FIELDS:
                    r.setdefault(field, "")
        else:
            result_rows = [_base_output_row(r, model_label, model_id) for r in rows]
            pending_indices = list(range(len(rows)))

        if not pending_indices:
            print(f"[SKIP] {model_label}: no pending rows")
            continue

        work_items = [
            (
                idx,
                api_key,
                model_id,
                prompt_text,
                rows[idx],
                args.timeout_seconds,
                args.max_retries,
                args.max_output_tokens,
            )
            for idx in pending_indices
        ]

        completed = 0
        errors = 0
        start_ts = time.time()
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_worker_eval, item): item[0] for item in work_items}
            for future in as_completed(futures):
                idx, parsed, err = future.result()

                if err:
                    errors += 1
                    row_out = _base_output_row(rows[idx], model_label, model_id)
                    row_out["evaluation_error"] = err
                else:
                    row_out = _base_output_row(rows[idx], model_label, model_id)
                    row_out["uuid"] = parsed.get("uuid", row_out["uuid"])
                    row_out["excerpt"] = parsed.get("excerpt", row_out["excerpt"])
                    row_out["question"] = parsed.get("question", row_out["question"])
                    row_out["complexity_score"] = parsed.get("complexity_score", "")
                    row_out["linguistic_score"] = parsed.get("linguistic_score", "")
                    row_out["reasoning"] = parsed.get("reasoning", "")

                result_rows[idx] = row_out
                completed += 1
                print("\r" + _model_progress_line(completed, len(work_items), start_ts), end="", flush=True)
        print()

        success = len(work_items) - errors
        print(f"[MODEL DONE] success={success}, errors={errors}")
        if success == 0:
            print(
                f"[FATAL] Model {model_label} returned only errors for all {len(work_items)} rows. Stopping."
            )
            write_jsonl(output_path, result_rows)
            return 1

        write_jsonl(output_path, result_rows)
        print(f"[WRITE] {output_path}")

    return 0


def consolidate_eval_jsonls_to_json(
    input_json: Path,
    output_folder: Path,
    model_map: Dict[str, str],
    output_json: Path,
) -> int:
    base_name = input_json.stem
    rows_out: List[Dict[str, object]] = []

    for model_label, model_id in model_map.items():
        eval_output = output_folder / f"{base_name}_eval_{model_label}.jsonl"
        if not eval_output.exists():
            raise SystemExit(f"Expected evaluator output JSONL not found: {eval_output}")

        with eval_output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                raw_complexity = str(row.get("complexity_score", "")).strip()
                raw_linguistic = str(row.get("linguistic_score", "")).strip()

                complexity_score = int(raw_complexity) if raw_complexity.isdigit() else None
                linguistic_score = int(raw_linguistic) if raw_linguistic.isdigit() else None

                rows_out.append(
                    {
                        "uuid": str(row.get("uuid") or ""),
                        "excerpt": str(row.get("excerpt") or ""),
                        "question": str(row.get("question") or ""),
                        "complexity_score": complexity_score,
                        "linguistic_score": linguistic_score,
                        "reasoning": str(row.get("reasoning") or ""),
                        "evaluator_model_name": str(row.get("evaluator_model_name") or model_label),
                        "evaluator_model_id": str(row.get("evaluator_model_id") or model_id),
                        "evaluation_error": str(row.get("evaluation_error") or ""),
                    }
                )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows_out)


def to_project_relative(path: Path, project_root: Path) -> Path:
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return path


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    ensure_openrouter_key(project_root)

    args.input_json = to_project_relative(args.input_json, project_root)
    args.prompt_file = to_project_relative(args.prompt_file, project_root)
    args.output_folder = to_project_relative(args.output_folder, project_root)
    args.output_json = to_project_relative(args.output_json, project_root)

    rc = evaluate_rows_for_input(args)
    if rc == 0:
        model_map = parse_model_map(args.models)
        written = consolidate_eval_jsonls_to_json(
            input_json=args.input_json,
            output_folder=args.output_folder,
            model_map=model_map,
            output_json=args.output_json,
        )
        print(f"Wrote consolidated JSON {args.output_json} with {written} rows")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
