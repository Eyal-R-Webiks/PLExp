# data_prep

Data preparation, generation, and evaluation workspace for the Hebrew QA pipeline.

## Current Structure

```
data_prep/
├── README.md
├── summarize_knesset_gemini-3.1-pro-preview.py
├── generate_questions_gemini-3.1-pro-preview.py
├── evaluation_4_models.py
├── prompts/
│   ├── 01_knesset_summaries.md
│   ├── 02_question_generation.md
│   └── 03_question_assessment.md
├── questions/
│   ├── docs_sampled/               # sampled source documents/excerpts
│   ├── generation/                 # generated question outputs
│   ├── eval/                       # evaluation input files
│   └── eval_openrouter/            # per-model evaluation outputs
├── original_data_sets/             # local corpora (large; ignored in git)
│   ├── il-hym/
│   ├── knesset/
│   └── wiki/
└── reports/
```

## Script Roles

- `summarize_knesset_gemini-3.1-pro-preview.py`
  - produces concise summaries from source material for downstream question generation

- `generate_questions_gemini-3.1-pro-preview.py`
  - generates Hebrew questions using prompt templates in `prompts/`

- `evaluation_4_models.py`
  - evaluates questions with four OpenRouter models
  - system prompt: `prompts/03_question_assessment.md`
  - default input: `questions/eval/all_questions_for_eval.jsonl`
  - outputs per-model JSONL files to `questions/eval_openrouter/`

## Data and Git Notes

- `original_data_sets/` is intentionally not versioned because of dataset size.
- Keep API credentials only in root `.env`.
- Use a dedicated `smoke_tests/` folder at repository root for smoke-run artifacts.
