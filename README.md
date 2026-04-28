# Abstractive_QA

Hebrew abstractive QA data-preparation workspace.

The repository currently focuses on:
- generating summaries and questions from source corpora
- evaluating generated questions with multiple LLM evaluators through OpenRouter
- storing prompts, sampled documents, generated questions, and evaluation outputs

## Current Repository Layout

```
Abstractive_QA/
├── PRD.md
├── README.md
├── data_prep/
│   ├── README.md
│   ├── summarize_knesset_gemini-3.1-pro-preview.py
│   ├── generate_questions_gemini-3.1-pro-preview.py
│   ├── evaluation_4_models.py
│   ├── prompts/
│   │   ├── 01_knesset_summaries.md
│   │   ├── 02_question_generation.md
│   │   └── 03_question_assessment.md
│   ├── questions/
│   │   ├── docs_sampled/
│   │   ├── generation/
│   │   ├── eval/
│   │   └── eval_openrouter/
│   ├── original_data_sets/          # Local large datasets (ignored in git)
│   └── reports/
└── resources/
		├── Bloom_taxonomy/
		├── MAFAT_requirements_doc.md
		├── llms_bloom_summary.html
		└── chunks_for_RAG/
```

## Important Notes

- Large source corpora under `data_prep/original_data_sets/` are intentionally ignored in git.
- API credentials are loaded from the root `.env` file.
- Smoke or ad-hoc test outputs should be written under `smoke_tests/`.

## Main Scripts

- `data_prep/summarize_knesset_gemini-3.1-pro-preview.py`
	- builds short summaries from source materials
- `data_prep/generate_questions_gemini-3.1-pro-preview.py`
	- generates Hebrew questions from excerpts/summaries
- `data_prep/evaluation_4_models.py`
	- evaluates questions with 4 OpenRouter models using the assessment prompt

## Evaluation Defaults

The evaluator defaults to:
- input: `data_prep/questions/eval/all_questions_for_eval.jsonl`
- system prompt: `data_prep/prompts/03_question_assessment.md`
- per-model jsonl output: `data_prep/questions/eval_openrouter/`
- consolidated json output: `data_prep/questions/eval/all_questions_for_eval_scored.json`
