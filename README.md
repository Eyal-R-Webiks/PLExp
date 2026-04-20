# NLP_Abstractive_QA

Project for building a Hebrew abstractive QA dataset (1,000 question–answer pairs) for the MAFAT/Dicta evaluation leaderboard.
The pipeline generates information-seeking Hebrew questions from three native-Hebrew corpora using Gemini 3, produces abstractive answers, and supports human annotation of quality dimensions via Label Studio.

## Repository Structure

```
Abstractive_QA/
├── PRD.md                  # Product Requirements Document
├── pilot/                  # Completed pilot (scripts, runs, evaluation outputs)
├── data_prep/              # Corpus data and preparation scripts (large files not versioned)
└── resources/              # Reference docs, paper PDFs, and RAG info-sheet chunks
```

See each folder's `README.md` for details:
- [pilot/README.md](pilot/README.md) — setup, pipeline stages, script reference
- [data_prep/README.md](data_prep/README.md) — corpus descriptions and file formats

## Quick Start

```bash
cd pilot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Set API keys in the root-level .env file
```

## Corpora

| Corpus | Source | Size | Status |
|--------|--------|------|--------|
| Hebrew Wikipedia | `data_prep/wiki/` | 310 articles | Versioned |
| Israel HaYom | `data_prep/il-hym/` | ~275K articles | Not versioned (2 GB) |
| Knesset protocols | `data_prep/knesset/` | 91 shards | Not versioned (4.5 GB) |
