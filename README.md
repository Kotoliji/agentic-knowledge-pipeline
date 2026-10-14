# Agentic Knowledge Pipeline

Multi-agent question answering over internal documents: a LangGraph agent graph
that routes, retrieves, drafts and self-checks answers with citations.

Work in progress — built in phases (see commit history).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The `corpus/` folder contains a small **fictional** company handbook
("Halden Systems") used as demo data and as the basis for evaluation.
