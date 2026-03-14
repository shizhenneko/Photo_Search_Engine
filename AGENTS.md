# Repository Guidelines

## Project Structure & Module Organization
`main.py` boots the Flask app and wires provider clients from `config.py`. HTTP routes live in `api/routes.py`. Core indexing and retrieval flow is split between `core/indexer.py` and `core/searcher.py`. Reusable integrations are under `utils/`, including `embedding_service.py`, `vision_llm_service.py`, `rerank_service.py`, `vector_store.py`, and `path_utils.py`. The UI is a server-rendered single page in `templates/index.html`. Tests live in `tests/`, with focused helper stubs in `tests/helpers.py`.

## Build, Test, and Development Commands
Use `uv` for local setup and execution.

- `uv venv .venv --python 3.12`: create the project virtual environment.
- `uv pip install --python .venv/bin/python -r requirements.txt`: install runtime and test dependencies.
- `./.venv/bin/python main.py`: start the Flask server locally.
- `./.venv/bin/python -m pytest -q`: run the full test suite.
- `./.venv/bin/python -m pytest tests/test_routes.py -q`: run a focused backend slice while iterating.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and short public docstrings. Follow `snake_case` for files, functions, and variables, and `PascalCase` for classes such as `Searcher` or `VisualRerankService`. Keep provider-specific logic inside `utils/` and avoid leaking API details into route handlers. Prefer additive, testable helpers over large inline blocks in templates or routes.

## Testing Guidelines
This repo uses `pytest` with `unittest`-style test cases. Name tests `test_<module>.py`. When changing retrieval logic, cover both route behavior and lower-level scoring or path handling. Use the lightweight fakes in `tests/helpers.py` instead of real model calls. Before merging UI work, start the app and verify the page manually in a browser, not just through template rendering tests.

## Commit & Pull Request Guidelines
Recent history uses short, direct summaries such as `finish rerank` and `trained and tested`. Keep commits scoped and imperative. Pull requests should include: what changed, which provider/config variables changed, test commands run, and screenshots for UI updates. If the change affects indexing or embeddings, call out that a full reindex is required.

## Security & Configuration Tips
Do not commit real `.env` secrets or personal photo paths in examples. Keep `.env.example` aligned with any provider or model changes. This project now assumes `uv` as the standard environment manager; install dependencies and run validation from `.venv` before handing off.
