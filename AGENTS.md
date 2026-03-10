# Repository Guidelines

## Project Structure & Module Organization
- `cal_ads_customer_type.py`: main PySpark ETL job. It reads MySQL + Hive sources, merges labels, and writes `ads_acct_customer_type`.
- `park_ocr/`: OCR-related asset folder (currently image files).
- `.venv/`: local developer environment (machine-specific, not a source module).
- If the codebase grows, place reusable logic in `src/` and tests in `tests/`.

## Build, Test, and Development Commands
- `python -m venv .venv`: create a local virtual environment.
- `.\.venv\Scripts\Activate.ps1`: activate the environment in PowerShell.
- `pip install pyspark mysql-connector-python pytest`: install runtime and test dependencies.
- `python -m py_compile cal_ads_customer_type.py`: quick syntax validation.
- `python cal_ads_customer_type.py`: run directly (requires Spark/Hive/MySQL connectivity).
- `spark-submit --master yarn cal_ads_customer_type.py`: preferred cluster execution path.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and `snake_case` for functions/variables.
- Keep DataFrame stages readable with explicit intermediate names such as `wl_df`, `label_df`, `result_df`.
- Group imports: standard library, then third-party.
- Avoid hard-coded credentials and endpoints in code; load from environment variables or a local config file excluded from version control.

## Testing Guidelines
- Use `pytest` for new tests.
- Name test files `tests/test_<module>.py` and test functions `test_<behavior>()`.
- Prefer unit tests for transformation logic with a local Spark session (`master("local[2]")`).
- Mark environment-dependent tests as integration tests and keep them out of default runs.
- Run `pytest -q` before opening a PR.

## Commit & Pull Request Guidelines
- This folder currently has no Git metadata; when initialized, use Conventional Commits (for example, `feat: add label merge guard`, `fix: handle null acct_labels`).
- Keep commits focused and atomic.
- PRs should include: purpose, impacted tables/fields, validation evidence (sample counts/log excerpts), and any config changes.

## Security & Configuration Tips
- Treat database credentials as secrets; never commit production usernames/passwords.
- Use environment variables such as `MYSQL_URL`, `MYSQL_USER`, and `MYSQL_PASSWORD`.
- If credentials were previously exposed in source, rotate them immediately.
