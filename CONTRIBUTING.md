# Contributing

Thanks for investing in the Customer Lifetime Value Forecasting toolkit! Below is the quickest path to productive contributions.

1. **Set up the environment**
   - `python -m venv .venv && .venv\Scripts\activate`
   - `pip install -r requirements.txt -r requirements-dev.txt`
   - `python scripts/run_all.py` to regenerate artifacts.

2. **Coding standards**
   - Run `ruff check . --fix` and `black src scripts dashboards` before committing.
   - Keep notebooks explanatory; heavy logic should live in `src/` or `scripts/`.

3. **Testing**
   - Add or update smoke tests under `tests/` whenever outputs or schemas change.
   - Execute `pytest -q` locally.

4. **Pull requests**
   - Reference the related issue / feature request.
   - Summarize modeling impact and attach relevant plots if behavior changes.

By opening a pull request you agree that your contributions will be licensed under the MIT License.
