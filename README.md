# Cerebral Courtroom

MVP scaffold for Cerebral Courtroom — a multi-agent courtroom simulation using GPT-5-Codex.

Quickstart

1. Create a Python virtual environment and activate it.

2. Install backend dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

3. Set your OpenAI API key in the environment:

   ```powershell
   $env:OPENAI_API_KEY = "sk-..."
   ```

4. Run the backend:

   ```powershell
   uvicorn backend.main:app --reload --port 8000
   ```

   Demo endpoint (single-agent opposing counsel):

   1. Start the backend as above.
   2. POST to `http://localhost:8000/api/demo/opposing` with JSON:

   ```powershell
   curl -X POST "http://localhost:8000/api/demo/opposing" -H "Content-Type: application/json" -d '{"facts": "Alice saw Bob at the store.", "argument": "The defendant was at the scene."}'
   ```

   This returns a JSON `reply` from the Opposing Counsel agent.

Frontend and further instructions will be added during the hackathon.

Running end-to-end Playwright tests

- Mocked (default, safe for CI):

   ```powershell
   # from the project root
   python -m pytest tests/e2e/test_demo_playwright.py -q -s
   ```

- Real OpenAI API (opt-in):

   ```powershell
   # set your API key and enable real API mode
   $env:OPENAI_API_KEY = "sk-..."
   $env:DEMO_USE_REAL_API = "1"
   python -m pytest tests/e2e/test_demo_playwright.py -q -s
   ```

   Or use pytest CLI flag (sets the env var for you):

   ```powershell
   python -m pytest tests/e2e/test_demo_playwright.py -q -s --real-api
   ```

Notes:
- Real API runs will call OpenAI and consume tokens; use only when validating real model behavior.
- If Playwright browsers are not installed, run `python -m playwright install chromium` first.

CI / GitHub Actions
-------------------

Two GitHub Actions workflows are included to make CI and manual real‑API runs easy:

- `.github/workflows/ci.yml` — runs on push and pull requests to `main`. It installs dependencies and Playwright browsers, then runs the test suite in mocked mode (the e2e Playwright test runs with `DEMO_USE_REAL_API=0` by default so CI is fast and deterministic).

- `.github/workflows/real-e2e.yml` — a manual `workflow_dispatch` job for running the real Playwright e2e test against the live OpenAI API. To run it, navigate to Actions → Real E2E (manual) and provide the repository secret `OPENAI_API_KEY`. The job sets `DEMO_USE_REAL_API=1` during the run.

Security note: do not store real API keys in repository files. Use GitHub repository secrets for `OPENAI_API_KEY` when running the manual workflow.