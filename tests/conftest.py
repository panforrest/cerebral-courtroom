import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--real-api",
        action="store_true",
        default=False,
        help="Run tests that call the real OpenAI API (must set OPENAI_API_KEY).",
    )


def pytest_configure(config):
    if config.getoption("--real-api"):
        os.environ["DEMO_USE_REAL_API"] = "1"
