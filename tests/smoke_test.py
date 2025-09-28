"""Smoke test that mocks OpenAI client to validate SSE + fallback logic without network calls."""
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

import sys
sys.path.insert(0, r"C:\Users\Forrest Pan\Cerebral Courtroom")

import backend.main as mainmod

client = TestClient(mainmod.app)


def test_non_streaming_opposing_mock():
    # Mock the OpenAI client.responses.create to return an object with output_text
    fake_resp = MagicMock()
    fake_resp.output_text = "Mocked opposing counsel reply."

    # Ensure the endpoint thinks an API key is present so it exercises the OpenAI call path
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}):
        with patch('backend.main.OpenAI') as MockOpenAI:
            instance = MockOpenAI.return_value
            instance.responses.create.return_value = fake_resp

            r = client.post('/api/demo/opposing', json={
                'facts': 'Alice saw Bob',
                'argument': 'He was there'
            })
            assert r.status_code == 200
            j = r.json()
            assert 'reply' in j
            assert 'Mocked opposing counsel' in j['reply']


if __name__ == '__main__':
    test_non_streaming_opposing_mock()
    # smoke test runner
