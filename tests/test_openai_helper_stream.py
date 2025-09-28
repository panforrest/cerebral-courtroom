import types
import pytest

from backend import openai_helper


class _Evt:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_streaming_happy_path(monkeypatch):
    # Mock OpenAI client with a stream that yields delta events
    class FakeResponses:
        def stream(self, model, input, **kwargs):
            class Ctx:
                def __enter__(self_inner):
                    # yield generator
                    def gen():
                        yield _Evt(type='response.output_text.delta', delta='Hello ')
                        yield _Evt(type='response.output_text.delta', delta='world')
                    return gen()

                def __exit__(self_inner, exc_type, exc, tb):
                    return False
            return Ctx()

        def create(self, model, input, **kwargs):
            # fallback create used only if streaming unavailable
            return types.SimpleNamespace(output_text='fallback')

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.responses = FakeResponses()

    monkeypatch.setattr(openai_helper, 'OpenAI', FakeOpenAI)

    chunks = list(openai_helper.stream_responses('key', model='m', input_text='x'))
    assert ''.join(chunks) == 'Hello world'


def test_streaming_output_text_falls_through(monkeypatch):
    # Some SDKs provide output_text directly on the event
    class FakeResponses:
        def stream(self, model, input, **kwargs):
            class Ctx:
                def __enter__(self_inner):
                    def gen():
                        yield _Evt(output_text='Partial1')
                        yield _Evt(output_text='Partial2')
                    return gen()

                def __exit__(self_inner, exc_type, exc, tb):
                    return False
            return Ctx()

        def create(self, model, input, **kwargs):
            return types.SimpleNamespace(output_text='fallback')

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.responses = FakeResponses()

    monkeypatch.setattr(openai_helper, 'OpenAI', FakeOpenAI)

    chunks = list(openai_helper.stream_responses('key', model='m', input_text='x'))
    assert ''.join(chunks) == 'Partial1Partial2'


def test_streaming_unavailable_falls_back(monkeypatch):
    # Simulate stream raising so helper falls back to call_responses
    class FakeResponses:
        def stream(self, model, input, **kwargs):
            raise RuntimeError('stream not supported')

        def create(self, model, input, **kwargs):
            return types.SimpleNamespace(output_text='FULL TEXT')

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.responses = FakeResponses()

    monkeypatch.setattr(openai_helper, 'OpenAI', FakeOpenAI)

    chunks = list(openai_helper.stream_responses('key', model='m', input_text='x'))
    assert chunks == ['FULL TEXT']
