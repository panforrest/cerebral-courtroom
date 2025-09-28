import typing
import traceback

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


def _get_text_from_resp(resp) -> str:
    # Try common response shapes and fall back to str()
    try:
        text = getattr(resp, 'output_text', None)
        if text:
            return text
    except Exception:
        pass

    try:
        out = getattr(resp, 'output', None)
        if out and len(out) > 0:
            # new Responses shape: output[0].content[0].text
            try:
                return out[0].content[0].text
            except Exception:
                # fallback to stringify
                return str(out)
    except Exception:
        pass

    try:
        return str(resp)
    except Exception:
        return ''


def call_responses(api_key: str | None, model: str, input_text: str, **kwargs) -> str:
    """Call the OpenAI Responses API in a resilient way.

    Tries a couple of call shapes if the installed SDK rejects some kwargs
    (for example older/newer SDKs may not accept `max_tokens`).

    Returns a plain text string (best-effort). If OpenAI client is not
    available or api_key is None, raises RuntimeError.
    """
    if not api_key:
        raise RuntimeError('OPENAI API key not provided')
    if OpenAI is None:
        raise RuntimeError('openai package not installed')

    client = OpenAI(api_key=api_key)

    # Try primary call shape first
    last_exc = None
    try:
        resp = client.responses.create(model=model, input=input_text, **kwargs)
        return _get_text_from_resp(resp)
    except TypeError as e:
        # often caused by unexpected keyword arguments; try fallback shapes
        last_exc = e
    except Exception as e:
        # non-TypeError (network etc.) — rethrow
        raise

    # Fallback 1: remove common unsupported kwargs like max_tokens
    try:
        alt_kwargs = dict(kwargs)
        if 'max_tokens' in alt_kwargs:
            alt_kwargs.pop('max_tokens')
        resp = client.responses.create(model=model, input=input_text, **alt_kwargs)
        return _get_text_from_resp(resp)
    except Exception as e:
        last_exc = e

    # Fallback 2: try without any kwargs
    try:
        resp = client.responses.create(model=model, input=input_text)
        return _get_text_from_resp(resp)
    except Exception as e:
        last_exc = e

    # All attempts failed — surface a helpful error
    tb = traceback.format_exception(type(last_exc), last_exc, last_exc.__traceback__)
    raise RuntimeError('OpenAI Responses call failed. Last error:\n' + ''.join(tb))


def stream_responses(api_key: str | None, model: str, input_text: str, **kwargs):
    """Stream responses from the OpenAI Responses API in a resilient way.

    This function yields text deltas (str). It attempts to normalize different
    SDK stream shapes. If streaming is not available for the installed SDK,
    it will fall back to calling `call_responses` and yield the full text once.

    Usage:
        for chunk in stream_responses(...):
            handle(chunk)
    """
    if not api_key:
        raise RuntimeError('OPENAI API key not provided')
    if OpenAI is None:
        raise RuntimeError('openai package not installed')

    client = OpenAI(api_key=api_key)

    # Try to use a streaming context if available on the client
    try:
        # Many SDKs expose `client.responses.stream(...)` as a context manager
        stream_ctx = client.responses.stream(model=model, input=input_text, **kwargs)
    except TypeError:
        # unsupported kwargs — try with fewer kwargs
        try:
            alt_kwargs = dict(kwargs)
            if 'max_tokens' in alt_kwargs:
                alt_kwargs.pop('max_tokens')
            stream_ctx = client.responses.stream(model=model, input=input_text, **alt_kwargs)
        except Exception:
            stream_ctx = None
    except Exception:
        stream_ctx = None

    if stream_ctx is None:
        # Streaming not available; fallback to non-streaming call
        full = call_responses(api_key, model, input_text, **kwargs)
        yield full
        return

    # If stream_ctx is a context manager, iterate inside a with-block.
    try:
        with stream_ctx as stream:
            for event in stream:
                # Common streaming event patterns:
                # - event.type == 'response.output_text.delta' and event.delta contains text
                # - event.output_text may be present on final event
                try:
                    etype = getattr(event, 'type', None)
                    if etype == 'response.output_text.delta':
                        delta = getattr(event, 'delta', '')
                        if delta is None:
                            delta = ''
                        yield str(delta)
                        continue
                except Exception:
                    pass

                # Some SDKs yield partial Response objects with output_text attribute
                try:
                    partial = getattr(event, 'output_text', None)
                    if partial:
                        yield str(partial)
                        continue
                except Exception:
                    pass

                # As a last resort, stringify the event
                try:
                    s = str(event)
                    if s:
                        yield s
                except Exception:
                    continue
    except Exception:
        # If streaming failed mid-way, try to return a final non-streaming text
        try:
            final = call_responses(api_key, model, input_text)
            yield final
        except Exception:
            raise
