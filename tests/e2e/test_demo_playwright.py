import os
import time
import threading
import requests
from playwright.sync_api import sync_playwright


def _start_uvicorn_in_thread(port: int):
    # start uvicorn programmatically to make the test self-contained
    from uvicorn import Config, Server

    config = Config("backend.main:app", host="127.0.0.1", port=port, log_level="info")
    server = Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # wait for server to become responsive
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(40):
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return server, thread
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Uvicorn server failed to start in time")


def test_demo_flow():
    port = int(os.environ.get('DEMO_PORT', '8001'))
    base = os.environ.get('DEMO_BASE_URL', f'http://127.0.0.1:{port}')

    server = None
    thread = None
    # If DEMO_USE_EXTERNAL_SERVER is set, assume an external server is running
    use_external = os.environ.get('DEMO_USE_EXTERNAL_SERVER')
    # If DEMO_USE_REAL_API is set, allow real OpenAI calls; otherwise force mocks
    use_real_api = os.environ.get('DEMO_USE_REAL_API')
    try:
        # Ensure we don't call the real OpenAI API during e2e runs here unless opt-in
        if not use_real_api:
            os.environ.pop('OPENAI_API_KEY', None)
        if not use_external:
            server, thread = _start_uvicorn_in_thread(port)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base + '/')
            # click create session
            page.click('text=Create Session & Connect')
            # wait for connect and ensure present button is enabled
            page.wait_for_selector('#presentBtn', state='visible', timeout=10000)
            # ensure button is enabled (WebSocket onopen sets disabled=false)
            page.wait_for_function("() => !document.querySelector('#presentBtn').disabled", timeout=10000)
            # click present
            page.click('#presentBtn')
            # wait for at least one agent message in wsOutput
            page.wait_for_selector('#wsOutput div', timeout=15000)
            # check jury panel updates
            jury_text = page.inner_text('#juryVerdict')
            conf_text = page.inner_text('#juryConfidenceText')
            assert jury_text.strip() != '(no verdict yet)'
            assert conf_text.strip() != '--'
            browser.close()
    finally:
        if server:
            # request shutdown
            server.should_exit = True
            # give server a moment to shutdown
            time.sleep(0.5)
