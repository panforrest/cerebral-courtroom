from backend.utils import parse_jury_line
from backend.agent_manager import AgentManager


def test_parse_jury_line():
    line = "Verdict: Not Guilty; Confidence: 73%"
    parsed = parse_jury_line(line)
    assert parsed == ("Not Guilty", 73)


def test_run_turn_sequence_jury_parsing(monkeypatch):
    manager = AgentManager()
    sid = manager.create_session('t','Some facts here')
    # stub call_opposing and judge so we can focus on jury parsing
    monkeypatch.setattr(manager, 'call_opposing', lambda s, a: '(mock) opp')
    monkeypatch.setattr(manager, 'run_turn_sequence', manager.run_turn_sequence)
    # monkeypatch OpenAI usage by ensuring OPENAI_API_KEY is not set so we hit mock path
    import os
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    res = manager.run_turn_sequence(sid, 'an argument')
    # last result should be jury with parsed fields
    jury = res[-1]
    assert jury['agent'] == 'Jury'
    assert 'verdict' in jury and 'confidence' in jury
