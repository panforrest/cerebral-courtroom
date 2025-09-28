from fastapi.testclient import TestClient
from unittest.mock import patch

import sys
sys.path.insert(0, r"C:\Users\Forrest Pan\Cerebral Courtroom")

import backend.main as mainmod

client = TestClient(mainmod.app)


def test_ws_present_flow():
    # create session
    r = client.post('/api/session', json={'title':'t','facts':'f'})
    assert r.status_code == 200
    sid = r.json()['session_id']

    # patch manager.run_turn_sequence to return a sequence of agent replies
    with patch('backend.main.manager.run_turn_sequence') as mock_seq:
        mock_seq.return_value = [
            {'agent': 'Opposing', 'text': '(mocked) Opposing reply'},
            {'agent': 'Judge', 'text': '(mocked) JUDGE: SUSTAINED - Reason.'},
            {'agent': 'Jury', 'text': '(mocked) JURY: Not Guilty (confidence: 45%)'}
        ]
        with client.websocket_connect(f'/ws/session/{sid}') as ws:
            ws.send_json({'type':'present','text':'The defendant was there.'})
            # receive three agent messages
            msgs = [ws.receive_json(), ws.receive_json(), ws.receive_json()]
            assert len(msgs) == 3
            assert msgs[0]['agent'] == 'Opposing'
            assert '(mocked) Opposing reply' in msgs[0]['text']


if __name__ == '__main__':
    test_ws_present_flow()
