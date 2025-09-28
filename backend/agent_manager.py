import uuid
import os
from typing import Dict, Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from . import prompts


class AgentManager:
    def __init__(self):
        # sessions: session_id -> dict with facts, title, transcript(list of tuples (speaker,text))
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, title: str, facts: str) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            'title': title,
            'facts': facts,
            'transcript': []
        }
        return sid

    def get_session(self, sid: str):
        return self.sessions.get(sid)

    def add_user_presentation(self, sid: str, text: str):
        sess = self.get_session(sid)
        if sess is None:
            raise KeyError('session not found')
        sess['transcript'].append(('User', text))

    def call_opposing(self, sid: str, user_argument: str) -> str:
        """Call the Opposing Counsel agent (non-streaming) and return text reply."""
        sess = self.get_session(sid)
        if sess is None:
            raise KeyError('session not found')

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or OpenAI is None:
            # Return a deterministic mocked reply when API not available
            reply = "(mock) Opposing Counsel: The facts do not support that claim; can you prove presence?"
            sess['transcript'].append(('Opposing', reply))
            return reply
        from .openai_helper import call_responses

        prompt = prompts.OPPOSING_PROMPT_TEMPLATE.format(facts=sess['facts'], argument=user_argument)
        try:
            text = call_responses(api_key, model='gpt-5-codex', input_text=prompt, max_tokens=300)
            sess['transcript'].append(('Opposing', text))
            return text
        except Exception as e:
            err = f"(error) {e}"
            sess['transcript'].append(('Opposing', err))
            return err

    def run_turn_sequence(self, sid: str, user_argument: str):
        """Run a simple multi-agent turn sequence for the given session and user argument.

        Sequence: Opposing Counsel -> Judge -> Jury (verdict summary).
        Returns a list of dicts: [{'agent': 'Opposing', 'text': ...}, ...]
        """
        sess = self.get_session(sid)
        if sess is None:
            raise KeyError('session not found')

        results = []

        # 1) Opposing Counsel
        opposing_text = self.call_opposing(sid, user_argument)
        results.append({'agent': 'Opposing', 'text': opposing_text})

        # 2) Judge - short ruling based on facts and transcript
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or OpenAI is None:
            judge_reply = "(mock) JUDGE: SUSTAINED - The objection is supported by the facts."
            sess['transcript'].append(('Judge', judge_reply))
            results.append({'agent': 'Judge', 'text': judge_reply})
        else:
            # use openai_helper for resilience
            from .openai_helper import call_responses
            transcript_text = "\n".join([f"{s}: {t}" for s, t in sess.get('transcript', [])])
            prompt = prompts.JUDGE_PROMPT + "\nPinned facts:\n" + sess.get('facts', '') + "\nTranscript:\n" + transcript_text
            try:
                jtext = call_responses(api_key, model='gpt-5', input_text=prompt, max_tokens=150)
                sess['transcript'].append(('Judge', jtext))
                results.append({'agent': 'Judge', 'text': jtext})
            except Exception as e:
                jerr = f"(error) Judge: {e}"
                sess['transcript'].append(('Judge', jerr))
                results.append({'agent': 'Judge', 'text': jerr})

        # 3) Jury - short verdict/confidence summary
        # 3) Jury - short verdict/confidence summary (structured)
        try:
            from .utils import parse_jury_line
        except Exception:
            # fallback when running from different working directory
            from backend.utils import parse_jury_line

        if not api_key or OpenAI is None:
            jury_reply = "Verdict: Guilty; Confidence: 60%"
            sess['transcript'].append(('Jury', jury_reply))
            parsed = parse_jury_line(jury_reply)
            jury_result = {'agent': 'Jury', 'text': jury_reply}
            if parsed:
                jury_result['verdict'] = parsed[0]
                jury_result['confidence'] = parsed[1]
            results.append(jury_result)
        else:
            from .openai_helper import call_responses
            transcript_text = "\n".join([f"{s}: {t}" for s, t in sess.get('transcript', [])])
            jury_prompt = prompts.JURY_PROMPT.format(facts=sess.get('facts', ''), transcript=transcript_text)
            try:
                jtext = call_responses(api_key, model='gpt-5', input_text=jury_prompt, max_tokens=60)
                sess['transcript'].append(('Jury', jtext))
                jury_result = {'agent': 'Jury', 'text': jtext}
                parsed = parse_jury_line(jtext)
                if parsed:
                    jury_result['verdict'] = parsed[0]
                    jury_result['confidence'] = parsed[1]
                results.append(jury_result)
            except Exception as e:
                jerr = f"(error) Jury: {e}"
                sess['transcript'].append(('Jury', jerr))
                results.append({'agent': 'Jury', 'text': jerr})

        return results

    def run_turn_sequence_stream(self, sid: str, user_argument: str, send_sync):
        """Run the multi-agent sequence but stream deltas via send_sync callback.

        send_sync(payload) must be a thread-safe function that accepts a JSON-serializable dict
        and sends it to the WebSocket (or similar). This method blocks while streaming and
        returns when finished.
        """
        sess = self.get_session(sid)
        if sess is None:
            raise KeyError('session not found')

        # helper to append to transcript and optionally send final
        def send_final(agent, text, extra=None):
            sess['transcript'].append((agent, text))
            payload = {'type': 'done', 'agent': agent, 'text': text}
            if extra:
                payload.update(extra)
            try:
                send_sync(payload)
            except Exception:
                pass

        api_key = os.getenv('OPENAI_API_KEY')

        # 1) Opposing Counsel (stream if available)
        agent = 'Opposing'
        prompt = prompts.OPPOSING_PROMPT_TEMPLATE.format(facts=sess['facts'], argument=user_argument)
        if not api_key or OpenAI is None:
            # mock streaming: send a couple deltas then done
            parts = ['(mock) Opposing:', ' The facts do not support that claim.', ' Can you provide evidence?']
            accum = ''
            for p in parts:
                accum += p
                try:
                    send_sync({'type': 'delta', 'agent': agent, 'delta': p})
                except Exception:
                    pass
            send_final(agent, accum)
        else:
            try:
                from .openai_helper import stream_responses
                accum = ''
                for chunk in stream_responses(api_key, model='gpt-5-codex', input_text=prompt):
                    text = str(chunk)
                    accum += text
                    try:
                        send_sync({'type': 'delta', 'agent': agent, 'delta': text})
                    except Exception:
                        pass
                send_final(agent, accum)
            except Exception as e:
                send_final(agent, f"(error) {e}")

        # 2) Judge
        agent = 'Judge'
        transcript_text = "\n".join([f"{s}: {t}" for s, t in sess.get('transcript', [])])
        judge_prompt = prompts.JUDGE_PROMPT + "\nPinned facts:\n" + sess.get('facts', '') + "\nTranscript:\n" + transcript_text
        if not api_key or OpenAI is None:
            parts = ['(mock) JUDGE: SUSTAINED -', ' The objection is supported by the facts.']
            accum = ''
            for p in parts:
                accum += p
                try:
                    send_sync({'type': 'delta', 'agent': agent, 'delta': p})
                except Exception:
                    pass
            send_final(agent, accum)
        else:
            try:
                from .openai_helper import stream_responses
                accum = ''
                for chunk in stream_responses(api_key, model='gpt-5', input_text=judge_prompt):
                    text = str(chunk)
                    accum += text
                    try:
                        send_sync({'type': 'delta', 'agent': agent, 'delta': text})
                    except Exception:
                        pass
                send_final(agent, accum)
            except Exception as e:
                send_final(agent, f"(error) {e}")

        # 3) Jury
        from .utils import parse_jury_line
        agent = 'Jury'
        transcript_text = "\n".join([f"{s}: {t}" for s, t in sess.get('transcript', [])])
        jury_prompt = prompts.JURY_PROMPT.format(facts=sess.get('facts', ''), transcript=transcript_text)
        if not api_key or OpenAI is None:
            parts = ['Verdict: Guilty; ', 'Confidence: 60%']
            accum = ''
            for p in parts:
                accum += p
                try:
                    send_sync({'type': 'delta', 'agent': agent, 'delta': p})
                except Exception:
                    pass
            parsed = parse_jury_line(accum)
            extra = {}
            if parsed:
                extra['verdict'] = parsed[0]
                extra['confidence'] = parsed[1]
            send_final(agent, accum, extra=extra)
        else:
            try:
                from .openai_helper import stream_responses
                accum = ''
                for chunk in stream_responses(api_key, model='gpt-5', input_text=jury_prompt):
                    text = str(chunk)
                    accum += text
                    try:
                        send_sync({'type': 'delta', 'agent': agent, 'delta': text})
                    except Exception:
                        pass
                parsed = parse_jury_line(accum)
                extra = {}
                if parsed:
                    extra['verdict'] = parsed[0]
                    extra['confidence'] = parsed[1]
                send_final(agent, accum, extra=extra)
            except Exception as e:
                send_final(agent, f"(error) {e}")
