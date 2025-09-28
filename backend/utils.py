import re
from typing import Optional, Tuple

JURY_RE = re.compile(r"Verdict:\s*(Guilty|Not Guilty|No Verdict)\s*;\s*Confidence:\s*(\d{1,3})%", re.IGNORECASE)


def parse_jury_line(line: str) -> Optional[Tuple[str, int]]:
    """Parse a jury line of the form:
    Verdict: <Guilty|Not Guilty|No Verdict>; Confidence: <NN>%

    Returns (verdict, confidence) or None if parsing fails.
    """
    if not line:
        return None
    m = JURY_RE.search(line)
    if not m:
        return None
    verdict = m.group(1)
    try:
        confidence = int(m.group(2))
    except ValueError:
        return None
    return (verdict, confidence)
