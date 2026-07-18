#!/usr/bin/env python3
"""Answer extractors for model output. Stdlib only.

Pipeline: strip_wrappers() first (think blocks, tool-call XML/JSON
wrappers per src/api_common.h shapes), then one of extract_numeric /
extract_choice / extract_freeform. Every extractor returns None when
no answer is found -- callers must treat None as "no answer", never
as a match.
"""

import re

# --- wrapper stripping ------------------------------------------------------

# Shapes from src/api_common.h (read-only reference):
#   <think>...</think>              closed think block
#   <think>...<EOF>                 truncated mid-think: everything after the
#                                   opener is thought, drop to end
#   <tool_call>\n{json}\n</tool_call>
#   <tool_response>...</tool_response>
#   {"tool_call": ...              bare-JSON opener fragment (mode-4 wrapper
#                                   junk, may dangle unclosed at end)

_CLOSED = [
    re.compile(r"<think>.*?</think>", re.S),
    re.compile(r"<tool_call>.*?</tool_call>", re.S),
    re.compile(r"<tool_response>.*?</tool_response>", re.S),
]
_OPEN = [
    re.compile(r"<think>.*$", re.S),
    re.compile(r"<tool_call>.*$", re.S),
    re.compile(r"<tool_response>.*$", re.S),
    re.compile(r"\{\"tool_call\".*$", re.S),  # dangling bare-JSON opener
]


def strip_wrappers(text):
    """Remove think blocks and tool-call wrappers; unclosed openers are
    dropped through end-of-text (truncated output yields no answer, not
    a leaked thought)."""
    for rx in _CLOSED:
        text = rx.sub("", text)
    for rx in _OPEN:
        text = rx.sub("", text)
    return text


def _final_line(text):
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


# --- numeric ----------------------------------------------------------------

_NUM = re.compile(
    r"[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANS_NUM = re.compile(
    r"(?:final answer|answer|result)\s*(?:is|:|=)\s*(" + _NUM.pattern + r")",
    re.I,
)


def _to_float(tok):
    tok = tok.replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(tok)
    except ValueError:
        return None


def extract_numeric(text):
    """Numeric answer as float, or None. Priority: \\boxed{}, then the
    last 'answer is/=/:' pattern, then the last number on the final
    non-empty line. Numbers elsewhere in the text are ignored."""
    text = strip_wrappers(text)
    boxed = _BOXED.findall(text)
    if boxed:
        nums = _NUM.findall(boxed[-1])
        if nums:
            return _to_float(nums[-1])
        return None  # boxed but non-numeric: no numeric answer
    tagged = _ANS_NUM.findall(text)
    if tagged:
        return _to_float(tagged[-1])
    nums = _NUM.findall(_final_line(text))
    if nums:
        return _to_float(nums[-1])
    return None


# --- multiple choice --------------------------------------------------------

def extract_choice(text, choices="ABCDE"):
    """Choice letter (uppercase) or None. Only anchored forms count:
    \\boxed{B}, 'answer is B', '(B)' tagged, or a final line that IS the
    letter. A letter appearing inside prose is never extracted (the
    label-shadow trap)."""
    text = strip_wrappers(text)
    cls = "[" + re.escape(choices) + re.escape(choices.lower()) + "]"
    # standalone letter token: no letter glued on either side
    tok = r"(?<![A-Za-z])\(?(" + cls + r")\)?(?![A-Za-z])"
    boxed = _BOXED.findall(text)
    if boxed:
        m = re.fullmatch(r"\s*\(?(" + cls + r")\)?\s*\.?\s*", boxed[-1])
        return m.group(1).upper() if m else None
    m = None
    for m in re.finditer(
        r"(?:final answer|answer|choice|option)\s*(?:is|:|=)?\s*" + tok, text, re.I
    ):
        pass  # keep last
    if m:
        return m.group(1).upper()
    m = re.fullmatch(r"\(?(" + cls + r")\)?\.?", _final_line(text))
    if m:
        return m.group(1).upper()
    return None


# --- freeform short answer --------------------------------------------------

_ANS_FREE = re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*", re.I)
_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.I)


def normalize_freeform(s):
    """Case/whitespace/article folding for short answers. Applied to
    both prediction and gold before comparison."""
    s = s.strip().strip("\"'` ").rstrip(".!?").strip()
    s = _ARTICLE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def extract_freeform(text):
    """Normalized short answer string, or None on empty output. Takes
    the text after the last 'answer is/:' tag if present, else the
    final non-empty line."""
    text = strip_wrappers(text)
    parts = _ANS_FREE.split(text)
    if len(parts) > 1:
        tail = parts[-1]
        # answer is the remainder of that line/sentence
        tail = tail.splitlines()[0] if tail.splitlines() else tail
        tail = re.split(r"[.!?](?:\s|$)", tail)[0]
        norm = normalize_freeform(tail)
        return norm if norm else None
    line = _final_line(text)
    if not line:
        return None
    norm = normalize_freeform(line)
    return norm if norm else None
