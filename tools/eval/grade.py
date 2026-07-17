#!/usr/bin/env python3
"""Graders over extracted answers. Stdlib only.

Every grader returns (verdict, reason): verdict is a bool, reason a
short string. A None prediction always fails with reason
"no answer extracted" -- absence of an answer is never a pass.
"""

import math

from extract import normalize_freeform


def grade_exact(pred, gold):
    """Exact match after normalize_freeform on both sides."""
    if pred is None:
        return False, "no answer extracted"
    p, g = normalize_freeform(str(pred)), normalize_freeform(str(gold))
    if p == g:
        return True, "exact match: %r" % p
    return False, "mismatch: got %r, want %r" % (p, g)


def grade_numeric(pred, gold, rel_tol=1e-4, abs_tol=1e-9):
    """Numeric match within tolerance. pred/gold are floats (or
    parseable strings)."""
    if pred is None:
        return False, "no answer extracted"
    try:
        p, g = float(pred), float(gold)
    except (TypeError, ValueError):
        return False, "non-numeric: got %r, want %r" % (pred, gold)
    if math.isclose(p, g, rel_tol=rel_tol, abs_tol=abs_tol):
        return True, "numeric match: %g ~ %g" % (p, g)
    return False, "numeric mismatch: got %g, want %g" % (p, g)


def grade_choice(pred, gold):
    """Multiple-choice letter match, case-insensitive."""
    if pred is None:
        return False, "no answer extracted"
    p, g = str(pred).strip().upper(), str(gold).strip().upper()
    if len(p) != 1:
        return False, "not a single letter: %r" % pred
    if p == g:
        return True, "choice match: %s" % p
    return False, "choice mismatch: got %s, want %s" % (p, g)
