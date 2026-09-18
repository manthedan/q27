#!/usr/bin/env python3
"""CPU regression for per-request (not aggregate) XML engagement evidence."""
import unittest
from test_bonsai2_serving import check_grammar_log


class GrammarEvidence(unittest.TestCase):
    def test_each_request_must_engage_and_close_once(self):
        good = ["[toolgram] engaged (rem=0, dialect=xml)", "[toolgram] call closed"]
        for route in range(6):
            check_grammar_log(good, str(route))
        # Six aggregate engagements can hide a missing route. Each local
        # slice must reject both the duplicate and the empty request instead.
        for bad in (good * 2, [], good[:1], good[1:],
                    [line.replace("xml", "json") for line in good],
                    good + ["[toolgram] disengaged: error"]):
            with self.subTest(lines=bad), self.assertRaises(RuntimeError):
                check_grammar_log(bad, "route")


if __name__ == "__main__":
    unittest.main()
