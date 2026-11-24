"""
Unit tests for Phase 6: Verification & Tier Assignment
Spec: Section 4.4, Section 5 P6, Section 6 Prompt R11
"""

import unittest
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.p06_verify import (
    verify_relation_code_rules,
    assign_tier
)


class TestP06Verify(unittest.TestCase):

    def test_circular_cannot_supersede_act(self):
        source_header = {"doc_type": "CIRCULAR", "issue_date": "2020-01-01", "doc_number_norm": "100/2020"}
        rel = {"relation": "SUPERSEDES", "target_kind": "ACT_RULE", "target_number_norm": "ACT_1960"}
        passed, reason = verify_relation_code_rules(rel, source_header)
        self.assertFalse(passed)
        self.assertEqual(reason, "circular_cannot_supersede_act")

    def test_corrigendum_auto_correct_scope(self):
        source_header = {"doc_type": "CORRIGENDUM", "issue_date": "2020-01-01", "doc_number_norm": "100/2020"}
        rel = {"relation": "CORRECTS", "target_kind": "DOCUMENT", "scope": "whole"}
        passed, reason = verify_relation_code_rules(rel, source_header)
        self.assertTrue(passed)
        self.assertEqual(rel["scope"], "partial")

    def test_date_inversion_check(self):
        source_header = {"doc_type": "ORDER", "issue_date": "2019-01-01", "doc_number_norm": "100/2019"}
        rel = {"relation": "SUPERSEDES", "target_kind": "DOCUMENT", "target_date": "2021-05-01"}
        passed, reason = verify_relation_code_rules(rel, source_header)
        self.assertFalse(passed)
        self.assertIn("date_inversion", reason)

    def test_self_reference_check(self):
        source_header = {"doc_type": "ORDER", "issue_date": "2020-01-01", "doc_number_norm": "F-10/2020"}
        rel = {"relation": "AMENDS", "target_kind": "DOCUMENT", "target_number_norm": "F-10/2020"}
        passed, reason = verify_relation_code_rules(rel, source_header)
        self.assertFalse(passed)
        self.assertEqual(reason, "self_reference")

    def test_tier_assignment(self):
        # Tier A: code passed, high confidence (1.0), unanimous
        rel_a = {"relation": "SUPERSEDES", "confidence": 1.0}
        tier, _ = assign_tier(rel_a, code_passed=True, reject_reason=None, llm_verdict=None)
        self.assertEqual(tier, "Tier A")

        # Tier B: code passed, medium confidence (0.67)
        rel_b = {"relation": "AMENDS", "confidence": 0.67}
        tier, _ = assign_tier(rel_b, code_passed=True, reject_reason=None, llm_verdict=None)
        self.assertEqual(tier, "Tier B")

        # Tier C: code failed
        rel_c = {"relation": "SUPERSEDES", "confidence": 1.0}
        tier, _ = assign_tier(rel_c, code_passed=False, reject_reason="date_inversion", llm_verdict=None)
        self.assertEqual(tier, "Tier C")


if __name__ == "__main__":
    unittest.main()
