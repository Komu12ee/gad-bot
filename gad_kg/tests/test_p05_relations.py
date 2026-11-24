"""
Unit tests for Phase 5: Relation Extraction
Spec: Section 4.4, Section 5 P5, Section 6 Prompt R10
"""

import unittest
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.p05_relations import (
    build_ref_map_string,
    vote_relations,
    extract_relations_for_doc
)
from src.common.llm import verify_quote_substring


class TestP05Relations(unittest.TestCase):

    def test_ref_map_string(self):
        ref_items = [
            {"idx": 1, "number_norm": "F-1/2020", "date": "2020-05-10", "subject_raw": "Leave Rules"},
            {"idx": 2, "number_raw": "100/2019", "date_raw": "15 Jan 2019", "subject_raw": "Circular"},
        ]
        s = build_ref_map_string(ref_items)
        self.assertIn("(1) F-1/2020 dated 2020-05-10 Leave Rules", s)
        self.assertIn("(2) 100/2019 dated 15 Jan 2019 Circular", s)

    def test_vote_relations_unanimous(self):
        # 3 runs agree
        run0 = [{"relation": "SUPERSEDES", "target_number_raw": "F-10/2018", "quote": "supersedes earlier order", "scope": "whole"}]
        run1 = [{"relation": "SUPERSEDES", "target_number_raw": "F-10/2018", "quote": "supersedes earlier order", "scope": "whole"}]
        run2 = [{"relation": "SUPERSEDES", "target_number_raw": "F-10/2018", "quote": "supersedes earlier order", "scope": "whole"}]
        
        resolved = vote_relations([run0, run1, run2])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["relation"], "SUPERSEDES")
        self.assertEqual(resolved[0]["confidence"], 1.0)
        self.assertEqual(len(resolved[0]["votes"]), 3)

    def test_vote_relations_majority(self):
        # 2 runs agree, 1 disagrees
        run0 = [{"relation": "AMENDS", "target_number_raw": "F-5/2021", "quote": "partially amends order", "scope": "partial"}]
        run1 = [{"relation": "AMENDS", "target_number_raw": "F-5/2021", "quote": "partially amends order", "scope": "partial"}]
        run2 = [{"relation": "REFERS_TO", "target_number_raw": "F-5/2021", "quote": "partially amends order", "scope": "partial"}]

        resolved = vote_relations([run0, run1, run2])
        # Group with AMENDS should have count=2 (confidence 0.67)
        amends_group = next(r for r in resolved if r["relation"] == "AMENDS")
        self.assertAlmostEqual(amends_group["confidence"], 0.67, places=2)

    def test_quote_substring_invariant(self):
        source = "This order hereby supersedes order No. 120/GAD dated 01-01-2020 in its entirety."
        valid_q = "supersedes order No. 120/GAD"
        hallucinated_q = "supersedes order No. 999/GAD"

        self.assertTrue(verify_quote_substring(valid_q, source))
        self.assertFalse(verify_quote_substring(hallucinated_q, source))


if __name__ == "__main__":
    unittest.main()
