"""
Unit tests for Phase 8: Status Algorithm
Spec: Section 8.2
"""

import unittest
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.p08_status import Edge, EdgeStore, status_as_of


class TestP08Status(unittest.TestCase):

    def setUp(self):
        self.doc_a = {
            "doc_id": "D_A",
            "issue_date": "2015-01-01",
            "effective_from": "2015-01-01",
            "effective_to": None,
        }
        self.doc_b = {
            "doc_id": "D_B",
            "issue_date": "2018-06-01",
            "effective_from": "2018-06-01",
        }
        self.doc_c = {
            "doc_id": "D_C",
            "issue_date": "2021-01-01",
            "effective_from": "2021-01-01",
        }

    def test_not_yet_effective(self):
        edges = EdgeStore()
        status, ev = status_as_of(self.doc_a, "2014-12-31", edges)
        self.assertEqual(status, "NOT_YET_EFFECTIVE")

    def test_expired(self):
        doc = dict(self.doc_a, effective_to="2020-12-31")
        edges = EdgeStore()
        status, ev = status_as_of(doc, "2021-01-01", edges)
        self.assertEqual(status, "EXPIRED")

    def test_in_force_original(self):
        edges = EdgeStore()
        status, ev = status_as_of(self.doc_a, "2016-01-01", edges)
        self.assertEqual(status, "IN_FORCE")
        self.assertIsNone(ev)

    def test_superseded_whole(self):
        # B supersedes A as of 2018-06-01
        e = Edge(
            edge_id="E1",
            type="SUPERSEDES",
            source_doc_id="D_B",
            target_doc_id="D_A",
            scope="whole",
            source_issue_date="2018-06-01",
            tier="Tier A",
        )
        edges = EdgeStore([e])

        # Before B: A is IN_FORCE
        s_before, _ = status_as_of(self.doc_a, "2018-05-30", edges)
        self.assertEqual(s_before, "IN_FORCE")

        # After B: A is SUPERSEDED by E1
        s_after, ev = status_as_of(self.doc_a, "2018-06-02", edges)
        self.assertEqual(s_after, "SUPERSEDED")
        self.assertEqual(ev.edge_id, "E1")

    def test_rescinded(self):
        e = Edge(
            edge_id="E2",
            type="RESCINDS",
            source_doc_id="D_B",
            target_doc_id="D_A",
            scope="whole",
            source_issue_date="2018-06-01",
            tier="Tier A",
        )
        edges = EdgeStore([e])
        status, ev = status_as_of(self.doc_a, "2019-01-01", edges)
        self.assertEqual(status, "RESCINDED")
        self.assertEqual(ev.edge_id, "E2")

    def test_in_force_amended(self):
        e = Edge(
            edge_id="E3",
            type="AMENDS",
            source_doc_id="D_B",
            target_doc_id="D_A",
            scope="partial",
            source_issue_date="2018-06-01",
            tier="Tier A",
        )
        edges = EdgeStore([e])
        status, ev = status_as_of(self.doc_a, "2019-01-01", edges)
        self.assertEqual(status, "IN_FORCE_AMENDED")
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].edge_id, "E3")

    def test_chain_supersession(self):
        # A superseded by B in 2018; B superseded by C in 2021
        e1 = Edge(edge_id="E1", type="SUPERSEDES", source_doc_id="D_B", target_doc_id="D_A", scope="whole", source_issue_date="2018-06-01", tier="Tier A")
        e2 = Edge(edge_id="E2", type="SUPERSEDES", source_doc_id="D_C", target_doc_id="D_B", scope="whole", source_issue_date="2021-01-01", tier="Tier A")
        edges = EdgeStore([e1, e2])

        # On 2019-01-01: A is SUPERSEDED, B is IN_FORCE, C is NOT_YET_EFFECTIVE
        s_a, _ = status_as_of(self.doc_a, "2019-01-01", edges)
        s_b, _ = status_as_of(self.doc_b, "2019-01-01", edges)
        s_c, _ = status_as_of(self.doc_c, "2019-01-01", edges)
        self.assertEqual(s_a, "SUPERSEDED")
        self.assertEqual(s_b, "IN_FORCE")
        self.assertEqual(s_c, "NOT_YET_EFFECTIVE")

        # On 2022-01-01: A is SUPERSEDED, B is SUPERSEDED, C is IN_FORCE
        s_a2, _ = status_as_of(self.doc_a, "2022-01-01", edges)
        s_b2, _ = status_as_of(self.doc_b, "2022-01-01", edges)
        s_c2, _ = status_as_of(self.doc_c, "2022-01-01", edges)
        self.assertEqual(s_a2, "SUPERSEDED")
        self.assertEqual(s_b2, "SUPERSEDED")
        self.assertEqual(s_c2, "IN_FORCE")


if __name__ == "__main__":
    unittest.main()
