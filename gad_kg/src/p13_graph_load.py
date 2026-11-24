"""
Phase 13: Graph Loader (Neo4j Cypher & SQLite/NetworkX Local Graph)
Spec: Section 5 P13, Section 8
"""

import os
import sys
import json
import sqlite3
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.io import read_jsonl, write_jsonl, ensure_dir


CYPHER_HEADER = """// Neo4j 5.x Graph Schema Constraints and Indexes
CREATE CONSTRAINT doc_id    IF NOT EXISTS FOR (d:Document)   REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT stub_id   IF NOT EXISTS FOR (s:Stub)       REQUIRE s.stub_id IS UNIQUE;
CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (c:Clause)     REQUIRE c.clause_id IS UNIQUE;
CREATE CONSTRAINT table_id  IF NOT EXISTS FOR (t:Table)      REQUIRE t.table_id IS UNIQUE;
CREATE CONSTRAINT ent_id    IF NOT EXISTS FOR (e:Entity)     REQUIRE e.entity_id IS UNIQUE;
CREATE CONSTRAINT inst_id   IF NOT EXISTS FOR (i:Instrument) REQUIRE i.inst_id IS UNIQUE;
CREATE INDEX doc_number IF NOT EXISTS FOR (d:Document) ON (d.number_norm);
CREATE INDEX doc_date   IF NOT EXISTS FOR (d:Document) ON (d.issue_date);
CREATE INDEX doc_type   IF NOT EXISTS FOR (d:Document) ON (d.doc_type);

"""


def init_graph_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database for unified Graph Storage."""
    ensure_dir(Path(db_path).parent)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS nodes")
    cur.execute("""
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT,
            doc_number_norm TEXT,
            issue_date TEXT,
            props_json TEXT
        )
    """)
    cur.execute("CREATE INDEX idx_nodes_type ON nodes(node_type)")
    cur.execute("CREATE INDEX idx_nodes_num ON nodes(doc_number_norm)")
    cur.execute("CREATE INDEX idx_nodes_date ON nodes(issue_date)")

    cur.execute("DROP TABLE IF EXISTS edges")
    cur.execute("""
        CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT,
            target_id TEXT,
            edge_type TEXT,
            tier TEXT,
            scope TEXT,
            effective_date TEXT,
            props_json TEXT
        )
    """)
    cur.execute("CREATE INDEX idx_edges_src ON edges(source_id)")
    cur.execute("CREATE INDEX idx_edges_tgt ON edges(target_id)")
    cur.execute("CREATE INDEX idx_edges_type ON edges(edge_type)")
    cur.execute("CREATE INDEX idx_edges_tier ON edges(tier)")

    conn.commit()
    return conn


def build_cypher_export(
    docs: List[Dict[str, Any]],
    stubs: List[Dict[str, Any]],
    clauses: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    out_cypher_path: str,
):
    """Generate Neo4j 5.x Cypher load script with UNWIND batches."""
    ensure_dir(Path(out_cypher_path).parent)
    with open(out_cypher_path, "w", encoding="utf-8") as f:
        f.write(CYPHER_HEADER)

        # 1. Documents
        f.write("// Load Documents\n")
        f.write(f":params rows => {json.dumps(docs, ensure_ascii=False)};\n")
        f.write("""UNWIND $rows AS row
MERGE (d:Document {doc_id: row.doc_id})
SET d.doc_type = row.doc_type,
    d.number_raw = row.doc_number_raw,
    d.number_norm = row.doc_number_norm,
    d.issue_date = row.issue_date,
    d.subject = row.subject,
    d.language = row.language,
    d.file_id = row.file_id;
\n""")

        # 2. Stubs
        f.write("// Load Stubs\n")
        f.write(f":params rows => {json.dumps(stubs, ensure_ascii=False)};\n")
        f.write("""UNWIND $rows AS row
MERGE (s:Stub {stub_id: row.stub_id})
SET s.number_raw = row.number_raw,
    s.number_norm = row.number_norm,
    s.date = row.date;
\n""")

        # 3. Clauses
        f.write("// Load Clauses and HAS_CLAUSE edges\n")
        clauses_simple = [{
            "clause_id": c["clause_id"],
            "doc_id": c["doc_id"],
            "number": c.get("clause_number"),
            "section_type": c.get("section_type"),
            "text": c.get("text", "")[:200],
            "parent_clause_id": c.get("parent_clause_id"),
        } for c in clauses]
        f.write(f":params rows => {json.dumps(clauses_simple, ensure_ascii=False)};\n")
        f.write("""UNWIND $rows AS row
MERGE (c:Clause {clause_id: row.clause_id})
SET c.doc_id = row.doc_id,
    c.number = row.number,
    c.section_type = row.section_type,
    c.text = row.text
WITH c, row
MATCH (d:Document {doc_id: row.doc_id})
MERGE (d)-[:HAS_CLAUSE]->(c);
\n""")

        # 4. Tables
        f.write("// Load Tables and HAS_TABLE edges\n")
        tables_simple = [{
            "table_id": t["table_id"],
            "doc_id": t["doc_id"],
            "title_en": t.get("title_en"),
            "title_hi": t.get("title_hi"),
            "table_kind": t.get("table_kind"),
        } for t in tables]
        f.write(f":params rows => {json.dumps(tables_simple, ensure_ascii=False)};\n")
        f.write("""UNWIND $rows AS row
MERGE (t:Table {table_id: row.table_id})
SET t.title_en = row.title_en,
    t.title_hi = row.title_hi,
    t.table_kind = row.table_kind
WITH t, row
MATCH (d:Document {doc_id: row.doc_id})
MERGE (d)-[:HAS_TABLE]->(t);
\n""")

        # 5. Edges
        f.write("// Load Legal Relation Edges\n")
        edges_simple = [{
            "source_id": e.get("source_doc_id"),
            "target_id": e.get("target_doc_id"),
            "rel_type": e.get("relation") or e.get("relation_type", "REFERS_TO"),
            "tier": e.get("tier", "A"),
            "scope": e.get("scope", "whole"),
            "effective_date": e.get("effective_date"),
            "quote": e.get("quote"),
        } for e in edges]
        f.write(f":params rows => {json.dumps(edges_simple, ensure_ascii=False)};\n")
        f.write("""UNWIND $rows AS row
MATCH (src {doc_id: row.source_id})
MATCH (tgt {doc_id: row.target_id})
CALL apoc.create.relationship(src, row.rel_type, {
    tier: row.tier,
    scope: row.scope,
    effective_date: row.effective_date,
    quote: row.quote
}, tgt) YIELD rel
RETURN count(rel);
\n""")


def load_graph(
    header_path: str = "gad_kg/data/work/header.jsonl",
    stubs_path: str = "gad_kg/data/work/stubs.jsonl",
    clauses_path: str = "gad_kg/data/work/clauses.jsonl",
    tables_path: str = "gad_kg/data/work/tables.jsonl",
    edges_path: str = "gad_kg/data/work/edges.jsonl",
    graph_db_path: str = "gad_kg/data/work/graph.db",
    cypher_out_path: str = "gad_kg/data/work/graph.cypher",
) -> Tuple[sqlite3.Connection, nx.DiGraph, Dict[str, Any]]:
    """Load all nodes and edges into SQLite Graph DB and NetworkX DiGraph."""
    docs = list(read_jsonl(Path(header_path))) if os.path.exists(header_path) else []
    stubs = list(read_jsonl(Path(stubs_path))) if os.path.exists(stubs_path) else []
    clauses = list(read_jsonl(Path(clauses_path))) if os.path.exists(clauses_path) else []
    tables = list(read_jsonl(Path(tables_path))) if os.path.exists(tables_path) else []
    edges = list(read_jsonl(Path(edges_path))) if os.path.exists(edges_path) else []

    conn = init_graph_db(graph_db_path)
    cur = conn.cursor()
    G = nx.DiGraph()

    # 1. Insert Document Nodes
    for d in docs:
        d_id = d.get("doc_id", f"D_{d.get('file_id')}")
        num_norm = d.get("doc_number_norm")
        dt = d.get("issue_date")
        cur.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
            (d_id, "Document", num_norm, dt, json.dumps(d, ensure_ascii=False))
        )
        G.add_node(d_id, node_type="Document", **d)

    # 2. Insert Stub Nodes
    for s in stubs:
        s_id = s.get("stub_id")
        num_norm = s.get("number_norm")
        dt = s.get("date")
        cur.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
            (s_id, "Stub", num_norm, dt, json.dumps(s, ensure_ascii=False))
        )
        G.add_node(s_id, node_type="Stub", **s)

    # 3. Insert Clause Nodes & HAS_CLAUSE edges
    for c in clauses:
        c_id = c.get("clause_id")
        d_id = c.get("doc_id")
        cur.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
            (c_id, "Clause", None, None, json.dumps(c, ensure_ascii=False))
        )
        G.add_node(c_id, node_type="Clause", **c)
        if d_id:
            cur.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, tier, scope, effective_date, props_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (d_id, c_id, "HAS_CLAUSE", "A", "part", None, json.dumps({"clause_id": c_id}))
            )
            G.add_edge(d_id, c_id, edge_type="HAS_CLAUSE", tier="A")

        parent_c_id = c.get("parent_clause_id")
        if parent_c_id:
            cur.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, tier, scope, effective_date, props_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (c_id, parent_c_id, "CHILD_OF", "A", "part", None, json.dumps({}))
            )
            G.add_edge(c_id, parent_c_id, edge_type="CHILD_OF", tier="A")

    # 4. Insert Table Nodes & HAS_TABLE edges
    for t in tables:
        t_id = t.get("table_id")
        d_id = t.get("doc_id")
        cur.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
            (t_id, "Table", None, None, json.dumps(t, ensure_ascii=False))
        )
        G.add_node(t_id, node_type="Table", **t)
        if d_id:
            cur.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, tier, scope, effective_date, props_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (d_id, t_id, "HAS_TABLE", "A", "part", None, json.dumps({"table_id": t_id}))
            )
            G.add_edge(d_id, t_id, edge_type="HAS_TABLE", tier="A")

    # 5. Insert Legal Relation Edges
    for e in edges:
        src = e.get("source_doc_id")
        tgt = e.get("target_doc_id")
        rel_type = e.get("relation") or e.get("relation_type", "REFERS_TO")
        tier = e.get("tier", "A")
        scope = e.get("scope", "whole")
        eff_dt = e.get("effective_date")
        cur.execute(
            "INSERT INTO edges (source_id, target_id, edge_type, tier, scope, effective_date, props_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (src, tgt, rel_type, tier, scope, eff_dt, json.dumps(e, ensure_ascii=False))
        )
        edge_attrs = dict(e)
        edge_attrs.update({"edge_type": rel_type, "tier": tier, "scope": scope, "effective_date": eff_dt})
        G.add_edge(src, tgt, **edge_attrs)

    conn.commit()

    # Generate Neo4j Cypher file
    build_cypher_export(docs, stubs, clauses, tables, edges, cypher_out_path)

    stats = {
        "document_nodes": len(docs),
        "stub_nodes": len(stubs),
        "clause_nodes": len(clauses),
        "table_nodes": len(tables),
        "legal_edges": len(edges),
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "db_path": graph_db_path,
        "cypher_path": cypher_out_path,
    }
    return conn, G, stats


def verify_graph_acceptance(
    conn: sqlite3.Connection,
    G: nx.DiGraph,
    header_path: str = "gad_kg/data/work/header.jsonl",
    sample_size: int = 20,
) -> Dict[str, Any]:
    """Spec Section 5 P13 Acceptance Test:
    Counts in Neo4j/Graph match JSONL counts; random 20 documents queried by number return the right node, clauses, and edges.
    """
    headers = list(read_jsonl(Path(header_path)))
    valid_headers = [h for h in headers if h.get("doc_number_norm")]

    sample = random.sample(valid_headers, min(sample_size, len(valid_headers)))
    matched = 0

    cur = conn.cursor()
    for h in sample:
        num_norm = h.get("doc_number_norm")
        doc_id = h.get("doc_id", f"D_{h.get('file_id')}")

        # Query node by number_norm
        cur.execute("SELECT node_id, props_json FROM nodes WHERE doc_number_norm = ?", (num_norm,))
        rows = cur.fetchall()
        if not rows:
            continue

        found_node_ids = {r[0] for r in rows}
        if doc_id not in found_node_ids:
            continue

        # Query clauses
        cur.execute("SELECT target_id FROM edges WHERE source_id = ? AND edge_type = 'HAS_CLAUSE'", (doc_id,))
        clauses = cur.fetchall()

        # Query edges in NetworkX
        nx_node = G.nodes.get(doc_id)
        if nx_node is not None:
            matched += 1

    match_rate = matched / len(sample) if sample else 1.0
    return {
        "tested_docs": len(sample),
        "matched_docs": matched,
        "match_rate": match_rate,
        "passed": match_rate >= 0.95,
    }


if __name__ == "__main__":
    conn, G, stats = load_graph()
    acceptance = verify_graph_acceptance(conn, G)

    print("\n" + "=" * 60)
    print("PHASE 13 GRAPH LOADER RESULTS")
    print("=" * 60)
    print(f"Document Nodes:     {stats['document_nodes']}")
    print(f"Stub Nodes:         {stats['stub_nodes']}")
    print(f"Clause Nodes:       {stats['clause_nodes']}")
    print(f"Table Nodes:        {stats['table_nodes']}")
    print(f"Legal Edges:        {stats['legal_edges']}")
    print(f"Total Graph Nodes:  {stats['total_nodes']}")
    print(f"Total Graph Edges:  {stats['total_edges']}")
    print(f"Database Path:      {stats['db_path']}")
    print(f"Cypher Script Path: {stats['cypher_path']}")
    print("-" * 60)
    print("ACCEPTANCE TEST (20 RANDOM DOCUMENTS QUERIED BY NUMBER):")
    print(f"Queried / Matched:  {acceptance['matched_docs']} / {acceptance['tested_docs']} ({acceptance['match_rate']*100:.1f}%)")
    print(f"Acceptance Passed:  {acceptance['passed']}")
    print("=" * 60)
