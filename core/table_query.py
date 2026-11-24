"""Counts over tables — METADATA_LAYER_PLAN.md step 5.

"Durg domicile ke kitne Section Officers hain?" is not in any passage: it is a
count over the rows of a table (a civil list). For a COUNT question (query_plan),
the tables behind the best-ranked table chunks are loaded into an in-memory
SQLite table, the answer model writes ONE read-only SELECT for it (JSON schema,
thinking off), the query runs under guards, and its result — with the rows it
counted — goes to the answer model as one more passage it can cite.

Guards: SELECT only, on table `t` only, one statement, no functions besides a
number parser and SQLite's own aggregates, a row cap and a time limit. A question
the table cannot answer gets no passage (the model says "answerable": false), and
the normal passages still answer.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_TABLES = 3
SHOW_ROWS = 15
SQL_SCHEMA = {"type": "object",
              "properties": {"answerable": {"type": "boolean"},
                             "sql": {"type": "string", "maxLength": 600}},
              "required": ["answerable", "sql"]}
SYSTEM = """You write ONE SQLite SELECT statement over table `t` to answer a counting question.
Columns are c1, c2, … — their meaning is given. Values are OCR text: match words with LIKE '%…%'
(a short distinctive part of the word, in the language the values use), never exact equality.
Numbers: NUM(column) turns "1,25,000.00" into a number. Count with COUNT(*).
If this table cannot answer the question, return answerable=false and sql="".
Return JSON only."""

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|"
                        r"reindex|load_extension|with\s+recursive)\b", re.I)


def _num(v):
    if v is None:
        return None
    s = re.sub(r"[^\d.\-]", "", str(v).translate(str.maketrans("०१२३४५६७८९", "0123456789")))
    try:
        return float(s) if s not in ("", "-", ".") else None
    except ValueError:
        return None


def safe_sql(sql: str) -> str | None:
    """The statement if it is one read-only SELECT on table t, else None."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s.lower().startswith("select") or ";" in s or _FORBIDDEN.search(s):
        return None
    tables = re.findall(r"\b(?:from|join)\s+([A-Za-z_][\w]*)", s, re.I)
    if not tables or any(t.lower() != "t" for t in tables):
        return None
    return s


def load_table(columns: list[str], rows: list[list[str]]) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.create_function("NUM", 1, _num, deterministic=True)
    n = max([len(columns)] + [len(r) for r in rows]) or 1
    con.execute(f"CREATE TABLE t ({', '.join(f'c{i + 1} TEXT' for i in range(n))})")
    con.executemany(f"INSERT INTO t VALUES ({', '.join('?' * n)})",
                    [list(r) + [""] * (n - len(r)) for r in rows])
    return con


def run_sql(con: sqlite3.Connection, sql: str, max_rows: int = 200, max_steps: int = 2_000_000):
    steps = {"n": 0}

    def guard():
        steps["n"] += 1
        return 1 if steps["n"] > max_steps // 1000 else 0       # abort a runaway query

    con.set_progress_handler(guard, 1000)
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description or []]
    return cols, cur.fetchmany(max_rows)


# ── finding the tables behind the retrieved chunks ───────────────────────────
def tables_for(results: list, stage2: Path) -> list[dict]:
    """[{doc_dir, table_group, columns, rows, caption, unit}] for the table chunks
    among the results, best-ranked first; a table that runs over several pages is
    joined into one."""
    import chunk_builder as cb
    out, seen = [], set()
    for r in results:
        p = getattr(r.get("point"), "payload", None) or {}
        if p.get("chunk_type") != "table_rows":
            continue
        key = (p.get("source"), p.get("table_group") or p.get("table_id"))
        if key in seen or not key[0]:
            continue
        seen.add(key)
        md = stage2 / key[0] / "structured.md"
        if not md.exists():
            continue
        group = []
        for pg in cb.page_segments(md.read_text(encoding="utf-8", errors="replace"), stage2 / key[0]):
            for kind, t in pg["segments"]:
                if kind == "table" and (t.get("table_group") or t.get("id")) == key[1]:
                    group.append(t)
        if not group:
            continue
        cols = list(group[0].get("columns") or [])
        rows = [row for t in group for row in (t.get("rows") or []) if any((c or "").strip() for c in row)]
        if not cols and rows and cb._looks_like_header(rows[0]):
            cols, rows = rows[0], rows[1:]
        out.append({"doc_dir": key[0], "table_group": key[1], "columns": cols, "rows": rows,
                    "caption": group[0].get("caption"), "unit": group[0].get("unit")})
        if len(out) >= MAX_TABLES:
            break
    return out


def ask_sql(question: str, table: dict, chat) -> str | None:
    cols = table["columns"] or [f"column {i + 1}" for i in range(max(len(r) for r in table["rows"]))]
    desc = "\n".join(f"c{i + 1}: {c or '(no name)'}" for i, c in enumerate(cols))
    sample = "\n".join(" | ".join(row) for row in table["rows"][:6])
    user = (f"Table: {table.get('caption') or '—'}\nColumns:\n{desc}\n\nFirst rows:\n{sample}\n\n"
            f"Rows in the table: {len(table['rows'])}\n\nQuestion: {question}")
    out = json.loads(chat(SYSTEM, user))
    return safe_sql(out.get("sql")) if out.get("answerable") else None


def evidence(question: str, results: list, stage2: Path, chat, label) -> dict | None:
    """One synthetic result (a passage) holding the count and the rows counted, or
    None. `chat(system, user) -> JSON text`; `label(doc_dir) -> (name, url)`."""
    for table in tables_for(results, stage2):
        if not table["rows"]:
            continue
        try:
            sql = ask_sql(question, table, chat)
            if not sql:
                continue
            con = load_table(table["columns"], table["rows"])
            cols, rows = run_sql(con, sql)
            # the rows a COUNT counted, so the answer can show them
            shown = []
            m = re.search(r"\bwhere\b(.+)$", sql, re.I | re.S)
            if m and re.search(r"count\s*\(", sql, re.I):
                _, shown = run_sql(con, f"SELECT * FROM t WHERE {m.group(1)}", max_rows=SHOW_ROWS)
        except (sqlite3.Error, ValueError, json.JSONDecodeError) as e:
            print(f"  ⚠ table query skipped ({type(e).__name__}: {e})")
            continue
        name, _ = label(table["doc_dir"])
        result = "; ".join(" | ".join(str(v) for v in row) for row in rows[:10])
        # Neutral wording on purpose: a query over OCR'd text can miss a spelling
        # variant, so the answer must rest on the rows shown, not on "the database".
        lines = [f"Rows of the table \"{table.get('caption') or name}\" ({len(table['rows'])} rows in all) "
                 f"that match the question, found by a query over the table — the rows are listed "
                 f"below; OCR spelling variants may be missed:",
                 f"Query: {sql}", f"Result: {result}"]
        if shown:
            hdr = " | ".join(table["columns"]) if table["columns"] else ""
            lines += ["Rows counted" + (f" ({hdr})" if hdr else "") + ":"] + [" | ".join(r) for r in shown]
            if len(shown) == SHOW_ROWS:
                lines.append("…")
        text = "# Chunk sql\nDocument: {d}\nPages: Unknown\nTokens: 0\nHeadings: []\nType: table_count\n\n---\n\n{b}".format(
            d=table["doc_dir"], b="\n".join(lines))
        print(f"  🧮 table count: {sql} -> {result}")
        point = SimpleNamespace(id=f"sql:{table['doc_dir']}:{table['table_group']}",
                                payload={"text": text, "source": table["doc_dir"], "chunk": "sql",
                                         "chunk_type": "table_count"})
        return {"point": point, "score": 1.0, "rank": 0}
    return None
