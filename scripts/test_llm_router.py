import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gad_kg"))

from src.common.llm import call_llm

_ROUTER_CACHE = {}

def route_query_intent(query: str) -> str:
    q_norm = query.strip().lower()
    if q_norm in _ROUTER_CACHE:
        return _ROUTER_CACHE[q_norm]

    prompt = (
        "You are an intent classifier for a Government Administration Department (GAD) knowledge system.\n"
        "Determine whether the user query is asking about:\n"
        '- "OFFICER": Inquiries about specific officers, personnel, people, civil servants, IAS/SAS officers, Secretaries, Collectors, SPs, DFOs, CHiPS leadership, or government personnel directories.\n'
        '- "DOCUMENT": Inquiries about policies, leave rules, circulars, notifications, transfer orders/letters, pension schemes, RTI, acts, rules, government procedures, or general administrative documents.\n\n'
        f"Query: {query}\n\n"
        "Output JSON matching schema."
    )
    schema = {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["OFFICER", "DOCUMENT"]}
        },
        "required": ["route"]
    }
    try:
        res = call_llm(prompt, schema=schema, schema_name="router_schema", temperature=0.0, max_tokens=25)
        route = res.get("route", "DOCUMENT")
    except Exception as e:
        print("Router fallback:", e)
        route = "DOCUMENT"

    _ROUTER_CACHE[q_norm] = route
    return route

queries = [
    "Can you tell me about the leave policy?",
    "Can you tell me about the latest transfer order letter?",
    "who is the anupam asish toppo",
    "who is mayank agrawal?",
    "what are the rules for earned leave?",
    "what is in the IaS establishment section?",
    "tell me abt Shri Chandan Kumar"
]

for q in queries:
    r = route_query_intent(q)
    print(f"'{q}' => {r}")
