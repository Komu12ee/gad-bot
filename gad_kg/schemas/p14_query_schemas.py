"""
Schemas for Phase 14: Query Service & Answering Engine
Spec: Section 4, Section 6 Prompts R14 & R16, Section 9
"""

R14_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {
            "type": "string",
            "enum": ["hi", "en", "mixed"]
        },
        "intent": {
            "type": "string",
            "enum": [
                "LOOKUP",
                "CURRENT_RULE",
                "AS_OF_RULE",
                "HISTORY",
                "STATUS",
                "DOC_LOOKUP",
                "LIST",
                "TABLE_QUERY",
                "COMPARE",
                "OTHER"
            ]
        },
        "as_of_date": {
            "type": ["string", "null"]
        },
        "doc_refs": {
            "type": "array",
            "items": {"type": "string"}
        },
        "filters": {
            "type": "object",
            "properties": {
                "doc_type": {"type": ["string", "null"]},
                "year_from": {"type": ["integer", "null"]},
                "year_to": {"type": ["integer", "null"]},
                "topic_hint": {"type": ["string", "null"]}
            },
            "required": ["doc_type", "year_from", "year_to", "topic_hint"],
            "additionalProperties": False
        },
        "queries_en": {
            "type": "array",
            "items": {"type": "string"}
        },
        "queries_hi": {
            "type": "array",
            "items": {"type": "string"}
        },
        "needs_table": {
            "type": "boolean"
        }
    },
    "required": ["language", "intent", "as_of_date", "doc_refs", "filters", "queries_en", "queries_hi", "needs_table"],
    "additionalProperties": False
}

R16_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string"
        },
        "used_labels": {
            "type": "array",
            "items": {"type": "string"}
        },
        "answerable": {
            "type": "boolean"
        }
    },
    "required": ["answer", "used_labels", "answerable"],
    "additionalProperties": False
}
