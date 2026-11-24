"""
JSON Schemas for Phase 4 Header Extraction Prompts (R1 - R8)
Spec: Section 4.3, Section 6
"""

R1_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {
            "type": "string",
            "enum": ["GR", "CIRCULAR", "NOTIFICATION", "ORDER", "LETTER", "CORRIGENDUM", "RULES", "OTHER"]
        },
        "quote": {"type": ["string", "null"]}
    },
    "required": ["doc_type"],
    "additionalProperties": False
}

R2_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": ["integer", "null"]},
        "quote": {"type": ["string", "null"]}
    },
    "required": ["choice"],
    "additionalProperties": False
}

R3_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": ["integer", "null"]},
        "quote": {"type": ["string", "null"]}
    },
    "required": ["choice"],
    "additionalProperties": False
}

R4_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": ["string", "null"]},
        "subject_en": {"type": ["string", "null"]}
    },
    "required": ["subject"],
    "additionalProperties": False
}

R5_SCHEMA = {
    "type": "object",
    "properties": {
        "number_raw": {"type": ["string", "null"]},
        "date_raw": {"type": ["string", "null"]},
        "subject_raw": {"type": ["string", "null"]},
        "doc_type_guess": {
            "type": "string",
            "enum": ["GR", "CIRCULAR", "NOTIFICATION", "ORDER", "LETTER", "CORRIGENDUM", "RULES", "ACT", "COURT_ORDER", "OTHER"]
        }
    },
    "required": ["doc_type_guess"],
    "additionalProperties": False
}

R6_SCHEMA = {
    "type": "object",
    "properties": {
        "quote": {"type": "string"},
        "kind": {
            "type": "string",
            "enum": ["EFFECTIVE_FROM", "EFFECTIVE_TO", "NONE"]
        },
        "date_raw": {"type": ["string", "null"]},
        "retrospective": {"type": "boolean"}
    },
    "required": ["quote", "kind", "retrospective"],
    "additionalProperties": False
}

R7_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3
        }
    },
    "required": ["topic_ids"],
    "additionalProperties": False
}

R8_SCHEMA = {
    "type": "object",
    "properties": {
        "summary_en": {"type": "string"},
        "summary_hi": {"type": "string"}
    },
    "required": ["summary_en", "summary_hi"],
    "additionalProperties": False
}
