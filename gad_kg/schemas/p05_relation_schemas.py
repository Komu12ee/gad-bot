"""
Phase 5 JSON Schemas: Relation Extraction (R10) and Verification (R11)
Spec: Section 4.4, Section 5 P5 & P6, Section 6 Prompts R10, R11
"""

R10_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": [
                            "SUPERSEDES", "SUPERSEDED_BY",
                            "AMENDS", "AMENDED_BY",
                            "RESCINDS", "RESCINDED_BY",
                            "CLARIFIES", "CLARIFIED_BY",
                            "EXTENDS", "EXTENDED_BY",
                            "CORRECTS", "CORRECTED_BY",
                            "IMPLEMENTS", "REFERS_TO", "NONE"
                        ]
                    },
                    "target_kind": {
                        "type": "string",
                        "enum": ["DOCUMENT", "ACT_RULE", "COURT_ORDER", "UNKNOWN"]
                    },
                    "target_ref_idx": {"type": ["integer", "null"]},
                    "target_number_raw": {"type": ["string", "null"]},
                    "target_date_raw": {"type": ["string", "null"]},
                    "target_subject_raw": {"type": ["string", "null"]},
                    "scope": {
                        "type": "string",
                        "enum": ["whole", "partial", "unknown"]
                    },
                    "scope_text": {"type": ["string", "null"]}
                },
                "required": [
                    "quote", "relation", "target_kind", "target_ref_idx",
                    "target_number_raw", "target_date_raw", "target_subject_raw",
                    "scope", "scope_text"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["relations"],
    "additionalProperties": False
}

R11_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "NOT_SUPPORTED", "UNCLEAR"]
        },
        "reason": {"type": "string"}
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False
}
