"""
Phase 12: Contextual Prefix & Retrieval Schemas
Spec: Section 5 P12, Section 6 Prompt R9
"""

R9_SCHEMA = {
    "type": "object",
    "properties": {
        "covers": {"type": "string"},
        "keywords_en": {
            "type": "array",
            "items": {"type": "string"}
        },
        "keywords_hi": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["covers", "keywords_en", "keywords_hi"],
    "additionalProperties": False
}
