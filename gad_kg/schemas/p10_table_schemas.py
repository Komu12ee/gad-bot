"""
Phase 10: Table Extraction and Structured Store Schemas
Spec: Section 4.7, Section 5 P10, Section 6 Prompt R12
"""

R12_SCHEMA = {
    "type": "object",
    "properties": {
        "title_en": {"type": "string"},
        "title_hi": {"type": ["string", "null"]},
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name_raw": {"type": "string"},
                    "name_en": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["text", "int", "float", "date", "amount", "percent"]
                    },
                    "unit": {"type": ["string", "null"]}
                },
                "required": ["name_raw", "name_en", "type", "unit"],
                "additionalProperties": False
            }
        },
        "table_kind": {
            "type": "string",
            "enum": [
                "list_of_posts",
                "pay_scale",
                "allowance_rates",
                "eligibility",
                "schedule",
                "annexure_list",
                "other"
            ]
        }
    },
    "required": ["title_en", "title_hi", "columns", "table_kind"],
    "additionalProperties": False
}
