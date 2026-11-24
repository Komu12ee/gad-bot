"""
Phase 11: Entity Mentions & Controlled Ontology Schemas
Spec: Section 5 P11, Section 6 Prompt R13
"""

R13_SCHEMA = {
    "type": "object",
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["POST", "SCHEME", "INSTRUMENT", "BENEFIT", "DEPARTMENT"]
                    }
                },
                "required": ["text", "kind"],
                "additionalProperties": False
            }
        }
    },
    "required": ["mentions"],
    "additionalProperties": False
}
