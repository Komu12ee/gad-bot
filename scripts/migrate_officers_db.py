import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gad_kg" / "src"))

from officer_store import OFFICERS_JSON, persist_officer_store

with open(OFFICERS_JSON, "r", encoding="utf-8") as f:
    officers = json.load(f)

print(f"Persisting and migrating {len(officers)} officers with document metadata...")
persist_officer_store(officers)
print("Migration completed successfully!")
