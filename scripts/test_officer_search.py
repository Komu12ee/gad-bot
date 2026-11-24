import sys
from pathlib import Path

# Add gad_kg/src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gad_kg" / "src"))
from officer_store import OfficerStore

def test_queries():
    store = OfficerStore()
    queries = [
        "Sammi Abedi",
        "Who is Sammi Abedi?",
        "secretary of the Revenue and Disaster Department",
        "secretary of the Forest Department",
        "abhi chief secretary of Government of Chhattisgarh kon hain?",
        "tell me abt Shri Chandan Kumar",
        "who is ceo of chips",
        "retired ias"
    ]
    for q in queries:
        res = store.search_officers(q)
        print(f'Query: "{q}" -> {len(res)} matches')
        if res:
            top = res[0]
            print(f'   Top match: {top["name_en"]} ({top["name_hi"]}) | {top["designation_hi"]} | Dept: {top["department_hi"]} | Charges: {top["additional_charges"]}')
            print(f'   Notifications linked: {len(top["notifications"])}')

if __name__ == "__main__":
    test_queries()
