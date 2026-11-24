import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
work_dir = ROOT / "data" / "work"
work_dir.mkdir(parents=True, exist_ok=True)

questions = [
    {"id": "Q01", "question": "please provide me the current list of secretaries of Govt of Chhattisgarh", "language": "en", "intent": "LIST", "entity_type": "Post", "target_post": "Secretary"},
    {"id": "Q02", "question": "abhi chief secretary of Government of Chhattisgarh kon hain?", "language": "mixed", "intent": "CURRENT_RULE", "entity_type": "Post", "target_post": "Chief Secretary"},
    {"id": "Q03", "question": "Who is Sammi Abedi?", "language": "en", "intent": "LOOKUP", "entity_type": "Person/Officer", "target_person": "Sammi Abedi"},
    {"id": "Q04", "question": "Who is the secretary of the Revenue and Disaster Department?", "language": "en", "intent": "CURRENT_RULE", "entity_type": "Department/Post", "target_department": "Revenue and Disaster Management"},
    {"id": "Q05", "question": "Who is the secretary of the Forest Department?", "language": "en", "intent": "CURRENT_RULE", "entity_type": "Department/Post", "target_department": "Forest Department"},
    {"id": "Q06", "question": "Give me the department where this transfer is occurring.", "language": "en", "intent": "DOC_LOOKUP", "entity_type": "Transfer"},
    {"id": "Q07", "question": "tell me abt Shri Chandan Kumar", "language": "mixed", "intent": "LOOKUP", "entity_type": "Person/Officer", "target_person": "Chandan Kumar"},
    {"id": "Q08", "question": "list all the CEO of chhattisgarh infotech promotion society before , list them all", "language": "en", "intent": "HISTORY", "entity_type": "Department/Post", "target_post": "CEO CHiPS"},
    {"id": "Q09", "question": "who is CEO of chhattisgarh enfotech promotion society?", "language": "en", "intent": "CURRENT_RULE", "entity_type": "Post", "target_post": "CEO CHiPS"},
    {"id": "Q10", "question": "who is ceo of chips", "language": "en", "intent": "CURRENT_RULE", "entity_type": "Post", "target_post": "CEO CHiPS"},
    {"id": "Q11", "question": "Give me a list of retired IaS.", "language": "en", "intent": "LIST", "entity_type": "Cadre", "target_cadre": "Retired IAS"},
    {"id": "Q12", "question": "35 स्वतंत्रता संग्राम सेनानी/ लोक सेनानी(मीसाबंदी)", "language": "hi", "intent": "LOOKUP", "entity_type": "Scheme/Benefit", "target_scheme": "Lok Senani Misabandi"},
    {"id": "Q13", "question": "Summarize the data entry status.", "language": "en", "intent": "TABLE_QUERY", "entity_type": "Status"},
    {"id": "Q14", "question": "What is in the IaS establishment section?", "language": "en", "intent": "LOOKUP", "entity_type": "Department/Branch", "target_branch": "IAS Establishment"}
]

out_path = work_dir / "failing_chatbot_questions.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"Saved {len(questions)} failing chatbot questions to {out_path}")
