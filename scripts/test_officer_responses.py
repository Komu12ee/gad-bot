import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gad_kg" / "src"))

from officer_store import OfficerStore

store = OfficerStore()

def answer_officer_query(q: str):
    q_low = q.lower()
    
    # 1. Chief Secretary
    if any(k in q_low for k in ["chief secretary", "मुख्य सचिव"]):
        cs = store.get_chief_secretary()
        if cs:
            cs = store.get_officer(cs['id']) or cs
            ans = f"### मुख्य सचिव (Chief Secretary) - छत्तीसगढ़ शासन\n\n"
            ans += f"**{cs['name_hi']} ({cs['name_en']})** छत्तीसगढ़ शासन के वर्तमान **मुख्य सचिव (Chief Secretary)** हैं।\n\n"
            ans += f"- **पदनाम:** {cs['designation_hi']} ({cs['designation_en']})\n"
            ans += f"- **विभाग:** {cs['department_hi']} ({cs['department_en']})\n"
            ans += f"- **ई-मेल:** `{cs['email']}`\n"
            ans += f"- **स्रोतः** {cs['source_name']} ([Display_civil_list_Email.aspx]({cs['source_url']}))\n\n"
            if cs['notifications']:
                ans += "#### संबंधित प्रमुख शासकीय अधिसूचनाएं एवं परिपत्र:\n"
                for n in cs['notifications'][:5]:
                    ans += f"- **{n['order_no']}** ({n['order_date']}): {n['subject']}\n"
            return ans

    # 2. List Secretaries
    if "list" in q_low and any(k in q_low for k in ["secretaries", "सचिव"]):
        secs = store.get_secretaries()
        ans = f"### छत्तीसगढ़ शासन के प्रमुख सचिवों एवं अपर मुख्य सचिवों की सूची (Secretaries to Govt of CG):\n\n"
        ans += f"वर्तमान राज्य सिविल लिस्ट एवं सामान्य प्रशासन विभाग के अनुसार प्रमुख सचिव निम्नलिखित हैं:\n\n"
        ans += "| क्र. | नाम (Officer Name) | पदनाम (Designation) | विभाग (Department) | ई-मेल (Email) |\n"
        ans += "|---|---|---|---|---|\n"
        for i, s in enumerate(secs[:25], 1):
            ans += f"| {i} | **{s['name_hi']}** ({s['name_en']}) | {s['designation_hi']} | {s['department_hi'] or 'मंत्रालय'} | `{s['email']}` |\n"
        ans += f"\n*(कुल {len(secs)} सचिवीय अधिकारी दर्ज हैं। पूरा विवरण आधिकारिक सिविल लिस्ट डायरेक्टरी में उपलब्ध है।)*\n"
        return ans

    # 3. Retired IAS
    if any(k in q_low for k in ["retired ias", "सेवानिवृत्त ias", "सेवानिवृत्त भा.प्र.से"]):
        ret = store.get_retired_ias()
        ans = f"### सेवानिवृत्त भारतीय प्रशासनिक सेवा (Retired IAS) अधिकारी - छत्तीसगढ़ संवर्ग:\n\n"
        ans += f"सामान्य प्रशासन विभाग के अधिवार्षिकी सेवानिवृत्ति आदेश **क्रमांक ई 3-01/2003/एक/2** के अनुसार सेवानिवृत्त आईएएस अधिकारियों की सूची:\n\n"
        ans += "| क्र. | अधिकारी का नाम | बैच | पूर्व पदस्थापना | सेवानिवृत्ति तिथि |\n"
        ans += "|---|---|---|---|---|\n"
        for i, r in enumerate(ret, 1):
            batch = next((c.replace("बैच: ", "") for c in r.get("additional_charges", []) if "बैच" in c), "N/A")
            dt = next((c.replace("सेवानिवृत्ति तिथि: ", "") for c in r.get("additional_charges", []) if "सेवानिवृत्ति" in c), "N/A")
            ans += f"| {i} | **{r['name_hi']}** ({r['name_en']}) | {batch} | {r['designation_hi'].split(' - पूर्व ')[-1]} | {dt} |\n"
        return ans

    # 4. CHiPS CEOs
    if any(k in q_low for k in ["ceo of chips", "chips ceo", "chhattisgarh infotech", "चिप्स"]):
        ceos = store.get_chips_ceos()
        ans = f"### छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (CHiPS) - नेतृत्व एवं पूर्व मुख्य कार्यपालन अधिकारी (CEOs):\n\n"
        ans += f"शासकीय अधिसूचनाओं, पदक्रम सूची व मंत्रालय अभिलेखों के अनुसार चिप्स (CHiPS) के प्रमुख अधिकारी:\n\n"
        for c in ceos:
            ans += f"- **{c['name_hi']} ({c['name_en']})**: {c['designation_hi']}\n"
            if c['additional_charges']:
                ans += f"  - *प्रभार:* {', '.join(c['additional_charges'])}\n"
            if c['email']:
                ans += f"  - *ई-मेल:* `{c['email']}`\n"
        return ans

    # 5. IAS Establishment Section
    if any(k in q_low for k in ["ias establishment", "स्थापना", "establishment section"]):
        est = store.get_ias_establishment()
        ans = f"### {est['section']}\n\n"
        ans += f"**विभाग:** {est['department']}\n"
        ans += f"**कार्यक्षेत्र:** {est['mandate']}\n\n"
        ans += "#### प्रमुख कार्य एवं दायित्व (Key Functions):\n"
        for fn in est['key_functions']:
            ans += f"- {fn}\n"
        return ans

    # 6. Specific Officer Search
    matches = store.search_officers(q, limit=3)
    if matches:
        top = matches[0]
        ans = f"### {top['name_hi']} ({top['name_en']})\n\n"
        ans += f"- **पदनाम (Designation):** {top['designation_hi']} ({top['designation_en']})\n"
        ans += f"- **विभाग (Department):** {top['department_hi']} ({top['department_en']})\n"
        ans += f"- **संवर्ग श्रेणी (Category):** {top['category_label']}\n"
        if top['additional_charges']:
            ans += f"- **अतिरिक्त प्रभार (Additional Charges):** {', '.join(top['additional_charges'])}\n"
        if top['email']:
            ans += f"- **ई-मेल (Official Email):** `{top['email']}`\n"
        if top['phone']:
            ans += f"- **संपर्क (Contact):** {top['phone']}\n"
        ans += f"- **अभिलेख स्रोतः** {top['source_name']}\n\n"

        if top['notifications']:
            ans += "#### संबंधित शासकीय परिपत्र एवं स्थानांतरण/पदस्थापना आदेश:\n"
            for n in top['notifications'][:6]:
                ans += f"- **आदेश {n['order_no']}** ({n['order_date']}): {n['subject']}\n"
                if n['excerpt']:
                    ans += f"  > *\"{n['excerpt'][:180]}...\"*\n"
        return ans

    return None

if __name__ == "__main__":
    for test_q in [
        "abhi chief secretary of Government of Chhattisgarh kon hain?",
        "Who is Sammi Abedi?",
        "Who is the secretary of the Forest Department?",
        "Who is the secretary of the Revenue and Disaster Department?",
        "tell me abt Shri Chandan Kumar",
        "who is ceo of chips",
        "please provide me the current list of secretaries of Govt of Chhattisgarh",
        "Give me a list of retired IaS.",
        "What is in the IaS establishment section?"
    ]:
        print("=" * 60)
        print("Q:", test_q)
        print("=" * 60)
        print(answer_officer_query(test_q))
