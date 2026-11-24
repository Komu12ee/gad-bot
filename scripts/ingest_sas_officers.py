"""
Ingest SAS (State Administrative Service) Civil List Officers into officers.db and temporal_graph.db.
Document: 63dfd3acb3182dc5.md (202600089_002.pdf - CIVIL LIST OF SAS OFFICERS As On 25.03.2026)
"""

import os
import re
import sqlite3
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_MD = REPO_ROOT / "gad_kg" / "data" / "raw_md" / "63dfd3acb3182dc5.md"
OFFICERS_DB = REPO_ROOT / "gad_kg" / "data" / "work" / "officers.db"
TEMPORAL_DB = REPO_ROOT / "data" / "fg2" / "temporal_graph.db"
URL_INDEX_PATH = REPO_ROOT / "gad_kg" / "data" / "work" / "doc_url_index.json"

# Transliteration dictionary for common Indian civil service names
TRANSLIT_MAP = {
    "anupam": "अनुपम", "ashish": "आशीष", "asish": "आशीष", "toppo": "टोप्पो",
    "saumya": "सौम्या", "chaurasia": "चौरसिया", "aarti": "आरती", "vasnik": "वासनिक",
    "dev": "देव", "narayan": "नारायण", "kashyap": "कश्यप", "chhannulal": "छन्नूलाल",
    "markandey": "मार्कण्डेय", "surendra": "सुरेंद्र", "prasad": "प्रसाद", "vaidya": "वैद्य",
    "abhishek": "अभिषेक", "agrawal": "अग्रवाल", "sachin": "सचिन", "bhutada": "भूतड़ा",
    "rajeev": "राजीव", "kumar": "कुमार", "pandey": "पांडेय", "prem": "प्रेम",
    "prakash": "प्रकाश", "sharma": "शर्मा", "arvind": "अरविंद", "sunil": "सुनील",
    "nayak": "नायक", "pradeep": "प्रदीप", "sahu": "साहू", "nishtha": "निष्ठा",
    "tiwari": "तिवारी", "chandrakant": "चंद्रकांत", "koushik": "कौशिक", "suryakiran": "सूर्यकिरण",
    "tekchand": "टेकचंद", "avinash": "अविनाश", "bhoi": "भोई", "patel": "पटेल",
    "durgesh": "दुर्गेश", "verma": "वर्मा", "neelam": "नीलम", "apoorv": "अपूर्व",
    "priyesh": "प्रियेश", "vinay": "विनय", "arpita": "अर्पिता", "pathak": "पाठक",
    "rahul": "राहुल", "rajak": "रजक", "subrat": "सुब्रत", "pradhan": "प्रधान",
    "uttam": "उत्तम", "singh": "सिंह", "panchari": "पंचारी", "saroj": "सरोज",
    "mahilange": "महिलंगे", "seema": "सीमा", "patre": "पात्रे", "kaveri": "कावेरी",
    "markam": "मरकाम", "vivek": "विवेक", "chandra": "चंद्रा", "omkareshwar": "ओंकारेश्वर",
    "hitesh": "हितेश", "pisda": "पिसदा", "mukesh": "मुकेश", "kothari": "कोठारी",
    "lavkesh": "लवकेश", "dhruv": "ध्रुव", "kalpana": "कल्पना", "rekha": "रेखा",
    "sanjay": "संजय", "praveen": "प्रवीण", "bhagat": "भगत", "amit": "अमित",
    "nath": "नाथ", "yogi": "योगी", "bajrang": "बजरंग", "aniket": "अनिकेत",
}


def transliterate_to_hi(name_en: str) -> str:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", name_en)]
    hi_tokens = [TRANSLIT_MAP.get(t, t.capitalize()) for t in tokens]
    return "श्री " + " ".join(hi_tokens)


def parse_sas_officers():
    if not RAW_MD.exists():
        print(f"File not found: {RAW_MD}")
        return []

    text = RAW_MD.read_text(encoding="utf-8")
    officers = {}

    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if len(cols) >= 8:
            sn = cols[0]
            if re.match(r"^\d+$", sn):
                name = cols[1]
                if name and not name.lower().startswith("name of"):
                    officers[sn] = {
                        "sn": sn, "name": name, "email": cols[2], "qual": cols[3],
                        "dob": cols[4], "domicile": cols[5], "appt_date": cols[6], "post": cols[7]
                    }
            elif " / " in cols[0] and " / " in cols[1]:
                sns = [s.strip() for s in cols[0].split("/")]
                names = [s.strip() for s in cols[1].split("/")]
                emails = [s.strip() for s in cols[2].split("/")]
                quals = [s.strip() for s in cols[3].split("/")] if len(cols) > 3 else []
                dobs = [s.strip() for s in cols[4].split("/")] if len(cols) > 4 else []
                doms = [s.strip() for s in cols[5].split("/")] if len(cols) > 5 else []
                appts = [s.strip() for s in cols[6].split("/")] if len(cols) > 6 else []
                posts = [s.strip() for s in cols[7].split("/")] if len(cols) > 7 else []

                n = min(len(sns), len(names))
                for i in range(n):
                    if names[i] and not names[i].lower().startswith("name of"):
                        officers[sns[i]] = {
                            "sn": sns[i],
                            "name": names[i],
                            "email": emails[i] if i < len(emails) else "",
                            "qual": quals[i] if i < len(quals) else "",
                            "dob": dobs[i] if i < len(dobs) else "",
                            "domicile": doms[i] if i < len(doms) else "",
                            "appt_date": appts[i] if i < len(appts) else "",
                            "post": posts[i] if i < len(posts) else ""
                        }

    return list(officers.values())


def ingest():
    officers = parse_sas_officers()
    print(f"Parsed {len(officers)} unique SAS officers from {RAW_MD.name}")

    # Load source URL info
    source_url = "https://gad.cg.gov.in/statecirculars/Criculars/Scan-file-2000-2002/year2026/202600089_002.pdf"
    doc_id = "63dfd3acb3182dc5"
    can_id = "4a9f4dd585434d158310787a3389b7eb"
    doc_title = "CIVIL LIST OF SAS OFFICERS As On 25.03.2026"
    filename = "202600089_002.pdf"

    for db_path in [OFFICERS_DB, TEMPORAL_DB]:
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        inserted = 0
        for off in officers:
            name_en = off["name"].strip()
            name_hi = transliterate_to_hi(name_en)
            slug = re.sub(r"[^a-z0-9]+", "_", name_en.lower()).strip("_")
            off_id = f"sas_{slug}"

            # Post / designation
            post_desc = off["post"].strip()
            if not post_desc:
                desig_en = "State Administrative Service Officer"
                desig_hi = "राज्य प्रशासनिक सेवा अधिकारी"
            else:
                desig_en = f"State Administrative Service Officer ({post_desc})"
                desig_hi = f"राज्य प्रशासनिक सेवा अधिकारी ({post_desc})"

            add_charges = []
            if off.get("dob"):
                add_charges.append(f"जन्म तिथि: {off['dob']}")
            if off.get("domicile"):
                add_charges.append(f"गृह जिला: {off['domicile']}")
            if off.get("appt_date"):
                add_charges.append(f"रा.प्र.से. नियुक्ति: {off['appt_date']}")
            if off.get("qual"):
                add_charges.append(f"शैक्षणिक योग्यता: {off['qual']}")

            charges_str = " | ".join(add_charges)

            cur.execute("""
                INSERT OR REPLACE INTO officers (
                    id, name_hi, name_en, category, category_label,
                    designation_hi, designation_en, department_hi, department_en,
                    additional_charges, email, phone, status,
                    source_doc_id, canonical_source_id, source_name, source_title,
                    source_url, filename, notification_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                off_id, name_hi, name_en, "sas_officers",
                "राज्य प्रशासनिक सेवा (State Administrative Service - SAS)",
                desig_hi, desig_en, "सामान्य प्रशासन विभाग (GAD)", "General Administration Department (GAD)",
                charges_str, off.get("email", ""), "", "active",
                doc_id, can_id, "CIVIL LIST OF SAS OFFICERS", doc_title,
                source_url, filename, 1
            ))

            # Add officer notification entry
            cur.execute("""
                INSERT OR REPLACE INTO officer_notifications (
                    officer_id, doc_id, canonical_doc_id, source_url, filename,
                    order_no, order_date, subject, doc_title, excerpt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                off_id, doc_id, can_id, source_url, filename,
                "CIVIL LIST OF SAS OFFICERS", "25.03.2026",
                doc_title, doc_title,
                f"रा.प्र.से. पदक्रम सूची सरल क्र. {off['sn']}: {name_en} ({name_hi}), जन्म तिथि: {off.get('dob')}, पदस्थापना: {off.get('post')}, ई-मेल: {off.get('email')}"
            ))

            # Update FTS
            try:
                cur.execute("DELETE FROM fts_officers WHERE officer_id = ?", (off_id,))
            except Exception:
                pass

            try:
                cur.execute("""
                    INSERT INTO fts_officers (
                        officer_id, name_hi, name_en, designation_hi, designation_en,
                        department_hi, department_en, email
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    off_id, name_hi, name_en, desig_hi, desig_en,
                    "सामान्य प्रशासन विभाग", "General Administration Department", off.get("email", "")
                ))
            except Exception as e:
                pass

            inserted += 1

        conn.commit()
        conn.close()
        print(f"Successfully ingested {inserted} SAS officers into {db_path.name}")


if __name__ == "__main__":
    ingest()
