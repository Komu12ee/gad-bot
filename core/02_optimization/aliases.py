"""Fixed name lists — districts, posts, departments — and a matcher for them.

Used to tag documents (doc_metadata.py) and, later, to read questions: a question
typed in Latin script ("Durg", "E&IT department", "Under Secretary") must find the
Devanagari names the documents use, and the reverse.

Each entry: canonical id -> (Hindi canonical, English canonical, [other spellings]).
Spellings cover the OCR/typing variants seen in this corpus (ँ/ं, ण्ड/ंड, spaced
hyphens). Hand-maintained: add a variant here when a document or question misses.
(Schemes come next — mined from the corpus, reviewed by hand; METADATA_LAYER_PLAN §6.)
"""

from __future__ import annotations

import re
import unicodedata

# The 33 districts of Chhattisgarh.
DISTRICTS = {
    "raipur": ("रायपुर", "Raipur", []),
    "bilaspur": ("बिलासपुर", "Bilaspur", []),
    "durg": ("दुर्ग", "Durg", []),
    "rajnandgaon": ("राजनांदगांव", "Rajnandgaon", ["राजनांदगाँव", "राजनादगांव", "राजनंदगांव"]),
    "korba": ("कोरबा", "Korba", []),
    "raigarh": ("रायगढ़", "Raigarh", ["रायगढ"]),
    "janjgir_champa": ("जांजगीर-चांपा", "Janjgir-Champa", ["जांजगीर चांपा", "जांजगीर", "जाँजगीर-चाँपा", "Janjgir"]),
    "jashpur": ("जशपुर", "Jashpur", []),
    "surguja": ("सरगुजा", "Surguja", ["Sarguja"]),
    "koriya": ("कोरिया", "Koriya", ["Korea"]),
    "surajpur": ("सूरजपुर", "Surajpur", []),
    "balrampur": ("बलरामपुर-रामानुजगंज", "Balrampur-Ramanujganj", ["बलरामपुर", "Balrampur"]),
    "kabirdham": ("कबीरधाम", "Kabirdham", ["कवर्धा", "Kawardha"]),
    "mungeli": ("मुंगेली", "Mungeli", []),
    "bemetara": ("बेमेतरा", "Bemetara", []),
    "balod": ("बालोद", "Balod", []),
    "balodabazar": ("बलौदाबाजार-भाटापारा", "Balodabazar-Bhatapara",
                    ["बलौदाबाजार", "बलौदा बाजार", "बलौदाबाज़ार", "Baloda Bazar", "Balodabazar"]),
    "gariaband": ("गरियाबंद", "Gariaband", []),
    "mahasamund": ("महासमुंद", "Mahasamund", []),
    "dhamtari": ("धमतरी", "Dhamtari", []),
    "kanker": ("उत्तर बस्तर कांकेर", "Kanker", ["कांकेर", "Uttar Bastar Kanker"]),
    "kondagaon": ("कोंडागांव", "Kondagaon", ["कोण्डागांव", "कोंडागाँव"]),
    "narayanpur": ("नारायणपुर", "Narayanpur", []),
    "bastar": ("बस्तर", "Bastar", []),
    "dantewada": ("दक्षिण बस्तर दंतेवाड़ा", "Dantewada", ["दंतेवाड़ा", "दन्तेवाड़ा", "Dakshin Bastar Dantewada"]),
    "sukma": ("सुकमा", "Sukma", []),
    "bijapur": ("बीजापुर", "Bijapur", []),
    "gpm": ("गौरेला-पेंड्रा-मरवाही", "Gaurela-Pendra-Marwahi", ["गौरेला पेंड्रा मरवाही", "GPM"]),
    "mcb": ("मनेंद्रगढ़-चिरमिरी-भरतपुर", "Manendragarh-Chirmiri-Bharatpur",
            ["मनेन्द्रगढ़-चिरमिरी-भरतपुर", "मनेंद्रगढ़", "मनेन्द्रगढ़", "Manendragarh"]),
    "mma": ("मोहला-मानपुर-अंबागढ़ चौकी", "Mohla-Manpur-Ambagarh Chowki",
            ["मोहला-मानपुर-अम्बागढ़ चौकी", "मोहला मानपुर", "Mohla-Manpur"]),
    "sarangarh": ("सारंगढ़-बिलाईगढ़", "Sarangarh-Bilaigarh", ["सारंगढ़", "Sarangarh"]),
    "kcg": ("खैरागढ़-छुईखदान-गंडई", "Khairagarh-Chhuikhadan-Gandai", ["खैरागढ़", "Khairagarh"]),
    "sakti": ("सक्ती", "Sakti", []),
}

# Posts / designations used in state orders.
POSTS = {
    "chief_secretary": ("मुख्य सचिव", "Chief Secretary", []),
    "acs": ("अपर मुख्य सचिव", "Additional Chief Secretary", []),
    "principal_secretary": ("प्रमुख सचिव", "Principal Secretary", []),
    "special_secretary": ("विशेष सचिव", "Special Secretary", []),
    "joint_secretary": ("संयुक्त सचिव", "Joint Secretary", []),
    "deputy_secretary": ("उप सचिव", "Deputy Secretary", ["उपसचिव"]),
    "under_secretary": ("अवर सचिव", "Under Secretary", []),
    "private_secretary": ("निज सचिव", "Private Secretary", []),
    "secretary": ("सचिव", "Secretary", []),
    "section_officer": ("अनुभाग अधिकारी", "Section Officer", []),
    "aso": ("सहायक अनुभाग अधिकारी", "Assistant Section Officer", []),
    "assistant_grade": ("सहायक ग्रेड", "Assistant Grade", ["सहायक ग्रेड-1", "सहायक ग्रेड-2", "सहायक ग्रेड-3"]),
    "stenographer": ("शीघ्रलेखक", "Stenographer", ["स्टेनोग्राफर", "स्टेनो टायपिस्ट"]),
    "personal_assistant": ("निज सहायक", "Personal Assistant", []),
    "divisional_commissioner": ("संभागायुक्त", "Divisional Commissioner", ["संभाग आयुक्त", "कमिश्नर"]),
    "collector": ("कलेक्टर", "Collector", ["जिलाध्यक्ष"]),
    "additional_collector": ("अपर कलेक्टर", "Additional Collector", []),
    "deputy_collector": ("डिप्टी कलेक्टर", "Deputy Collector", ["उप जिलाध्यक्ष", "संयुक्त कलेक्टर"]),
    "sdm": ("अनुविभागीय अधिकारी", "Sub Divisional Officer", ["एसडीएम", "SDM"]),
    "tehsildar": ("तहसीलदार", "Tehsildar", []),
    "naib_tehsildar": ("नायब तहसीलदार", "Naib Tehsildar", []),
    "ceo": ("मुख्य कार्यपालन अधिकारी", "Chief Executive Officer", []),
    "sp": ("पुलिस अधीक्षक", "Superintendent of Police", []),
    "director": ("संचालक", "Director", []),
    "additional_director": ("अपर संचालक", "Additional Director", []),
    "joint_director": ("संयुक्त संचालक", "Joint Director", []),
    "deputy_director": ("उप संचालक", "Deputy Director", []),
    "assistant_director": ("सहायक संचालक", "Assistant Director", []),
    "accounts_officer": ("लेखा अधिकारी", "Accounts Officer", []),
    "treasury_officer": ("कोषालय अधिकारी", "Treasury Officer", ["जिला कोषालय अधिकारी"]),
    "accountant_general": ("महालेखाकार", "Accountant General", []),
}

# Departments of the Government of Chhattisgarh (Hindi, English, other forms).
DEPARTMENTS = {
    "gad": ("सामान्य प्रशासन विभाग", "General Administration Department", ["GAD", "सा.प्र.वि."]),
    "finance": ("वित्त विभाग", "Finance Department", []),
    "home": ("गृह विभाग", "Home Department", []),
    "revenue": ("राजस्व एवं आपदा प्रबंधन विभाग", "Revenue and Disaster Management Department", ["राजस्व विभाग"]),
    "school_education": ("स्कूल शिक्षा विभाग", "School Education Department", []),
    "higher_education": ("उच्च शिक्षा विभाग", "Higher Education Department", []),
    "technical_education": ("तकनीकी शिक्षा, रोजगार एवं प्रशिक्षण विभाग", "Technical Education, Employment and Training Department",
                            ["तकनीकी शिक्षा विभाग"]),
    "health": ("लोक स्वास्थ्य एवं परिवार कल्याण विभाग", "Public Health and Family Welfare Department",
               ["स्वास्थ्य एवं परिवार कल्याण विभाग", "Health Department"]),
    "medical_education": ("चिकित्सा शिक्षा विभाग", "Medical Education Department", []),
    "agriculture": ("कृषि विकास एवं किसान कल्याण तथा जैव प्रौद्योगिकी विभाग", "Agriculture Department",
                    ["कृषि विभाग", "Agriculture Development and Farmer Welfare"]),
    "panchayat": ("पंचायत एवं ग्रामीण विकास विभाग", "Panchayat and Rural Development Department", []),
    "urban": ("नगरीय प्रशासन एवं विकास विभाग", "Urban Administration and Development Department",
              ["नगरीय प्रशासन विभाग"]),
    "pwd": ("लोक निर्माण विभाग", "Public Works Department", ["PWD"]),
    "water_resources": ("जल संसाधन विभाग", "Water Resources Department", []),
    "phe": ("लोक स्वास्थ्य यांत्रिकी विभाग", "Public Health Engineering Department", ["PHE"]),
    "energy": ("ऊर्जा विभाग", "Energy Department", []),
    "mineral": ("खनिज साधन विभाग", "Mineral Resources Department", []),
    "forest": ("वन एवं जलवायु परिवर्तन विभाग", "Forest and Climate Change Department", ["वन विभाग", "Forest Department"]),
    "tribal": ("आदिम जाति तथा अनुसूचित जाति विकास विभाग", "Tribal and Scheduled Caste Development Department",
               ["आदिम जाति विकास विभाग", "आदिम जाति कल्याण विभाग"]),
    "wcd": ("महिला एवं बाल विकास विभाग", "Women and Child Development Department", []),
    "social_welfare": ("समाज कल्याण विभाग", "Social Welfare Department", []),
    "food": ("खाद्य, नागरिक आपूर्ति एवं उपभोक्ता संरक्षण विभाग", "Food, Civil Supplies and Consumer Protection Department",
             ["खाद्य विभाग"]),
    "cooperation": ("सहकारिता विभाग", "Cooperation Department", []),
    "industry": ("वाणिज्य एवं उद्योग विभाग", "Commerce and Industry Department", []),
    "commercial_tax": ("वाणिज्यिक कर विभाग", "Commercial Tax Department", []),
    "excise": ("वाणिज्यिक कर (आबकारी) विभाग", "Excise Department", ["आबकारी विभाग"]),
    "transport": ("परिवहन विभाग", "Transport Department", []),
    "housing": ("आवास एवं पर्यावरण विभाग", "Housing and Environment Department", []),
    "labour": ("श्रम विभाग", "Labour Department", []),
    "law": ("विधि एवं विधायी कार्य विभाग", "Law and Legislative Affairs Department", ["विधि विभाग"]),
    "parliamentary": ("संसदीय कार्य विभाग", "Parliamentary Affairs Department", []),
    "eit": ("इलेक्ट्रॉनिकी एवं सूचना प्रौद्योगिकी विभाग", "Electronics and Information Technology Department",
            ["इलेक्ट्रानिकी एवं सूचना प्रौद्योगिकी विभाग", "E&IT", "EIT", "IT Department"]),
    "public_relations": ("जनसंपर्क विभाग", "Public Relations Department", []),
    "tourism": ("पर्यटन विभाग", "Tourism Department", []),
    "culture": ("संस्कृति विभाग", "Culture Department", ["संस्कृति एवं पुरातत्व विभाग"]),
    "sports": ("खेल एवं युवा कल्याण विभाग", "Sports and Youth Welfare Department", []),
    "religious_trusts": ("धार्मिक न्यास एवं धर्मस्व विभाग", "Religious Trusts and Endowments Department", []),
    "planning": ("योजना, आर्थिक एवं सांख्यिकी विभाग", "Planning, Economics and Statistics Department", []),
    "village_industry": ("ग्रामोद्योग विभाग", "Village Industries Department", []),
    "animal_husbandry": ("पशुधन विकास विभाग", "Animal Husbandry Department", []),
    "fisheries": ("मछली पालन विभाग", "Fisheries Department", []),
    "aviation": ("विमानन विभाग", "Aviation Department", []),
    "backward_classes": ("पिछड़ा वर्ग एवं अल्पसंख्यक विकास विभाग", "Backward Classes and Minority Development Department", []),
    "science_tech": ("विज्ञान एवं प्रौद्योगिकी विभाग", "Science and Technology Department", []),
}

_DEV = "ऀ-ॿ"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    # whitespace first, then the dash: same result as `\s*[-–—]\s*` -> "-", but that
    # pattern re-scanned every run of blank lines from each position (quadratic —
    # 36 s on a page of 60,000 blank lines)
    s = re.sub(r"\s+", " ", s)
    return re.sub(r" ?[-–—] ?", "-", s)          # "जांजगीर - चांपा" -> "जांजगीर-चांपा"


def _pattern(variant: str) -> str:
    v = re.escape(_norm(variant)).replace(r"\-", r"\s*[-–—]?\s*").replace(r"\ ", r"\s+")
    if re.search(f"[{_DEV}]", variant):
        # Devanagari has no \b: a name must not continue into more letters/matras
        # ("दुर्ग" must not match "दुर्गा").
        return rf"(?<![{_DEV}]){v}(?![{_DEV}])"
    return rf"(?<![A-Za-z]){v}(?![A-Za-z])"


def _compile(table: dict):
    pats = []
    for key, (hi, en, other) in table.items():
        for v in {hi, en, *other}:
            pats.append((len(v), key, re.compile(_pattern(v), re.I if not re.search(f"[{_DEV}]", v) else 0)))
    pats.sort(key=lambda t: -t[0])               # longest first: "अवर सचिव" before "सचिव"
    return pats


_COMPILED = {"district": _compile(DISTRICTS), "post": _compile(POSTS), "department": _compile(DEPARTMENTS)}
TABLES = {"district": DISTRICTS, "post": POSTS, "department": DEPARTMENTS}


def find(text: str, kind: str) -> dict:
    """{canonical id: mention count} of `kind` ('district' | 'post' | 'department')
    in `text`. Longer names win: a span matched by "उत्तर बस्तर कांकेर" is not also
    counted as "बस्तर"."""
    t = _norm(text)
    taken = [False] * (len(t) + 1)
    counts: dict = {}
    for _, key, rx in _COMPILED[kind]:
        for m in rx.finditer(t):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            counts[key] = counts.get(key, 0) + 1
    return counts


def hindi(kind: str, key: str) -> str:
    return TABLES[kind][key][0]


def english(kind: str, key: str) -> str:
    return TABLES[kind][key][1]
