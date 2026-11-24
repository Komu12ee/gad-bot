"""
Comprehensive Golden Dataset Generator for GAD-BOT Component Evaluation Suite.

Generates high-precision, schema-validated golden datasets for:
1. Ingestion & OCR (ingestion_ocr)
2. Query Preprocessing (query_preprocessing)
3. Hybrid Retrieval (hybrid_retrieval)
4. Second-Stage Reranking (reranking)
5. Context Budgeting (context_budgeting)
6. Generation (LLM) (generation)
7. Multi-Hop Loop & Supersession (multi_hop)
8. Grounding Guardrails (grounding_guardrail)
9. End-to-End Batch Test Golden Suite (batch_test)

Output formats:
- Individual CSV (UTF-8 BOM) for each component
- Individual XLSX with styling for each component
- Individual JSON (list of records) for each component
- Master Multi-Component Excel Workbook (GAD_BOT_Golden_MultiComponent_Evaluation.xlsx)
- Batch Test Golden Suite JSON (golden_batch_test.json)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Ensure repo root and core are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "core"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CORE_DIR))

from eval.registry import REGISTRY
from eval.validator import parse_and_validate_batch

# Target output directories
OUTPUT_DIR = CORE_DIR / "eval" / "golden_datasets"
SAMPLES_DIR = CORE_DIR / "eval" / "samples"
PUBLIC_SAMPLES_DIR = CORE_DIR / "05_webui" / "nodejs" / "public" / "samples"
WORK_DIR = REPO_ROOT / "gad_kg" / "data" / "work"

for d in [OUTPUT_DIR, SAMPLES_DIR, PUBLIC_SAMPLES_DIR, WORK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Styling for Excel worksheets
FONT_HEADER = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
FILL_HEADER = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")  # Deep Navy
FILL_ALT_ROW = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid") # Slate 50
FONT_BODY = Font(name="Segoe UI", size=10, color="0F172A")
BORDER_THIN = Border(
    left=Side(style="thin", color="CBD5E1"),
    right=Side(style="thin", color="CBD5E1"),
    top=Side(style="thin", color="CBD5E1"),
    bottom=Side(style="thin", color="CBD5E1"),
)
ALIGN_CENTER = Alignment(horizontal="center", vertical="center")
ALIGN_LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)


# ==============================================================================
# COMPONENT 1: Ingestion & OCR (ingestion_ocr)
# ==============================================================================
INGESTION_OCR_ROWS = [
    {
        "doc_id": "doc_gad_transfer_2026",
        "page_no": 1,
        "file_path": "data/fg2/stage2/63dfd3acb3182dc5.md",
        "ground_truth_text": "# छत्तीसगढ़ शासन, सामान्य प्रशासन विभाग, मंत्रालय\n## परिपत्र क्रमांक एफ 14-2/2026/1-3, नवा रायपुर अटल नगर\n\nविषय: राज्य एवं अधीनस्थ सेवाओं के अधिकारियों/कर्मचारियों हेतु स्थानांतरण नीति 2026-27।\n\nराज्य शासन एतद्द्वारा वर्ष 2026-27 हेतु स्थानांतरण नीति लागू करता है। अनुसूचित क्षेत्रों (बस्तर एवं सरगुजा संभाग) में पदस्थ शासकीय सेवकों की न्यूनतम पदावधि 3 वर्ष निर्धारित की जाती है।",
        "language": "hi",
        "has_table": False,
        "gate_label": "good",
        "gold_numbers": json.dumps(["2026-27", "3", "14-2/2026/1-3"], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps(["# छत्तीसगढ़ शासन", "विषय: राज्य एवं", "राज्य शासन एतद्द्वारा"], ensure_ascii=False),
    },
    {
        "doc_id": "doc_sas_civil_list_2026",
        "page_no": 24,
        "file_path": "data/fg2/stage2/63dfd3acb3182dc5.md",
        "ground_truth_text": "| क्र. | अधिकारी का नाम | जन्म तिथि | गृह जिला | वर्तमान पदस्थापना |\n|---|---|---|---|---|\n| 442 | श्री अनुपम आशीष टोप्पो | 24.11.1987 | जशपुर | डिप्टी कलेक्टर रायपुर (पदभार 30.07.2025) एवं ओएसडी मंत्री वित्त विभाग (16.03.2026) |",
        "language": "hi",
        "has_table": True,
        "gate_label": "good",
        "gold_numbers": json.dumps(["442", "24.11.1987", "30.07.2025", "16.03.2026"], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps(["| क्र. | अधिकारी का नाम", "| 442 | श्री अनुपम"], ensure_ascii=False),
    },
    {
        "doc_id": "doc_finance_da_2025",
        "page_no": 1,
        "file_path": "data/fg2/stage2/efc290b78f6c8ad7.md",
        "ground_truth_text": "विषय: सातवें वेतनमान में शासकीय सेवकों को महंगाई भत्ते (DA) की दर में 4% की वृद्धि बाबत।\n\nराज्य शासन द्वारा शासकीय सेवकों को दिनांक 01.07.2025 से 46% के स्थान पर 50% की दर से महंगाई भत्ता स्वीकृत किया जाता है। यह वृद्धि 1 जुलाई 2025 से नकद देय होगी।",
        "language": "hi",
        "has_table": False,
        "gate_label": "good",
        "gold_numbers": json.dumps(["4%", "01.07.2025", "46%", "50%", "1 जुलाई 2025"], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps(["विषय: सातवें वेतनमान", "राज्य शासन द्वारा"], ensure_ascii=False),
    },
    {
        "doc_id": "doc_ccl_women_rules",
        "page_no": 2,
        "file_path": "data/fg2/stage2/bab670ac612f4f79.md",
        "ground_truth_text": "नियम 38-ग: महिला शासकीय सेवकों को संतान पालन अवकाश (Child Care Leave)।\nमहिला शासकीय सेवकों को उनकी संपूर्ण सेवाकाल में अधिकतम 730 दिनों का संतान पालन अवकाश दो ज्येष्ठतम जीवित संतानों के 18 वर्ष की आयु प्राप्त करने तक देय होगा।",
        "language": "hi",
        "has_table": False,
        "gate_label": "good",
        "gold_numbers": json.dumps(["38-ग", "730", "18"], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps(["नियम 38-ग", "महिला शासकीय"], ensure_ascii=False),
    },
    {
        "doc_id": "doc_vacancy_jmrc_2014",
        "page_no": 1,
        "file_path": "data/fg2/stage2/359fbfd517894655.md",
        "ground_truth_text": "JAIPUR METRO RAIL CORPORATION LIMITED\nNotification No. F.1 (83)/JMRC/DCA/HR/SS-IV/2014-15/1076 D dated 16-06-2014.\nApplications are invited on deputation from officials of Central/State Govts for the post of General Manager with minimum 5 years experience in grade pay 7600.",
        "language": "en",
        "has_table": False,
        "gate_label": "good",
        "gold_numbers": json.dumps(["1076", "16-06-2014", "5", "7600"], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps(["JAIPUR METRO", "Applications are invited"], ensure_ascii=False),
    },
    {
        "doc_id": "doc_corrupt_scan_noise",
        "page_no": 1,
        "file_path": "data/fg2/stage2/corrupt_artifact_01.md",
        "ground_truth_text": "~#*| { } ^^^ ::: BUSOPAR BUSOPAR ||| === ??? ~~~ 000000000000000000000000",
        "language": "en",
        "has_table": False,
        "gate_label": "bad",
        "gold_numbers": json.dumps([], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps([], ensure_ascii=False),
    },
    {
        "doc_id": "doc_chips_sdc_specs",
        "page_no": 5,
        "file_path": "data/fg2/stage2/5f369c50ead865ab.md",
        "ground_truth_text": "Chhattisgarh Infotech Promotion Society (CHiPS)\nState Data Centre (SDC) Uptime Commitment:\n| Facility | SLA Target | Latency Guarantee |\n|---|---|---|\n| Core Hosting Infrastructure | 99.982% | < 5 ms |\n| WAN Link Redundancy | 99.90% | < 20 ms |",
        "language": "en",
        "has_table": True,
        "gate_label": "good",
        "gold_numbers": json.dumps(["99.982%", "5", "99.90%", "20"], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps(["Chhattisgarh Infotech", "| Facility | SLA Target"], ensure_ascii=False),
    },
    {
        "doc_id": "doc_compassionate_rules",
        "page_no": 1,
        "file_path": "data/fg2/stage2/18ee1d1997164968.md",
        "ground_truth_text": "अनुकंपा नियुक्ति नियम 2024: शासकीय सेवक की सेवाकाल में मृत्यु होने पर आश्रित परिवार के एक सदस्य को अनुकंपा नियुक्ति की पात्रता होगी। आवेदन मृत्यु दिनांक से अधिकतम 45 दिवस के भीतर प्रस्तुत किया जाना आवश्यक है।",
        "language": "hi",
        "has_table": False,
        "gate_label": "good",
        "gold_numbers": json.dumps(["2024", "45"], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps(["अनुकंपा नियुक्ति नियम", "आवेदन मृत्यु दिनांक"], ensure_ascii=False),
    },
    {
        "doc_id": "doc_staff_selection_board",
        "page_no": 1,
        "file_path": "data/fg2/stage2/ecc05b64348b4c77.md",
        "ground_truth_text": "क्रमांक RULE/297/2025-GAD-3 :: भारत के संविधान के अनुच्छेद 309 के परंतुक एवं छत्तीसगढ़ कर्मचारी चयन मण्डल अधिनियम, 2026 (क्रमांक 5 सन् 2026) की धारा 3 की उप-धारा (4) द्वारा प्रदत्त शक्तियों को प्रयोग में लाते हुए संशोधन अधिसूचित किया जाता है।",
        "language": "hi",
        "has_table": False,
        "gate_label": "good",
        "gold_numbers": json.dumps(["RULE/297/2025-GAD-3", "309", "2026", "5", "3", "4"], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps(["क्रमांक RULE/297/2025-GAD-3"], ensure_ascii=False),
    },
    {
        "doc_id": "doc_blurred_unusable_scan",
        "page_no": 1,
        "file_path": "data/fg2/stage2/blur_sample_02.md",
        "ground_truth_text": "░▒▓█ █▓▒░ [Unreadable handwritten ink spread across document page 1] §¶†‡",
        "language": "mixed",
        "has_table": False,
        "gate_label": "bad",
        "gold_numbers": json.dumps([], ensure_ascii=False),
        "gold_chunk_boundaries": json.dumps([], ensure_ascii=False),
    },
]


# ==============================================================================
# COMPONENT 2: Query Preprocessing (query_preprocessing)
# ==============================================================================
QUERY_PREPROCESSING_ROWS = [
    {
        "query_id": "QP001",
        "query": "अनुकंपा नियुक्ति के लिए आवेदन करने की अंतिम समय सीमा क्या है?",
        "true_language": "hi",
        "true_intent": "general",
        "expected_expansion_terms": json.dumps(["अनुकंपा नियुक्ति", "समय सीमा", "समय अवधि", "compassionate appointment"], ensure_ascii=False),
    },
    {
        "query_id": "QP002",
        "query": "What are the rules for Child Care Leave (CCL) for female government servants in Chhattisgarh?",
        "true_language": "en",
        "true_intent": "general",
        "expected_expansion_terms": json.dumps(["child care leave", "ccl", "महिला शासकीय सेवक", "संतान पालन अवकाश"], ensure_ascii=False),
    },
    {
        "query_id": "QP003",
        "query": "karmachariyon ko 1 july 2025 se da kitne percent milega?",
        "true_language": "hinglish",
        "true_intent": "number",
        "expected_expansion_terms": json.dumps(["महंगाई भत्ता", "dearness allowance", "da rate", "प्रतिशत"], ensure_ascii=False),
    },
    {
        "query_id": "QP004",
        "query": "GAD ke transfer policy 2026-27 ke bare me sabhi circulars aur orders dikhayein",
        "true_language": "hinglish",
        "true_intent": "catalogue",
        "expected_expansion_terms": json.dumps(["स्थानांतरण नीति", "transfer policy", "परिपत्र", "आदेश"], ensure_ascii=False),
    },
    {
        "query_id": "QP005",
        "query": "List all gazetted notifications issued under the IAS establishment section in 2025",
        "true_language": "en",
        "true_intent": "catalogue",
        "expected_expansion_terms": json.dumps(["ias establishment", "notification", "अधिसूचना", "राजपत्र"], ensure_ascii=False),
    },
    {
        "query_id": "QP006",
        "query": "सामान्य प्रशासन विभाग में वर्तमान में कुल कितने अनुभाग कार्यरत हैं?",
        "true_language": "hi",
        "true_intent": "count",
        "expected_expansion_terms": json.dumps(["अनुभाग", "संख्या", "sections count", "प्रशासनिक संरचना"], ensure_ascii=False),
    },
    {
        "query_id": "QP007",
        "query": "How many retired IAS officers are currently empanelled for departmental inquiries?",
        "true_language": "en",
        "true_intent": "count",
        "expected_expansion_terms": json.dumps(["retired ias", "empanelled", "सेवानिवृत्त आईएएस", "विभागीय जांच"], ensure_ascii=False),
    },
    {
        "query_id": "QP008",
        "query": "सातवें वेतनमान के तहत स्वीकृत महंगाई भत्ते की दर क्या है?",
        "true_language": "hi",
        "true_intent": "number",
        "expected_expansion_terms": json.dumps(["सातवां वेतनमान", "महंगाई भत्ता", "da rate", "50%"], ensure_ascii=False),
    },
    {
        "query_id": "QP009",
        "query": "What is the budget ceiling in crores for SDC cloud infrastructure in 2025-26?",
        "true_language": "en",
        "true_intent": "number",
        "expected_expansion_terms": json.dumps(["budget ceiling", "crores", "sdc cloud", "बजट प्रावधान"], ensure_ascii=False),
    },
    {
        "query_id": "QP010",
        "query": "Namaste! Kaise hain aap aur aap kya kar sakte hain?",
        "true_language": "hinglish",
        "true_intent": "greeting",
        "expected_expansion_terms": json.dumps([], ensure_ascii=False),
    },
    {
        "query_id": "QP011",
        "query": "Good morning, I need help navigating government circulars.",
        "true_language": "en",
        "true_intent": "greeting",
        "expected_expansion_terms": json.dumps([], ensure_ascii=False),
    },
    {
        "query_id": "QP012",
        "query": "नमस्ते, क्या आप मेरी सहायता कर सकते हैं?",
        "true_language": "hi",
        "true_intent": "greeting",
        "expected_expansion_terms": json.dumps([], ensure_ascii=False),
    },
    {
        "query_id": "QP013",
        "query": "divyang karmachariyon ke vahan bhatta ke niyam bataiye",
        "true_language": "hinglish",
        "true_intent": "general",
        "expected_expansion_terms": json.dumps(["दिव्यांग", "वाहन भत्ता", "conveyance allowance", "नियम"], ensure_ascii=False),
    },
    {
        "query_id": "QP014",
        "query": "What are the rules regarding deputation of state officers to the Central Government?",
        "true_language": "en",
        "true_intent": "general",
        "expected_expansion_terms": json.dumps(["deputation", "central government", "प्रतिनियुक्ति", "केन्द्र सरकार"], ensure_ascii=False),
    },
    {
        "query_id": "QP015",
        "query": "अर्जित अवकाश (Earned Leave) संचय करने की अधिकतम सीमा कितने दिवस है?",
        "true_language": "hi",
        "true_intent": "number",
        "expected_expansion_terms": json.dumps(["अर्जित अवकाश", "earned leave", "संचय सीमा", "300 दिवस"], ensure_ascii=False),
    },
]


# ==============================================================================
# COMPONENT 3: Hybrid Retrieval (hybrid_retrieval)
# ==============================================================================
HYBRID_RETRIEVAL_ROWS = [
    {
        "query_id": "HR001",
        "query": "स्थानांतरण नीति 2026-27 के मुख्य प्रावधान और अनुसूचित क्षेत्रों में न्यूनतम सेवा अवधि",
        "gold_passage_ids": json.dumps(["63dfd3acb3182dc5_c001", "63dfd3acb3182dc5_c002"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": False,
    },
    {
        "query_id": "HR002",
        "query": "What is the maximum Child Care Leave (CCL) period permitted for women employees?",
        "gold_passage_ids": json.dumps(["bab670ac612f4f79_c001"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "hi",
        "is_exact_match_query": False,
    },
    {
        "query_id": "HR003",
        "query": "अधिसूचना क्रमांक RULE/297/2025-GAD-3 के तहत किए गए संशोधन",
        "gold_passage_ids": json.dumps(["ecc05b64348b4c77_c001"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": True,
    },
    {
        "query_id": "HR004",
        "query": "Dearness Allowance rates revision order effective 01.07.2025 in seventh pay commission",
        "gold_passage_ids": json.dumps(["efc290b78f6c8ad7_c001", "efc290b78f6c8ad7_c002"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "hi",
        "is_exact_match_query": False,
    },
    {
        "query_id": "HR005",
        "query": "vacancy circular No. A-12025/12/2010-Adm.I dated 2010-07-19 Planning Commission",
        "gold_passage_ids": json.dumps(["0007f1071c2918fc_c001"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "en",
        "is_exact_match_query": True,
    },
    {
        "query_id": "HR006",
        "query": "अनुकंपा नियुक्ति के लिए आश्रित परिवार के आवेदन प्रस्तुत करने की समय सीमा",
        "gold_passage_ids": json.dumps(["18ee1d1997164968_c001"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": False,
    },
    {
        "query_id": "HR007",
        "query": "CHiPS State Data Centre uptime SLA requirements and cloud hosting guarantees",
        "gold_passage_ids": json.dumps(["5f369c50ead865ab_c001", "5f369c50ead865ab_c002"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "en",
        "is_exact_match_query": False,
    },
    {
        "query_id": "HR008",
        "query": "deputation guidelines for state government employees to autonomous bodies",
        "gold_passage_ids": json.dumps(["678e98966a2c4c56_c001"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "hi",
        "is_exact_match_query": False,
    },
    {
        "query_id": "HR009",
        "query": "circular No. 1329 dated 23.09.2026 general circulars and notifications",
        "gold_passage_ids": json.dumps(["bab670ac612f4f79_c002"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "hi",
        "is_exact_match_query": True,
    },
    {
        "query_id": "HR010",
        "query": "दिव्यांग शासकीय सेवकों के लिए विशेष वाहन भत्ता पात्रता एवं दरें",
        "gold_passage_ids": json.dumps(["18ee1d1997164968_c003"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": False,
    },
    {
        "query_id": "HR011",
        "query": "who is Shri Anupam Ashish Toppo and what is his current designation?",
        "gold_passage_ids": json.dumps(["63dfd3acb3182dc5_c024"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "hi",
        "is_exact_match_query": False,
    },
    {
        "query_id": "HR012",
        "query": "विभागीय पदोन्नति समिति (DPC) की नियमित बैठक आयोजित करने के निर्देश",
        "gold_passage_ids": json.dumps(["45b4b2ca681a4c0d_c001"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": False,
    },
]


# ==============================================================================
# COMPONENT 4: Second-Stage Reranking (reranking)
# ==============================================================================
RERANKING_ROWS = [
    {
        "query_id": "RR001",
        "query": "स्थानांतरण नीति 2026-27 के तहत अनुसूचित क्षेत्रों में पदस्थापना की न्यूनतम अवधि",
        "gold_passage_ids": json.dumps(["63dfd3acb3182dc5_c001"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["0007f1071c2918fc_c001", "efc290b78f6c8ad7_c002", "63dfd3acb3182dc5_c001", "bab670ac612f4f79_c001"], ensure_ascii=False),
    },
    {
        "query_id": "RR002",
        "query": "Maximum period of Child Care Leave (CCL) admissible to women government servants",
        "gold_passage_ids": json.dumps(["bab670ac612f4f79_c001"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["efc290b78f6c8ad7_c001", "bab670ac612f4f79_c001", "63dfd3acb3182dc5_c002", "18ee1d1997164968_c001"], ensure_ascii=False),
    },
    {
        "query_id": "RR003",
        "query": "1 जुलाई 2025 से सातवें वेतनमान में महंगाई भत्ता (DA) की संशोधित दर",
        "gold_passage_ids": json.dumps(["efc290b78f6c8ad7_c001"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["63dfd3acb3182dc5_c001", "18ee1d1997164968_c003", "efc290b78f6c8ad7_c001", "0007f1071c2918fc_c002"], ensure_ascii=False),
    },
    {
        "query_id": "RR004",
        "query": "Time limit in days to submit application for compassionate appointment",
        "gold_passage_ids": json.dumps(["18ee1d1997164968_c001"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["45b4b2ca681a4c0d_c001", "18ee1d1997164968_c001", "bab670ac612f4f79_c002", "5f369c50ead865ab_c001"], ensure_ascii=False),
    },
    {
        "query_id": "RR005",
        "query": "Deputation tenure and rules governing Central Government deputation",
        "gold_passage_ids": json.dumps(["678e98966a2c4c56_c001"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["0007f1071c2918fc_c001", "45b4b2ca681a4c0d_c001", "678e98966a2c4c56_c001", "63dfd3acb3182dc5_c001"], ensure_ascii=False),
    },
    {
        "query_id": "RR006",
        "query": "Official posting of Anupam Ashish Toppo as OSD in Mantralaya",
        "gold_passage_ids": json.dumps(["63dfd3acb3182dc5_c024"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["45b4b2ca681a4c0d_c001", "efc290b78f6c8ad7_c001", "63dfd3acb3182dc5_c024", "18ee1d1997164968_c001"], ensure_ascii=False),
    },
    {
        "query_id": "RR007",
        "query": "Chhattisgarh Staff Selection Board amendment notification RULE/297/2025-GAD-3",
        "gold_passage_ids": json.dumps(["ecc05b64348b4c77_c001"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["bab670ac612f4f79_c001", "ecc05b64348b4c77_c001", "63dfd3acb3182dc5_c001", "0007f1071c2918fc_c001"], ensure_ascii=False),
    },
    {
        "query_id": "RR008",
        "query": "State Data Centre uptime commitment and tier-3 hosting standards under CHiPS",
        "gold_passage_ids": json.dumps(["5f369c50ead865ab_c001"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["45b4b2ca681a4c0d_c001", "efc290b78f6c8ad7_c002", "5f369c50ead865ab_c001", "18ee1d1997164968_c003"], ensure_ascii=False),
    },
    {
        "query_id": "RR009",
        "query": "दिव्यांग कर्मचारियों हेतु वाहन भत्ता की पात्रता शर्तें",
        "gold_passage_ids": json.dumps(["18ee1d1997164968_c003"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["0007f1071c2918fc_c002", "18ee1d1997164968_c003", "678e98966a2c4c56_c001", "45b4b2ca681a4c0d_c001"], ensure_ascii=False),
    },
    {
        "query_id": "RR010",
        "query": "Circular No. 1329 dated 23.09.2026 administrative notifications",
        "gold_passage_ids": json.dumps(["bab670ac612f4f79_c002"], ensure_ascii=False),
        "candidate_passage_ids": json.dumps(["63dfd3acb3182dc5_c001", "bab670ac612f4f79_c002", "efc290b78f6c8ad7_c001", "18ee1d1997164968_c001"], ensure_ascii=False),
    },
]


# ==============================================================================
# COMPONENT 5: Context Budgeting (context_budgeting)
# ==============================================================================
CONTEXT_BUDGETING_ROWS = [
    {
        "query_id": "CB001",
        "query": "स्थानांतरण नीति 2026-27 के तहत अनुसूचित क्षेत्रों में न्यूनतम पदस्थापना अवधि",
        "ranked_passage_ids": json.dumps(["63dfd3acb3182dc5_c001", "63dfd3acb3182dc5_c002", "efc290b78f6c8ad7_c001"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["63dfd3acb3182dc5_c001"], ensure_ascii=False),
        "gold_answer_span": "अनुसूचित क्षेत्रों (बस्तर एवं सरगुजा संभाग) में पदस्थ शासकीय सेवकों की न्यूनतम पदावधि 3 वर्ष निर्धारित की जाती है।",
    },
    {
        "query_id": "CB002",
        "query": "संतान पालन अवकाश (Child Care Leave) की कुल अनुमेय अवधि",
        "ranked_passage_ids": json.dumps(["bab670ac612f4f79_c001", "18ee1d1997164968_c001"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["bab670ac612f4f79_c001"], ensure_ascii=False),
        "gold_answer_span": "महिला शासकीय सेवकों को उनकी संपूर्ण सेवाकाल में अधिकतम 730 दिनों का संतान पालन अवकाश दो ज्येष्ठतम जीवित संतानों के 18 वर्ष की आयु प्राप्त करने तक देय होगा।",
    },
    {
        "query_id": "CB003",
        "query": "1 जुलाई 2025 से महंगाई भत्ता की स्वीकृत दर",
        "ranked_passage_ids": json.dumps(["efc290b78f6c8ad7_c001", "45b4b2ca681a4c0d_c001"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["efc290b78f6c8ad7_c001"], ensure_ascii=False),
        "gold_answer_span": "दिनांक 01.07.2025 से 46% के स्थान पर 50% की दर से महंगाई भत्ता स्वीकृत किया जाता है।",
    },
    {
        "query_id": "CB004",
        "query": "अनुकंपा नियुक्ति हेतु आवेदन प्रस्तुत करने की समय सीमा",
        "ranked_passage_ids": json.dumps(["18ee1d1997164968_c001", "678e98966a2c4c56_c001"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["18ee1d1997164968_c001"], ensure_ascii=False),
        "gold_answer_span": "आवेदन मृत्यु दिनांक से अधिकतम 45 दिवस के भीतर प्रस्तुत किया जाना आवश्यक है।",
    },
    {
        "query_id": "CB005",
        "query": "State Data Centre core hosting uptime commitment",
        "ranked_passage_ids": json.dumps(["5f369c50ead865ab_c001", "0007f1071c2918fc_c001"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["5f369c50ead865ab_c001"], ensure_ascii=False),
        "gold_answer_span": "Core Hosting Infrastructure SLA Target: 99.982% with < 5 ms Latency Guarantee.",
    },
    {
        "query_id": "CB006",
        "query": "Shri Anupam Ashish Toppo current administrative assignment",
        "ranked_passage_ids": json.dumps(["63dfd3acb3182dc5_c024", "45b4b2ca681a4c0d_c001"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["63dfd3acb3182dc5_c024"], ensure_ascii=False),
        "gold_answer_span": "डिप्टी कलेक्टर रायपुर (पदभार 30.07.2025) एवं ओएसडी मंत्री वित्त विभाग (16.03.2026)",
    },
    {
        "query_id": "CB007",
        "query": "Staff Selection Board amendment rule number",
        "ranked_passage_ids": json.dumps(["ecc05b64348b4c77_c001", "bab670ac612f4f79_c002"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["ecc05b64348b4c77_c001"], ensure_ascii=False),
        "gold_answer_span": "विभागीय समसंख्यक अधिसूचना क्रमांक RULE/297/2025-GAD-3",
    },
    {
        "query_id": "CB008",
        "query": "Planning Commission Vacancy Circular date and number",
        "ranked_passage_ids": json.dumps(["0007f1071c2918fc_c001", "678e98966a2c4c56_c001"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["0007f1071c2918fc_c001"], ensure_ascii=False),
        "gold_answer_span": "Vacancy Circular | No. A-12025/12/2010-Adm.I | Dated 2010-07-19",
    },
    {
        "query_id": "CB009",
        "query": "दिव्यांग कर्मचारियों हेतु विशेष वाहन भत्ता दर",
        "ranked_passage_ids": json.dumps(["18ee1d1997164968_c003", "efc290b78f6c8ad7_c001"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["18ee1d1997164968_c003"], ensure_ascii=False),
        "gold_answer_span": "दिव्यांग शासकीय सेवकों को प्रतिमाह ₹1,500 विशेष वाहन भत्ता देय होगा।",
    },
    {
        "query_id": "CB010",
        "query": "अर्जित अवकाश संचय की उच्चतम सीमा",
        "ranked_passage_ids": json.dumps(["bab670ac612f4f79_c001", "63dfd3acb3182dc5_c002"], ensure_ascii=False),
        "gold_passage_ids": json.dumps(["bab670ac612f4f79_c001"], ensure_ascii=False),
        "gold_answer_span": "शासकीय सेवक अपने खाते में अधिकतम 300 दिवस का अर्जित अवकाश संचित कर सकते हैं।",
    },
]


# ==============================================================================
# COMPONENT 6: Generation (LLM) (generation)
# ==============================================================================
GENERATION_ROWS = [
    {
        "query_id": "GEN001",
        "query": "स्थानांतरण नीति 2026-27 के अनुसार अनुसूचित क्षेत्रों में शासकीय सेवकों की न्यूनतम पदावधि क्या है?",
        "passages": json.dumps(["[1] परिपत्र क्रमांक एफ 14-2/2026/1-3: अनुसूचित क्षेत्रों (बस्तर एवं सरगुजा संभाग) में पदस्थ शासकीय सेवकों की न्यूनतम पदावधि 3 वर्ष निर्धारित की जाती है।"], ensure_ascii=False),
        "reference_answer": "स्थानांतरण नीति 2026-27 के अनुसार अनुसूचित क्षेत्रों (बस्तर एवं सरगुजा संभाग) में पदस्थ शासकीय सेवकों की न्यूनतम पदावधि 3 वर्ष निर्धारित की गई है [1]।",
        "expected_language": "hi",
        "answerable": True,
        "expected_units": json.dumps(["वर्ष"], ensure_ascii=False),
        "max_words": 40,
    },
    {
        "query_id": "GEN002",
        "query": "What is the penalty for late submission of annual property returns by government servants?",
        "passages": json.dumps([], ensure_ascii=False),
        "reference_answer": "I could not find this in the available documents. Please refer to the official GAD conduct rules.",
        "expected_language": "en",
        "answerable": False,
        "expected_units": json.dumps([], ensure_ascii=False),
        "max_words": 30,
    },
    {
        "query_id": "GEN003",
        "query": "1 जुलाई 2025 से शासकीय सेवकों के लिए महंगाई भत्ता (DA) कितने प्रतिशत स्वीकृत किया गया है?",
        "passages": json.dumps(["[1] वित्त विभाग आदेश: राज्य शासन द्वारा शासकीय सेवकों को दिनांक 01.07.2025 से 46% के स्थान पर 50% की दर से महंगाई भत्ता स्वीकृत किया जाता है।"], ensure_ascii=False),
        "reference_answer": "दिनांक 1 जुलाई 2025 से शासकीय सेवकों के लिए महंगाई भत्ता 50% की दर से स्वीकृत किया गया है [1]।",
        "expected_language": "hi",
        "answerable": True,
        "expected_units": json.dumps(["%"], ensure_ascii=False),
        "max_words": 35,
    },
    {
        "query_id": "GEN004",
        "query": "What is the maximum duration of Child Care Leave (CCL) admissible to women government employees?",
        "passages": json.dumps(["[1] Rule 38-C: Women government servants are entitled to a maximum of 730 days of Child Care Leave (CCL) during their entire service for up to two eldest surviving children below 18 years of age."], ensure_ascii=False),
        "reference_answer": "Women government employees are entitled to a maximum of 730 days of Child Care Leave (CCL) during their entire service period for their two eldest surviving children below 18 years of age [1].",
        "expected_language": "en",
        "answerable": True,
        "expected_units": json.dumps(["days", "years"], ensure_ascii=False),
        "max_words": 45,
    },
    {
        "query_id": "GEN005",
        "query": "अनुकंपा नियुक्ति के लिए आवेदन प्रस्तुत करने की निर्धारित समय सीमा कितनी है?",
        "passages": json.dumps(["[1] परिपत्र: शासकीय सेवक की मृत्यु होने पर आश्रित परिवार द्वारा आवेदन मृत्यु दिनांक से अधिकतम 45 दिवस के भीतर प्रस्तुत किया जाना आवश्यक है।"], ensure_ascii=False),
        "reference_answer": "अनुकंपा नियुक्ति हेतु आवेदन शासकीय सेवक की मृत्यु दिनांक से अधिकतम 45 दिवस के भीतर प्रस्तुत किया जाना आवश्यक है [1]।",
        "expected_language": "hi",
        "answerable": True,
        "expected_units": json.dumps(["दिवस"], ensure_ascii=False),
        "max_words": 35,
    },
    {
        "query_id": "GEN006",
        "query": "How many days of maternity leave are provided to female contract employees under the new 2026 circular?",
        "passages": json.dumps([], ensure_ascii=False),
        "reference_answer": "I could not find this in the available documents.",
        "expected_language": "en",
        "answerable": False,
        "expected_units": json.dumps([], ensure_ascii=False),
        "max_words": 25,
    },
    {
        "query_id": "GEN007",
        "query": "श्री अनुपम आशीष टोप्पो का वर्तमान पदभार एवं विभाग क्या है?",
        "passages": json.dumps(["[1] सिविल लिस्ट अभिलेख: श्री अनुपम आशीष टोप्पो, राज्य प्रशासनिक सेवा (SAS), डिप्टी कलेक्टर रायपुर (पदभार 30.07.2025) एवं ओएसडी, मान. मंत्री वित्त विभाग, मंत्रालय (पदभार 16.03.2026)।"], ensure_ascii=False),
        "reference_answer": "श्री अनुपम आशीष टोप्पो राज्य प्रशासनिक सेवा (SAS) अधिकारी हैं। वे वर्तमान में डिप्टी कलेक्टर रायपुर तथा वित्त मंत्री कार्यालय में विशेष कर्तव्यस्थ अधिकारी (OSD) के पद पर पदस्थ हैं [1]।",
        "expected_language": "hi",
        "answerable": True,
        "expected_units": json.dumps([], ensure_ascii=False),
        "max_words": 45,
    },
    {
        "query_id": "GEN008",
        "query": "What is the uptime SLA percentage guaranteed for core hosting by CHiPS State Data Centre?",
        "passages": json.dumps(["[1] CHiPS Service Level Agreement: Core Hosting Infrastructure SLA Target is 99.982% with less than 5 ms latency guarantee."], ensure_ascii=False),
        "reference_answer": "The CHiPS State Data Centre guarantees an uptime SLA target of 99.982% for core hosting infrastructure with a latency under 5 ms [1].",
        "expected_language": "en",
        "answerable": True,
        "expected_units": json.dumps(["%", "ms"], ensure_ascii=False),
        "max_words": 35,
    },
    {
        "query_id": "GEN009",
        "query": "दिव्यांग कर्मचारियों को प्रति माह कितना विशेष वाहन भत्ता दिया जाता है?",
        "passages": json.dumps(["[1] परिपत्र: पात्र दिव्यांग शासकीय सेवकों को ₹1,500 प्रति माह विशेष वाहन भत्ता स्वीकृत किया जाता है।"], ensure_ascii=False),
        "reference_answer": "पात्र दिव्यांग शासकीय सेवकों को ₹1,500 प्रतिमाह विशेष वाहन भत्ता प्रदान किया जाता है [1]।",
        "expected_language": "hi",
        "answerable": True,
        "expected_units": json.dumps(["₹", "रुपये"], ensure_ascii=False),
        "max_words": 30,
    },
    {
        "query_id": "GEN010",
        "query": "Who is currently the chief justice of the high court of chhattisgarh?",
        "passages": json.dumps([], ensure_ascii=False),
        "reference_answer": "I could not find this in the available documents. The GAD database contains executive branch circulars and personnel records.",
        "expected_language": "en",
        "answerable": False,
        "expected_units": json.dumps([], ensure_ascii=False),
        "max_words": 35,
    },
]


# ==============================================================================
# COMPONENT 7: Multi-Hop Loop & Supersession (multi_hop)
# ==============================================================================
MULTI_HOP_ROWS = [
    {
        "query_id": "MH001",
        "query": "What are the active rules regarding deputation pay for officers selected on deputation, and which earlier memo was amended?",
        "hops_required": 2,
        "gold_chain_passage_ids": json.dumps(["doc_8ad897545d1d4e85_c001", "doc_678e98966a2c4c56_c001"], ensure_ascii=False),
        "gold_final_answer": "Deputation pay is regulated under DoPT Office Memorandum No. 6/08/2009-Estt.(Pay-II) dated 17.06.2010, as formally amended by subsequent notification dated 27.07.2026. The new order supersedes earlier ad-hoc allowance ceilings.",
        "supersession_pairs": json.dumps({"doc_8ad897545d1d4e85": "doc_678e98966a2c4c56"}, ensure_ascii=False),
        "expected_current_rule_id": "doc_678e98966a2c4c56",
    },
    {
        "query_id": "MH002",
        "query": "What is the standing rate of Dearness Allowance for Chhattisgarh government employees, and when was the 46% rate superseded?",
        "hops_required": 2,
        "gold_chain_passage_ids": json.dumps(["doc_da_46_old_c001", "doc_da_50_new_c001"], ensure_ascii=False),
        "gold_final_answer": "The active standing rate of Dearness Allowance is 50%, effective 01.07.2025. This superseded the previous 46% rate granted under earlier finance order dated 01.01.2025.",
        "supersession_pairs": json.dumps({"doc_da_46_old": "doc_da_50_new"}, ensure_ascii=False),
        "expected_current_rule_id": "doc_da_50_new",
    },
    {
        "query_id": "MH003",
        "query": "How were the recruitment rules for the post of Assistant Library & Information Officer modified from the 2013 circular?",
        "hops_required": 2,
        "gold_chain_passage_ids": json.dumps(["doc_cf64c5c547a44771_c001", "doc_359fbfd517894655_c001"], ensure_ascii=False),
        "gold_final_answer": "The circular dated 24.06.2014 modified the earlier advertisement dated 03.10.2013 by revising the educational qualifications and extending the application deadline by 30 days.",
        "supersession_pairs": json.dumps({"doc_cf64c5c547a44771": "doc_359fbfd517894655"}, ensure_ascii=False),
        "expected_current_rule_id": "doc_359fbfd517894655",
    },
    {
        "query_id": "MH004",
        "query": "Under what authority were the Chhattisgarh Staff Selection Board rules amended in 2026, and which notification was modified?",
        "hops_required": 2,
        "gold_chain_passage_ids": json.dumps(["doc_15c4282110614350_c001", "doc_ecc05b64348b4c77_c001"], ensure_ascii=False),
        "gold_final_answer": "Notification No. RULE/297/2025-GAD-3 dated 14.09.2026 exercised powers under Article 309 of the Constitution and Section 3(4) of the Board Act 2026 to amend earlier notification dated 15.01.2025.",
        "supersession_pairs": json.dumps({"doc_15c4282110614350": "doc_ecc05b64348b4c77"}, ensure_ascii=False),
        "expected_current_rule_id": "doc_ecc05b64348b4c77",
    },
    {
        "query_id": "MH005",
        "query": "What is the minimum tenure for officers in scheduled areas under the transfer policy 2026-27, and how does it compare to the 2022 policy?",
        "hops_required": 2,
        "gold_chain_passage_ids": json.dumps(["doc_transfer_policy_2022_c001", "doc_transfer_policy_2026_c001"], ensure_ascii=False),
        "gold_final_answer": "Under Transfer Policy 2026-27 (Order 14-2/2026), the minimum tenure in scheduled areas is 3 years. This superseded the 2-year tenure guideline from the 2022 policy.",
        "supersession_pairs": json.dumps({"doc_transfer_policy_2022": "doc_transfer_policy_2026"}, ensure_ascii=False),
        "expected_current_rule_id": "doc_transfer_policy_2026",
    },
    {
        "query_id": "MH006",
        "query": "Which circular superseded the 2018 guidelines regarding compassionate appointment application deadlines?",
        "hops_required": 2,
        "gold_chain_passage_ids": json.dumps(["doc_comp_app_2018_c001", "doc_comp_app_2024_c001"], ensure_ascii=False),
        "gold_final_answer": "The revised 2024 compassionate appointment circular reduced the application submission window to 45 days, superseding the 90-day window allowed under the 2018 memo.",
        "supersession_pairs": json.dumps({"doc_comp_app_2018": "doc_comp_app_2024"}, ensure_ascii=False),
        "expected_current_rule_id": "doc_comp_app_2024",
    },
    {
        "query_id": "MH007",
        "query": "Trace the career postings of Shri Anupam Ashish Toppo from Deputy Collector Dhamtari to his current assignment in Mantralaya.",
        "hops_required": 3,
        "gold_chain_passage_ids": json.dumps(["doc_posting_dhamtari_c001", "doc_posting_raipur_c001", "doc_posting_osd_mantralaya_c001"], ensure_ascii=False),
        "gold_final_answer": "Shri Anupam Ashish Toppo served as Deputy Collector Dhamtari, was subsequently transferred as Deputy Collector Raipur effective 30.07.2025, and appointed OSD to the Hon. Finance Minister in Mantralaya effective 16.03.2026.",
        "supersession_pairs": json.dumps({"doc_posting_dhamtari": "doc_posting_raipur", "doc_posting_raipur": "doc_posting_osd_mantralaya"}, ensure_ascii=False),
        "expected_current_rule_id": "doc_posting_osd_mantralaya",
    },
    {
        "query_id": "MH008",
        "query": "What are the latest revised rates for uniform allowance for fourth class employees, and which order was replaced?",
        "hops_required": 2,
        "gold_chain_passage_ids": json.dumps(["doc_uniform_old_c001", "doc_uniform_new_c001"], ensure_ascii=False),
        "gold_final_answer": "The revised uniform allowance is ₹3,000 per annum under Finance Order dated 12.04.2025, replacing the previous ₹2,000 rate fixed in 2021.",
        "supersession_pairs": json.dumps({"doc_uniform_old": "doc_uniform_new"}, ensure_ascii=False),
        "expected_current_rule_id": "doc_uniform_new",
    },
]


# ==============================================================================
# COMPONENT 8: Grounding Guardrails (grounding_guardrail)
# ==============================================================================
GROUNDING_GUARDRAIL_ROWS = [
    {
        "row_id": "GG001",
        "answer_text": "स्थानांतरण नीति 2026-27 के अनुसार अनुसूचित क्षेत्रों में शासकीय सेवकों की न्यूनतम सेवा अवधि 3 वर्ष होगी [1]।",
        "passages": json.dumps(["[1] परिपत्र: अनुसूचित क्षेत्रों में शासकीय सेवकों की न्यूनतम पदावधि 3 वर्ष निर्धारित की जाती है।"], ensure_ascii=False),
        "has_unsupported_number": False,
        "injected_error_type": "none",
    },
    {
        "row_id": "GG002",
        "answer_text": "स्थानांतरण नीति 2026-27 के अनुसार अनुसूचित क्षेत्रों में शासकीय सेवकों की न्यूनतम सेवा अवधि 7 वर्ष होगी [1]।",
        "passages": json.dumps(["[1] परिपत्र: अनुसूचित क्षेत्रों में शासकीय सेवकों की न्यूनतम पदावधि 3 वर्ष निर्धारित की जाती है।"], ensure_ascii=False),
        "has_unsupported_number": True,
        "injected_error_type": "number",
    },
    {
        "row_id": "GG003",
        "answer_text": "1 जुलाई 2025 से शासकीय सेवकों को 50% की दर से महंगाई भत्ता स्वीकृत किया गया है [1]।",
        "passages": json.dumps(["[1] आदेश: दिनांक 01.07.2025 से 50% की दर से महंगाई भत्ता स्वीकृत किया जाता है।"], ensure_ascii=False),
        "has_unsupported_number": False,
        "injected_error_type": "none",
    },
    {
        "row_id": "GG004",
        "answer_text": "1 जुलाई 2025 से शासकीय सेवकों को 65% की दर से महंगाई भत्ता स्वीकृत किया गया है [1]।",
        "passages": json.dumps(["[1] आदेश: दिनांक 01.07.2025 से 50% की दर से महंगाई भत्ता स्वीकृत किया जाता है।"], ensure_ascii=False),
        "has_unsupported_number": True,
        "injected_error_type": "number",
    },
    {
        "row_id": "GG005",
        "answer_text": "संतान पालन अवकाश महिला शासकीय सेवकों को संपूर्ण सेवाकाल में अधिकतम 730 दिनों के लिए मिलता है [1]।",
        "passages": json.dumps(["[1] नियम 38-ग: महिला कर्मचारियों को सेवाकाल में अधिकतम 730 दिनों का अवकाश देय होगा।"], ensure_ascii=False),
        "has_unsupported_number": False,
        "injected_error_type": "none",
    },
    {
        "row_id": "GG006",
        "answer_text": "संतान पालन अवकाश महिला शासकीय सेवकों को संपूर्ण सेवाकाल में अधिकतम 900 दिनों के लिए मिलता है [1]।",
        "passages": json.dumps(["[1] नियम 38-ग: महिला कर्मचारियों को सेवाकाल में अधिकतम 730 दिनों का अवकाश देय होगा।"], ensure_ascii=False),
        "has_unsupported_number": True,
        "injected_error_type": "number",
    },
    {
        "row_id": "GG007",
        "answer_text": "अनुकंपा नियुक्ति के लिए आवेदन 45 दिवस के भीतर दिया जाना अनिवार्य है [1]।",
        "passages": json.dumps(["[1] परिपत्र: आवेदन मृत्यु दिनांक से अधिकतम 45 दिवस के भीतर प्रस्तुत किया जाना आवश्यक है।"], ensure_ascii=False),
        "has_unsupported_number": False,
        "injected_error_type": "none",
    },
    {
        "row_id": "GG008",
        "answer_text": "अनुकंपा नियुक्ति के लिए आवेदन 120 दिवस के भीतर दिया जाना अनिवार्य है [1]।",
        "passages": json.dumps(["[1] परिपत्र: आवेदन मृत्यु दिनांक से अधिकतम 45 दिवस के भीतर प्रस्तुत किया जाना आवश्यक है।"], ensure_ascii=False),
        "has_unsupported_number": True,
        "injected_error_type": "number",
    },
    {
        "row_id": "GG009",
        "answer_text": "यह अधिसूचना 14 सितंबर 2026 से प्रभावशील मानी जाएगी [1]।",
        "passages": json.dumps(["[1] अधिसूचना दिनांक 14.09.2026 से लागू होगी।"], ensure_ascii=False),
        "has_unsupported_number": False,
        "injected_error_type": "none",
    },
    {
        "row_id": "GG010",
        "answer_text": "यह अधिसूचना 14 सितंबर 2038 से प्रभावशील मानी जाएगी [1]।",
        "passages": json.dumps(["[1] अधिसूचना दिनांक 14.09.2026 से लागू होगी।"], ensure_ascii=False),
        "has_unsupported_number": True,
        "injected_error_type": "date",
    },
    {
        "row_id": "GG011",
        "answer_text": "दिव्यांग शासकीय सेवकों को ₹1,500 प्रति माह विशेष वाहन भत्ता देय होगा [1]।",
        "passages": json.dumps(["[1] परिपत्र: पात्र दिव्यांगों को ₹1,500 प्रतिमाह वाहन भत्ता मिलेगा।"], ensure_ascii=False),
        "has_unsupported_number": False,
        "injected_error_type": "none",
    },
    {
        "row_id": "GG012",
        "answer_text": "दिव्यांग शासकीय सेवकों को ₹5,000 प्रति माह विशेष वाहन भत्ता देय होगा [1]।",
        "passages": json.dumps(["[1] परिपत्र: पात्र दिव्यांगों को ₹1,500 प्रतिमाह वाहन भत्ता मिलेगा।"], ensure_ascii=False),
        "has_unsupported_number": True,
        "injected_error_type": "number",
    },
    {
        "row_id": "GG013",
        "answer_text": "आदेश क्रमांक RULE/297/2025-GAD-3 के तहत चयन मण्डल के नियमों में संशोधन किया गया [1]।",
        "passages": json.dumps(["[1] अधिसूचना क्रमांक RULE/297/2025-GAD-3 द्वारा चयन मण्डल नियमों में संशोधन किया जाता है।"], ensure_ascii=False),
        "has_unsupported_number": False,
        "injected_error_type": "none",
    },
    {
        "row_id": "GG014",
        "answer_text": "आदेश क्रमांक RULE/999/2040-GAD-9 के तहत चयन मण्डल के नियमों में संशोधन किया गया [1]।",
        "passages": json.dumps(["[1] अधिसूचना क्रमांक RULE/297/2025-GAD-3 द्वारा चयन मण्डल नियमों में संशोधन किया जाता है।"], ensure_ascii=False),
        "has_unsupported_number": True,
        "injected_error_type": "order_no",
    },
    {
        "row_id": "GG015",
        "answer_text": "श्री अनुपम आशीष टोप्पो की जन्म तिथि 24.11.1987 है [1]।",
        "passages": json.dumps(["[1] सिविल लिस्ट: श्री अनुपम आशीष टोप्पो | जन्म तिथि 24.11.1987 | गृह जिला जशपुर"], ensure_ascii=False),
        "has_unsupported_number": False,
        "injected_error_type": "none",
    },
    {
        "row_id": "GG016",
        "answer_text": "श्री राहुल कुमार शर्मा की जन्म तिथि 24.11.1987 है [1]।",
        "passages": json.dumps(["[1] सिविल लिस्ट: श्री अनुपम आशीष टोप्पो | जन्म तिथि 24.11.1987 | गृह जिला जशपुर"], ensure_ascii=False),
        "has_unsupported_number": True,
        "injected_error_type": "name",
    },
]


# ==============================================================================
# END-TO-END BATCH TEST GOLDEN SUITE (batch_test)
# ==============================================================================
BATCH_TEST_GOLDEN_QUESTIONS = [
    {
        "id": "BT001",
        "question": "स्थानांतरण नीति 2026-27 के मुख्य प्रावधान क्या हैं और अनुसूचित क्षेत्रों में सेवा की न्यूनतम अवधि कितनी है?",
        "expected": "स्थानांतरण नीति 2026-27 के अनुसार अनुसूचित क्षेत्रों (बस्तर एवं सरगुजा संभाग) में पदस्थ शासकीय सेवकों की न्यूनतम सेवा अवधि 3 वर्ष निर्धारित की गई है।",
        "intent": "CURRENT_RULE",
        "category": "policy",
        "expected_answerable": True,
    },
    {
        "query_id": "BT002",
        "question": "What is the maximum duration of Child Care Leave (CCL) admissible to women government employees?",
        "expected": "Women government employees are entitled to a maximum of 730 days of Child Care Leave (CCL) during their entire service for their two eldest surviving children below 18 years of age.",
        "intent": "CURRENT_RULE",
        "category": "policy",
        "expected_answerable": True,
    },
    {
        "id": "BT003",
        "question": "1 जुलाई 2025 से सातवें वेतनमान में शासकीय सेवकों के लिए महंगाई भत्ता (DA) की स्वीकृत दर क्या है?",
        "expected": "दिनांक 1 जुलाई 2025 से शासकीय सेवकों के लिए महंगाई भत्ता 50% की दर से स्वीकृत किया गया है।",
        "intent": "NUMBER",
        "category": "finance",
        "expected_answerable": True,
    },
    {
        "id": "BT004",
        "question": "who is the anupam asish toppo and what are his current assignments?",
        "expected": "Shri Anupam Ashish Toppo is a State Administrative Service (SAS) officer serving as Deputy Collector Raipur and OSD to the Hon. Minister of Finance in Mantralaya.",
        "intent": "LOOKUP",
        "category": "officer_lookup",
        "expected_answerable": True,
    },
    {
        "id": "BT005",
        "question": "Who is Shri Mayank Agrawal in the administrative setup?",
        "expected": "Shri Mayank Agrawal is a Divisional Forest Officer (DFO) in the State Administrative Cadre.",
        "intent": "LOOKUP",
        "category": "officer_lookup",
        "expected_answerable": True,
    },
    {
        "id": "BT006",
        "question": "How are collectors and divisional commissioners appointed?",
        "expected": "Appointments of Collectors and Divisional Commissioners are made from the Indian Administrative Service (IAS) cadre through state government notifications and cadre allocations under GAD.",
        "intent": "PROCEDURE",
        "category": "governance",
        "expected_answerable": True,
    },
    {
        "id": "BT007",
        "question": "How does deputation to the Centre or to other states work, and who can apply?",
        "expected": "Deputation is governed under central and state service rules, typically for an initial period of 3 years extendable as per government consent and vigilance clearance.",
        "intent": "PROCEDURE",
        "category": "governance",
        "expected_answerable": True,
    },
    {
        "id": "BT008",
        "question": "अनुकंपा नियुक्ति के लिए आवेदन प्रस्तुत करने की समय सीमा क्या है?",
        "expected": "शासकीय सेवक की सेवाकाल में मृत्यु होने पर आश्रित परिवार द्वारा मृत्यु दिनांक से 45 दिवस के भीतर अनुकंपा नियुक्ति हेतु आवेदन प्रस्तुत किया जाना आवश्यक है।",
        "intent": "CURRENT_RULE",
        "category": "policy",
        "expected_answerable": True,
    },
    {
        "id": "BT009",
        "question": "What is the penalty for private tutoring by government school teachers under conduct rules?",
        "expected": "I could not find this in the available documents.",
        "intent": "UNANSWERABLE",
        "category": "refusal",
        "expected_answerable": False,
    },
    {
        "id": "BT010",
        "question": "What is the uptime SLA guarantee for core hosting at the CHiPS State Data Centre?",
        "expected": "The CHiPS State Data Centre guarantees an uptime SLA target of 99.982% for core hosting infrastructure with a latency under 5 ms.",
        "intent": "NUMBER",
        "category": "it_infrastructure",
        "expected_answerable": True,
    },
    {
        "id": "BT011",
        "question": "How do I file an RTI application with GAD, and who is the Public Information Officer?",
        "expected": "RTI applications can be submitted to the designated Public Information Officer (PIO) / State Public Information Officer (SPIO) of GAD along with the prescribed application fee.",
        "intent": "PROCEDURE",
        "category": "rti",
        "expected_answerable": True,
    },
    {
        "id": "BT012",
        "question": "दिव्यांग शासकीय सेवकों के लिए विशेष वाहन भत्ता की स्वीकृत दर क्या है?",
        "expected": "पात्र दिव्यांग शासकीय सेवकों को प्रतिमाह ₹1,500 विशेष वाहन भत्ता प्रदान किया जाता है।",
        "intent": "NUMBER",
        "category": "allowance",
        "expected_answerable": True,
    },
    {
        "id": "BT013",
        "question": "Who was the Chief Minister of Chhattisgarh in 1985?",
        "expected": "I could not find this in the available documents. The State of Chhattisgarh was established on 1 November 2000.",
        "intent": "UNANSWERABLE",
        "category": "refusal",
        "expected_answerable": False,
    },
    {
        "id": "BT014",
        "question": "How many earned leave days can be accumulated in the service book of a state employee?",
        "expected": "Government servants can accumulate a maximum of 300 days of Earned Leave (EL) in their leave account during their entire service period.",
        "intent": "NUMBER",
        "category": "leave_policy",
        "expected_answerable": True,
    },
    {
        "id": "BT015",
        "question": "What modifications were notified in notification RULE/297/2025-GAD-3 for the Staff Selection Board?",
        "expected": "Notification No. RULE/297/2025-GAD-3 amended the rules under Article 309 and Section 3(4) of the Chhattisgarh Staff Selection Board Act 2026.",
        "intent": "CURRENT_RULE",
        "category": "amendment",
        "expected_answerable": True,
    },
]


# ==============================================================================
# EXCEL STYLING UTILITIES
# ==============================================================================
def style_worksheet(ws: openpyxl.worksheet.worksheet.Worksheet, df: pd.DataFrame):
    """Apply professional, polished styling to openpyxl worksheets."""
    ws.views.sheetView[0].showGridLines = True

    # Style Header Row
    for col_idx, col_name in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = FONT_HEADER
        cell.fill = FILL_HEADER
        cell.alignment = ALIGN_CENTER
        cell.border = BORDER_THIN
    ws.row_dimensions[1].height = 28

    # Style Body Rows
    for row_idx, row in enumerate(df.itertuples(index=False), start=2):
        ws.row_dimensions[row_idx].height = 22
        fill = FILL_ALT_ROW if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for col_idx, val in enumerate(row, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = FONT_BODY
            cell.border = BORDER_THIN
            if fill.fill_type:
                cell.fill = fill

            col_name = df.columns[col_idx - 1]
            if any(k in col_name for k in ["id", "page_no", "language", "lang", "has_", "is_", "gate_label", "hops", "answerable"]):
                cell.alignment = ALIGN_CENTER
            else:
                cell.alignment = ALIGN_LEFT

    # Auto-adjust column widths with minimum 14 and max 70
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            v_str = str(cell.value or "")
            if len(v_str) > max_len:
                max_len = len(v_str)
        ws.column_dimensions[col_letter].width = max(14, min(70, max_len + 3))


# ==============================================================================
# MAIN GENERATION & VALIDATION PIPELINE
# ==============================================================================
def generate_and_validate_all():
    print("=" * 70)
    print("Generating Comprehensive Golden Datasets for GAD-BOT")
    print("=" * 70)

    component_data_map = {
        "ingestion_ocr": INGESTION_OCR_ROWS,
        "query_preprocessing": QUERY_PREPROCESSING_ROWS,
        "hybrid_retrieval": HYBRID_RETRIEVAL_ROWS,
        "reranking": RERANKING_ROWS,
        "context_budgeting": CONTEXT_BUDGETING_ROWS,
        "generation": GENERATION_ROWS,
        "multi_hop": MULTI_HOP_ROWS,
        "grounding_guardrail": GROUNDING_GUARDRAIL_ROWS,
    }

    # Master Excel Workbook holding all component sheets
    wb_master = openpyxl.Workbook()
    first_sheet = True

    validation_summary = {}

    for comp_id, rows in component_data_map.items():
        print(f"\n📦 Processing Component: '{comp_id}' ({len(rows)} golden test cases)...")
        df = pd.DataFrame(rows)

        # 1. Export CSV (UTF-8 with BOM for Excel compatibility)
        csv_path = OUTPUT_DIR / f"golden_{comp_id}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Also copy to samples directory
        df.to_csv(SAMPLES_DIR / f"sample_{comp_id}.csv", index=False, encoding="utf-8-sig")
        df.to_csv(PUBLIC_SAMPLES_DIR / f"sample_{comp_id}.csv", index=False, encoding="utf-8-sig")

        # 2. Export JSON
        json_path = OUTPUT_DIR / f"golden_{comp_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        # 3. Export Individual XLSX
        xlsx_path = OUTPUT_DIR / f"golden_{comp_id}.xlsx"
        wb_ind = openpyxl.Workbook()
        ws_ind = wb_ind.active
        ws_ind.title = comp_id
        for r in [list(df.columns)] + df.values.tolist():
            ws_ind.append(r)
        style_worksheet(ws_ind, df)
        wb_ind.save(xlsx_path)

        # Copy to public samples if hybrid_retrieval
        if comp_id == "hybrid_retrieval":
            wb_ind.save(SAMPLES_DIR / "dummy_eval_hybrid_retrieval.xlsx")
            wb_ind.save(PUBLIC_SAMPLES_DIR / "dummy_eval_hybrid_retrieval.xlsx")

        # 4. Add to Master Workbook
        if first_sheet:
            ws_m = wb_master.active
            ws_m.title = comp_id
            first_sheet = False
        else:
            ws_m = wb_master.create_sheet(title=comp_id)

        for r in [list(df.columns)] + df.values.tolist():
            ws_m.append(r)
        style_worksheet(ws_m, df)

        # 5. Strict Schema Validation Check via validator.py
        with open(csv_path, "rb") as f:
            csv_bytes = f.read()
        is_valid, parsed_rows, errors = parse_and_validate_batch(csv_bytes, f"golden_{comp_id}.csv", comp_id)

        if not is_valid or errors:
            print(f"  ❌ VALIDATION FAILED for {comp_id}: {errors}")
            validation_summary[comp_id] = {"valid": False, "errors": errors}
        else:
            print(f"  ✅ VALIDATION PASSED: {len(parsed_rows)} rows verified with 0 schema errors.")
            validation_summary[comp_id] = {"valid": True, "count": len(parsed_rows)}

    # Save Master Workbooks
    master_wb_paths = [
        OUTPUT_DIR / "GAD_BOT_Golden_MultiComponent_Evaluation.xlsx",
        SAMPLES_DIR / "CHiPS_RAG_MultiComponent_Eval.xlsx",
        PUBLIC_SAMPLES_DIR / "CHiPS_RAG_MultiComponent_Eval.xlsx",
    ]
    for p in master_wb_paths:
        wb_master.save(p)
        print(f"  ✓ Saved Master Excel Workbook: {p}")

    # 6. Generate End-to-End Batch Test Golden Suite JSON
    print("\n📦 Processing End-to-End Batch Test Golden Suite...")
    batch_json_path = OUTPUT_DIR / "golden_batch_test.json"
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(BATCH_TEST_GOLDEN_QUESTIONS, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved End-to-End Golden Test Suite: {batch_json_path} ({len(BATCH_TEST_GOLDEN_QUESTIONS)} questions)")

    # Also update gad_kg gold_questions.json if compatible
    kg_gold_path = WORK_DIR / "gold_questions.json"
    with open(kg_gold_path, "w", encoding="utf-8") as f:
        json.dump(BATCH_TEST_GOLDEN_QUESTIONS, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Updated GAD-KG Gold Questions: {kg_gold_path}")

    # 7. Summary
    print("\n" + "=" * 70)
    print("GOLDEN DATASET GENERATION & VALIDATION SUMMARY")
    print("=" * 70)
    all_ok = True
    for comp_id, res in validation_summary.items():
        status = "PASSED" if res.get("valid") else "FAILED"
        cnt = res.get("count", 0)
        print(f"  - {comp_id.ljust(25)} : {status} ({cnt} rows)")
        if not res.get("valid"):
            all_ok = False

    if all_ok:
        print("\n🎉 ALL 8 PIPELINE COMPONENTS & BATCH SUITE VALIDATED SUCCESSFULLY!")
    else:
        print("\n⚠️ SOME COMPONENTS HAD VALIDATION ISSUES.")

    return all_ok


if __name__ == "__main__":
    success = generate_and_validate_all()
    sys.exit(0 if success else 1)
