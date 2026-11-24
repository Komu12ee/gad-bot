"""Generate dummy evaluation datasets and Excel files for manual testing of the evaluation UI.

Outputs:
1. core/eval/samples/dummy_eval_hybrid_retrieval.xlsx
2. core/eval/samples/dummy_eval_hybrid_retrieval.csv
3. core/eval/samples/CHiPS_RAG_MultiComponent_Eval.xlsx
4. Copies into core/05_webui/nodejs/public/samples/ for browser direct download.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = REPO_ROOT / "core" / "eval" / "samples"
PUBLIC_SAMPLES_DIR = REPO_ROOT / "core" / "05_webui" / "nodejs" / "public" / "samples"
TEST_DATA_DIR = REPO_ROOT / "test_data" / "eval"

for d in [SAMPLES_DIR, PUBLIC_SAMPLES_DIR, TEST_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Styling definitions for professional Excel appearance ─────────────────────
FONT_HEADER = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
FILL_HEADER = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")  # Navy Blue
FILL_ALT_ROW = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid") # Slate 50
FONT_BODY = Font(name="Calibri", size=10, color="0F172A")
BORDER_THIN = Border(
    left=Side(style="thin", color="E2E8F0"),
    right=Side(style="thin", color="E2E8F0"),
    top=Side(style="thin", color="E2E8F0"),
    bottom=Side(style="thin", color="E2E8F0"),
)
ALIGN_CENTER = Alignment(horizontal="center", vertical="center")
ALIGN_LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)

# ── Component 3: Hybrid Retrieval Data ───────────────────────────────────────
HYBRID_RETRIEVAL_ROWS = [
    {
        "query_id": "Q001",
        "query": "स्थानांतरण नीति 2026-27 के मुख्य प्रावधान क्या हैं?",
        "gold_passage_ids": json.dumps(["circular_transfer_2026.pdf#1", "circular_transfer_2026.pdf#2"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q002",
        "query": "What is the maximum age relaxation for OBC candidates in government recruitment?",
        "gold_passage_ids": json.dumps(["gad_recruitment_rules_2024.pdf#4", "age_relaxation_memo.pdf#1"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q003",
        "query": "आदेश क्रमांक एफ-2-4/2026/1-3 में अनुकंपा नियुक्ति के संबंध में क्या निर्देश हैं?",
        "gold_passage_ids": json.dumps(["order_f2_4_2026.pdf#1"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": "true",
    },
    {
        "query_id": "Q004",
        "query": "What are the major e-governance and IT infrastructure projects implemented by CHiPS?",
        "gold_passage_ids": json.dumps(["c74f4f5e4d5ce7f8", "e8b15659eb74c65f"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "en",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q005",
        "query": "छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (CHiPS) की स्थापना और मुख्य उद्देश्य",
        "gold_passage_ids": json.dumps(["5f369c50ead865ab", "3175465fb5fc7c23"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q006",
        "query": "Circular No. 14/2024/Fin regarding contingency fund expenditure ceiling",
        "gold_passage_ids": json.dumps(["fin_circular_14_2024.pdf#1"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "en",
        "is_exact_match_query": "true",
    },
    {
        "query_id": "Q007",
        "query": "शासकीय सेवकों के लिए 1 जुलाई 2025 से महंगाई भत्ता (DA) दरें",
        "gold_passage_ids": json.dumps(["finance_da_order_2025.pdf#1", "finance_da_order_2025.pdf#2"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q008",
        "query": "divyang karmachariyon ke liye vahan bhatta niyam aur eligibility",
        "gold_passage_ids": json.dumps(["divyang_allowance_circular.pdf#1"], ensure_ascii=False),
        "query_language": "hinglish",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q009",
        "query": "What is the procedure for Child Care Leave (CCL) for women government servants?",
        "gold_passage_ids": json.dumps(["gad_leave_rules_2024.pdf#2"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q010",
        "query": "संविदा भर्ती नियमों के तहत मातृत्व अवकाश और आकस्मिक अवकाश पात्रता",
        "gold_passage_ids": json.dumps(["contract_service_rules_2022.pdf#5"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q011",
        "query": "Notification No. GAD-EST-2023-88 regarding administrative officers seniority",
        "gold_passage_ids": json.dumps(["gad_order_88_2023.pdf#1"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "en",
        "is_exact_match_query": "true",
    },
    {
        "query_id": "Q012",
        "query": "State Data Centre (SDC) hosting services and cloud infrastructure under CHiPS",
        "gold_passage_ids": json.dumps(["3175465fb5fc7c23", "c74f4f5e4d5ce7f8"], ensure_ascii=False),
        "query_language": "en",
        "doc_language": "en",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q013",
        "query": "चतुर्थ श्रेणी कर्मचारियों के लिए वर्दी एवं धुलाई भत्ता की संशोधित दरें",
        "gold_passage_ids": json.dumps(["uniform_allowance_2024.pdf#1"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q014",
        "query": "chhattisgarh sachivalaya me online RTI apply karne ka niyam",
        "gold_passage_ids": json.dumps(["rti_rules_manual.pdf#5"], ensure_ascii=False),
        "query_language": "hinglish",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
    {
        "query_id": "Q015",
        "query": "विभागीय पदोन्नति समिति (DPC) की बैठक आयोजित करने की समय सारिणी",
        "gold_passage_ids": json.dumps(["dpc_calendar_circular.pdf#1"], ensure_ascii=False),
        "query_language": "hi",
        "doc_language": "hi",
        "is_exact_match_query": "false",
    },
]

# ── Component 2: Query Preprocessing Data ─────────────────────────────────────
QUERY_PREPROCESSING_ROWS = [
    {
        "query_id": "QP001",
        "query": "अनुकंपा नियुक्ति के नियम क्या हैं?",
        "true_language": "hi",
        "true_intent": "general",
        "expected_expansion_terms": json.dumps(["अनुकंपा", "नियुक्ति", "नियम"], ensure_ascii=False),
    },
    {
        "query_id": "QP002",
        "query": "vetan kab milega aur da kitna badha hai?",
        "true_language": "hinglish",
        "true_intent": "general",
        "expected_expansion_terms": json.dumps(["वेतन", "महंगाई भत्ता"], ensure_ascii=False),
    },
    {
        "query_id": "QP003",
        "query": "GAD ke published reports aur circulars kaun se hain?",
        "true_language": "hinglish",
        "true_intent": "catalogue",
        "expected_expansion_terms": json.dumps(["सामान्य प्रशासन विभाग", "प्रतिवेदन"], ensure_ascii=False),
    },
    {
        "query_id": "QP004",
        "query": "2026-27 में पूंजीगत व्यय का कुल बजट प्रावधान कितना है?",
        "true_language": "hi",
        "true_intent": "number",
        "expected_expansion_terms": json.dumps(["पूंजीगत व्यय", "बजट"], ensure_ascii=False),
    },
    {
        "query_id": "QP005",
        "query": "Hello, good morning! Can you help me?",
        "true_language": "en",
        "true_intent": "greeting",
        "expected_expansion_terms": json.dumps([], ensure_ascii=False),
    },
    {
        "query_id": "QP006",
        "query": "How many circulars were issued by Finance Department in 2024?",
        "true_language": "en",
        "true_intent": "count",
        "expected_expansion_terms": json.dumps(["परिपत्र", "वित्त विभाग"], ensure_ascii=False),
    },
]

# ── Component 6: Generation (LLM) Data ────────────────────────────────────────
GENERATION_ROWS = [
    {
        "query_id": "GEN001",
        "query": "स्थानांतरण नीति 2026-27 के तहत न्यूनतम सेवा अवधि क्या है?",
        "passages": json.dumps(["[1] परिपत्र क्र. 14/2026: अनुसूचित क्षेत्रों में शासकीय सेवकों की न्यूनतम सेवा अवधि 3 वर्ष होगी।"], ensure_ascii=False),
        "reference_answer": "स्थानांतरण नीति 2026-27 के अनुसार अनुसूचित क्षेत्रों में शासकीय सेवकों के लिए न्यूनतम सेवा अवधि 3 वर्ष निर्धारित की गई है [1]।",
        "expected_language": "hi",
        "answerable": "true",
        "expected_units": json.dumps(["वर्ष"], ensure_ascii=False),
        "max_words": 40,
    },
    {
        "query_id": "GEN002",
        "query": "What is the penalty for late submission of annual property returns?",
        "passages": json.dumps([], ensure_ascii=False),
        "reference_answer": "I could not find this in the available documents.",
        "expected_language": "en",
        "answerable": "false",
        "expected_units": json.dumps([], ensure_ascii=False),
        "max_words": 30,
    },
    {
        "query_id": "GEN003",
        "query": "कर्मचारियों को 1 जुलाई 2025 से कितना प्रतिशत महंगाई भत्ता स्वीकृत किया गया है?",
        "passages": json.dumps(["[1] वित्त विभाग आदेश 01.07.2025: शासकीय सेवकों को 50% की दर से महंगाई भत्ता देय होगा।"], ensure_ascii=False),
        "reference_answer": "1 जुलाई 2025 से शासकीय सेवकों के लिए महंगाई भत्ता 50% स्वीकृत किया गया है [1]।",
        "expected_language": "hi",
        "answerable": "true",
        "expected_units": json.dumps(["%"], ensure_ascii=False),
        "max_words": 35,
    },
]

# ── Component 8: Grounding Guardrails Data ─────────────────────────────────────
GROUNDING_ROWS = [
    {
        "row_id": "GG001",
        "answer_text": "शासकीय सेवकों को 50% महंगाई भत्ता देय होगा [1]।",
        "passages": json.dumps(["[1] वित्त विभाग आदेश: महंगाई भत्ता दर 50% निर्धारित की जाती है।"], ensure_ascii=False),
        "has_unsupported_number": "false",
        "injected_error_type": "none",
    },
    {
        "row_id": "GG002",
        "answer_text": "शासकीय सेवकों को 65% महंगाई भत्ता देय होगा [1]।",
        "passages": json.dumps(["[1] वित्त विभाग आदेश: महंगाई भत्ता दर 50% निर्धारित की जाती है।"], ensure_ascii=False),
        "has_unsupported_number": "true",
        "injected_error_type": "number",
    },
    {
        "row_id": "GG003",
        "answer_text": "यह आदेश दिनांक 14.08.2035 से प्रभावशील माना जाएगा [1]।",
        "passages": json.dumps(["[1] आदेश दिनांक 14.08.2024 से लागू होगा।"], ensure_ascii=False),
        "has_unsupported_number": "true",
        "injected_error_type": "date",
    },
    {
        "row_id": "GG004",
        "answer_text": "आवेदन हेतु ₹500 का शुल्क निर्धारित किया गया है [1]।",
        "passages": json.dumps(["[1] आवेदन शुल्क 500 रुपये लिया जाएगा।"], ensure_ascii=False),
        "has_unsupported_number": "false",
        "injected_error_type": "none",
    },
]


def style_worksheet(ws: openpyxl.worksheet.worksheet.Worksheet, df: pd.DataFrame):
    """Apply styling, headers, borders, and column widths to an openpyxl worksheet."""
    ws.views.sheetView[0].showGridLines = True

    # Style Header
    for col_idx, col_name in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = FONT_HEADER
        cell.fill = FILL_HEADER
        cell.alignment = ALIGN_CENTER
        cell.border = BORDER_THIN

    ws.row_dimensions[1].height = 26

    # Style Body
    for row_idx, row in enumerate(df.itertuples(index=False), start=2):
        ws.row_dimensions[row_idx].height = 22
        fill = FILL_ALT_ROW if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for col_idx, val in enumerate(row, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = FONT_BODY
            cell.border = BORDER_THIN
            if fill.fill_type:
                cell.fill = fill
            # Center ID and boolean columns
            col_name = df.columns[col_idx - 1]
            if "id" in col_name or "lang" in col_name or "is_" in col_name or "has_" in col_name or "answerable" in col_name:
                cell.alignment = ALIGN_CENTER
            else:
                cell.alignment = ALIGN_LEFT

    # Auto-adjust column widths with safety margins
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val_str = str(cell.value or "")
            if len(val_str) > max_len:
                max_len = len(val_str)
        # Cap between 14 and 65
        ws.column_dimensions[col_letter].width = max(14, min(65, max_len + 3))


def generate_all():
    print("Generating evaluation test datasets...")

    # 1. Primary Hybrid Retrieval Single Excel File
    df_hybrid = pd.DataFrame(HYBRID_RETRIEVAL_ROWS)
    wb_hybrid = openpyxl.Workbook()
    ws_hybrid = wb_hybrid.active
    ws_hybrid.title = "hybrid_retrieval"
    for r in [list(df_hybrid.columns)] + df_hybrid.values.tolist():
        ws_hybrid.append(r)
    style_worksheet(ws_hybrid, df_hybrid)

    hybrid_xlsx_paths = [
        SAMPLES_DIR / "dummy_eval_hybrid_retrieval.xlsx",
        PUBLIC_SAMPLES_DIR / "dummy_eval_hybrid_retrieval.xlsx",
        TEST_DATA_DIR / "dummy_eval_hybrid_retrieval.xlsx",
    ]
    for p in hybrid_xlsx_paths:
        wb_hybrid.save(p)
        print(f"  ✓ Saved: {p}")

    # 2. Hybrid Retrieval CSV File (UTF-8 with BOM for seamless Excel double-click)
    hybrid_csv_paths = [
        SAMPLES_DIR / "dummy_eval_hybrid_retrieval.csv",
        PUBLIC_SAMPLES_DIR / "dummy_eval_hybrid_retrieval.csv",
        TEST_DATA_DIR / "dummy_eval_hybrid_retrieval.csv",
    ]
    for p in hybrid_csv_paths:
        df_hybrid.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"  ✓ Saved: {p}")

    # 3. Master Multi-Component Excel File
    wb_master = openpyxl.Workbook()
    # Sheet 1: Hybrid Retrieval
    ws1 = wb_master.active
    ws1.title = "hybrid_retrieval"
    for r in [list(df_hybrid.columns)] + df_hybrid.values.tolist():
        ws1.append(r)
    style_worksheet(ws1, df_hybrid)

    # Sheet 2: Query Preprocessing
    df_qp = pd.DataFrame(QUERY_PREPROCESSING_ROWS)
    ws2 = wb_master.create_sheet(title="query_preprocessing")
    for r in [list(df_qp.columns)] + df_qp.values.tolist():
        ws2.append(r)
    style_worksheet(ws2, df_qp)

    # Sheet 3: Generation
    df_gen = pd.DataFrame(GENERATION_ROWS)
    ws3 = wb_master.create_sheet(title="generation")
    for r in [list(df_gen.columns)] + df_gen.values.tolist():
        ws3.append(r)
    style_worksheet(ws3, df_gen)

    # Sheet 4: Grounding Guardrails
    df_gg = pd.DataFrame(GROUNDING_ROWS)
    ws4 = wb_master.create_sheet(title="grounding_guardrail")
    for r in [list(df_gg.columns)] + df_gg.values.tolist():
        ws4.append(r)
    style_worksheet(ws4, df_gg)

    master_xlsx_paths = [
        SAMPLES_DIR / "CHiPS_RAG_MultiComponent_Eval.xlsx",
        PUBLIC_SAMPLES_DIR / "CHiPS_RAG_MultiComponent_Eval.xlsx",
        TEST_DATA_DIR / "CHiPS_RAG_MultiComponent_Eval.xlsx",
    ]
    for p in master_xlsx_paths:
        wb_master.save(p)
        print(f"  ✓ Saved: {p}")

    print("All dummy evaluation datasets generated successfully!")


if __name__ == "__main__":
    generate_all()
