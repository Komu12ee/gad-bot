# GPU OCR preprocessing

This folder contains the OCR implementation used by the GPU VM:

- `ocr_scanned.py`: selects scraped PDFs, resumes runs, and writes document results.
- `vlm_ocr.py`: renders page images and sends them to the local Qwen VLM server.
- `page_orientation.py`: records whether a page appears rotated for review.
- `doc_metadata.py`: extracts document metadata using rules and listing records.
- Quality checks and safe text cleanup live in `../02_optimization/text_gate.py`.
- Model metadata, spelling review and name aliases live in `../02_optimization/`.

From the repository root:

```bash
bash deploy/gpu-vm.sh ocr
```

This starts the OCR server, runs the trial and both OCR passes, and keeps logs in
`logs/gpu-vm/`. Model selection still uses the existing `OCR_PRIMARY_MODEL`,
`OCR_FALLBACK_MODEL` and `OCR_VLM_*` settings.

For an already running OCR server:

```bash
PIPELINE=fg2 uv run python core/01_preprocessing/ocr_scanned.py \
  --engine vlm --corpus data/scrapped_data/fg2
```

All PDF pages are OCR'd by default. PyMuPDF renders page images; Qwen reads the
text. Existing output is skipped unless explicitly forced. Output stays under
`data/<pipeline>/stage2/`; moving the code does not change the data paths.

The unused Docling/EasyOCR image-preparation modules, runners, diagnostics and
accidental pip redirection files were removed. Legacy stage-1/stage-2 commands
are no longer supported. Docling remains a dependency of the optional legacy
chunker in `03_chunking`.
