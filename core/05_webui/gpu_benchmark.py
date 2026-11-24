"""IndiaAI GPU Capacity Demonstration & Continuous Workload Benchmark.

Provides:
1. Real-time GPU telemetry (via `nvidia-smi` with hardware-aware fallback).
2. Continuous synthetic text and image request generator against the local model server (:8000)
   to keep the GPU tensor cores and VRAM fully saturated for capacity demonstration.
3. One-click Start and instant Stop controls with live throughput and latency metrics.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request
from PIL import Image, ImageDraw

logger = logging.getLogger("gpu_benchmark")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_LLM_URL = os.getenv("LLM_BASE_URL") or "http://127.0.0.1:8000/v1"
MOCK_MODE = os.getenv("GPU_BENCHMARK_MOCK", "").lower() in ("1", "true", "yes")

# Diverse synthetic government prompts for heavy transformer attention
SYNTHETIC_TEXT_PROMPTS = [
    (
        "छ.ग. सिविल सेवा (आचरण) नियम के अंतर्गत शासकीय सेवकों के आचरण, अवकाश एवं पेंशन स्वीकृति "
        "की संपूर्ण वैधानिक प्रक्रिया का विस्तार से विश्लेषण करें। सभी प्रासंगिक धाराओं एवं नियमों का उल्लेख करें।"
    ),
    (
        "Comprehensive review of Chhattisgarh General Administration Department circulars on public procurement, "
        "financial delegation of powers, and e-procurement compliance thresholds for state tenders."
    ),
    (
        "छत्तीसगढ़ शासन द्वारा शासकीय कर्मचारियों के स्थानांतरण नीति 2024-25 के अंतर्गत निर्धारित प्रतिबंधों, "
        "अपवादों एवं सक्षम प्राधिकारी की शक्तियों का तुलनात्मक विवरण प्रस्तुत करें।"
    ),
    (
        "Detailed structural analysis of State fiscal budget allocations across infrastructure, education, "
        "and public health with multi-year expenditure caps, revised estimates, and utilization audits."
    ),
    (
        "सूचना का अधिकार अधिनियम (RTI Act 2005) के तहत लोक सूचना अधिकारी (PIO) एवं प्रथम अपीलीय अधिकारी "
        "के वैधानिक दायित्वों, समय-सीमा, एवं दंडनीय प्रावधानों की क्रमबद्ध व्याख्या करें।"
    ),
    (
        "Generate a structured ministerial notification regarding creation of administrative posts, "
        "pay matrix Level 12 sanction, and staffing patterns under the Finance Department guidelines."
    ),
    (
        "शासकीय योजनाओं के सामाजिक अंकेक्षण (Social Audit) एवं जन शिकायत निवारण प्रणाली (Janta Samvad) "
        "के कार्यान्वयन हेतु प्रशासनिक कार्ययोजना एवं निगरानी तंत्र का खाका तैयार करें।"
    ),
    (
        "Deep technical and policy evaluation of artificial intelligence integration within state citizen "
        "services, focusing on bilingual LLM retrieval safety, hallucination guards, and data sovereign hosting."
    ),
]


# ── Telemetry Collector ───────────────────────────────────────────────────────
class GpuTelemetry:
    """Collects real-time GPU metrics from nvidia-smi with graceful fallback."""

    def __init__(self, history_len: int = 60):
        self._history: deque[dict[str, Any]] = deque(maxlen=history_len)
        self._lock = threading.Lock()
        self._cached: dict[str, Any] | None = None
        self._last_poll = 0.0

    def query_nvidia_smi(self) -> dict[str, Any] | None:
        """Query real GPU hardware metrics via nvidia-smi."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,driver_version,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 9:
                    def _float(v: str, default: float = 0.0) -> float:
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            return default

                    return {
                        "available": True,
                        "name": parts[0],
                        "driver_version": parts[1],
                        "gpu_util": _float(parts[2]),
                        "mem_util": _float(parts[3]),
                        "mem_used_mb": _float(parts[4]),
                        "mem_total_mb": _float(parts[5]),
                        "temperature": _float(parts[6]),
                        "power_draw_w": _float(parts[7]),
                        "power_limit_w": _float(parts[8]),
                        "source": "nvidia-smi",
                    }
        except Exception:
            pass
        return None

    def sample(self, is_workload_active: bool = False, active_workers: int = 0, tokens_per_sec: float = 0.0) -> dict[str, Any]:
        """Fetch latest telemetry, updating the rolling time-series window."""
        now = time.time()
        telemetry = self.query_nvidia_smi()

        if not telemetry:
            # Emulated telemetry (useful for local development and non-GPU machines)
            if is_workload_active:
                base_gpu = min(96.0, 78.0 + active_workers * 5.0 + random.uniform(-4.0, 5.0))
                mem_used = 56200.0 + random.uniform(-300.0, 500.0)
                temp = 54.0 + random.uniform(-1.0, 2.0)
                power = 325.0 + random.uniform(-15.0, 25.0)
            else:
                base_gpu = max(1.0, 3.0 + random.uniform(-1.0, 2.0))
                mem_used = 33400.0  # VLM idle weights
                temp = 39.0
                power = 72.0

            mem_total = 81920.0  # 80GB H100 representation
            telemetry = {
                "available": True,
                "name": "NVIDIA H100 80GB HBM3 (Emulated / Dev Mode)",
                "driver_version": "550.54.14",
                "gpu_util": round(base_gpu, 1),
                "mem_util": round((mem_used / mem_total) * 100, 1),
                "mem_used_mb": round(mem_used, 1),
                "mem_total_mb": mem_total,
                "temperature": round(temp, 1),
                "power_draw_w": round(power, 1),
                "power_limit_w": 700.0,
                "source": "emulated",
            }

        iso_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
        point = {
            "time": iso_time,
            "gpu_util": telemetry["gpu_util"],
            "mem_util": telemetry["mem_util"],
            "tokens_per_sec": round(tokens_per_sec, 1),
        }

        with self._lock:
            self._cached = telemetry
            self._history.append(point)
            self._last_poll = now

        return telemetry

    def get_latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._cached or {})

    def get_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)


# ── Synthetic Content Generator ───────────────────────────────────────────────
def generate_synthetic_document_image(seq_num: int = 1) -> str:
    """Generate a realistic synthetic government document page image as base64 JPEG."""
    width, height = 800, 1050
    img = Image.new("RGB", (width, height), color=(252, 252, 250))
    d = ImageDraw.Draw(img)

    # Outer document border
    d.rectangle([30, 30, width - 30, height - 30], outline=(40, 40, 40), width=2)
    d.rectangle([36, 36, width - 36, height - 36], outline=(180, 180, 180), width=1)

    # Header Emblem & Title
    d.ellipse([width // 2 - 30, 50, width // 2 + 30, 110], outline=(20, 60, 120), width=3)
    d.text((width // 2 - 190, 125), "GOVERNMENT OF CHHATTISGARH", fill=(20, 35, 75))
    d.text((width // 2 - 170, 145), "GENERAL ADMINISTRATION DEPARTMENT", fill=(20, 35, 75))
    d.text((width // 2 - 130, 165), "MANTRALAYA, MAHANADI BHAWAN, NAVA RAIPUR", fill=(70, 70, 70))
    d.line([50, 190, width - 50, 190], fill=(20, 60, 120), width=2)

    # Reference & Date
    ref_no = f"F 12-08/2026/GAD/BENCH-VLM-{seq_num:04d}"
    d.text((55, 205), f"Order No: {ref_no}", fill=(0, 0, 0))
    d.text((width - 240, 205), f"Date: {datetime.now().strftime('%d-%m-%Y')}", fill=(0, 0, 0))

    # Subject
    d.rectangle([50, 230, width - 50, 260], fill=(240, 244, 250), outline=(200, 215, 235))
    d.text((60, 238), f"SUBJECT: COMPREHENSIVE GPU CAPACITY AUDIT & VLM RETRIEVAL VERIFICATION #{seq_num}", fill=(10, 40, 90))

    # Paragraph text lines
    y = 280
    for line_idx in range(5):
        d.line([55, y, width - 55 - (line_idx * 35), y], fill=(60, 60, 60), width=2)
        y += 18

    # Table Grid
    y += 20
    d.rectangle([50, y, width - 50, y + 220], outline=(50, 50, 50), width=2)
    # Header row
    d.rectangle([50, y, width - 50, y + 30], fill=(230, 235, 245), outline=(50, 50, 50))
    d.text((65, y + 8), "S.No", fill=(0, 0, 0))
    d.text((130, y + 8), "Administrative Unit", fill=(0, 0, 0))
    d.text((360, y + 8), "Sanctioned Posts", fill=(0, 0, 0))
    d.text((530, y + 8), "Budget (Rs. Cr)", fill=(0, 0, 0))
    d.text((670, y + 8), "Status", fill=(0, 0, 0))

    # Table vertical dividers
    cols = [110, 340, 510, 650]
    for col_x in cols:
        d.line([col_x, y, col_x, y + 220], fill=(50, 50, 50), width=1)

    # Rows
    units = ["Finance Division", "State Data Centre", "GAD Vigilance", "IT Infrastructure", "Public Grievance", "Secretariat Pool"]
    row_y = y + 30
    for idx, unit in enumerate(units):
        d.line([50, row_y, width - 50, row_y], fill=(180, 180, 180), width=1)
        d.text((70, row_y + 6), str(idx + 1), fill=(40, 40, 40))
        d.text((125, row_y + 6), unit, fill=(20, 20, 20))
        d.text((400, row_y + 6), str(random.randint(15, 250)), fill=(30, 30, 30))
        d.text((560, row_y + 6), f"{random.uniform(12.5, 340.0):.2f}", fill=(30, 30, 30))
        d.text((680, row_y + 6), "SANCTIONED" if idx % 2 == 0 else "AUDITED", fill=(0, 120, 0) if idx % 2 == 0 else (140, 80, 0))
        row_y += 31

    # Official Seal & Signature Block
    seal_y = height - 200
    d.ellipse([70, seal_y, 190, seal_y + 120], outline=(180, 30, 30), width=2)
    d.text((95, seal_y + 40), "OFFICIAL SEAL", fill=(180, 30, 30))
    d.text((90, seal_y + 60), "GOVT OF CHHATTISGARH", fill=(180, 30, 30))

    d.text((width - 250, seal_y + 30), "Digitally Signed by:", fill=(0, 0, 0))
    d.text((width - 250, seal_y + 50), "Under Secretary to Govt", fill=(0, 0, 0))
    d.text((width - 250, seal_y + 70), "General Administration Dept", fill=(70, 70, 70))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Synthetic Workload Runner ────────────────────────────────────────────────
class SyntheticWorkloadRunner:
    """Manages continuous loop of synthetic text and image requests to local model server."""

    def __init__(self, telemetry: GpuTelemetry):
        self.telemetry = telemetry
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._sampler_thread: threading.Thread | None = None

        self.is_running = False
        self.concurrency = 2
        self.include_images = True
        self.start_time: float | None = None

        # Counters & Rates
        self.total_requests = 0
        self.text_requests = 0
        self.image_requests = 0
        self.failed_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.tokens_per_second = 0.0
        self.requests_per_second = 0.0
        self.avg_latency_ms = 0.0

        self.multimodal_supported = True
        self.target_url = DEFAULT_LLM_URL
        self.target_model = os.getenv("LLM_MODEL") or "ocr-32b"
        self.last_error: str | None = None

        self._recent_latencies: deque[float] = deque(maxlen=20)
        self._recent_logs: deque[dict[str, Any]] = deque(maxlen=30)
        self._tokens_window: deque[tuple[float, int]] = deque(maxlen=60)

    def discover_target_model(self) -> str:
        """Find currently loaded model from model server or fallback to config."""
        url = self.target_url.rstrip("/")
        try:
            import requests
            res = requests.get(f"{url}/models", timeout=2.0)
            if res.status_code == 200:
                data = res.json()
                models = data.get("data", [])
                if models and isinstance(models, list):
                    m_id = models[0].get("id")
                    if m_id:
                        self.target_model = m_id
                        return m_id
        except Exception:
            pass
        return self.target_model

    def _worker_loop(self, worker_id: int):
        """Worker thread executing continuous synthetic requests."""
        import requests

        logger.info(f"Workload worker-{worker_id} started against {self.target_url} ({self.target_model})")
        req_counter = 0

        while not self._stop_event.is_set():
            req_counter += 1
            t0 = time.time()
            # Decide request type: alternate or prioritize text if multimodal disabled
            do_image = (
                self.include_images
                and self.multimodal_supported
                and (req_counter % 2 == 0)
            )

            req_type = "image" if do_image else "text"
            prompt_preview = ""
            status_ok = False
            tokens_in = 0
            tokens_out = 0
            err_msg = None

            try:
                if MOCK_MODE:
                    # Simulated execution mode for testing/CI without live GPU
                    time.sleep(random.uniform(0.3, 0.7))
                    status_ok = True
                    tokens_in = random.randint(300, 800) if do_image else random.randint(150, 450)
                    tokens_out = random.randint(180, 320)
                    prompt_preview = f"Synthetic {'Image Multimodal' if do_image else 'Text QA'} #{req_counter}"
                else:
                    if do_image:
                        b64_img = generate_synthetic_document_image(req_counter)
                        payload = {
                            "model": self.target_model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                                        {"type": "text", "text": "Extract all numerical tables, administrative orders, and dates from this document."},
                                    ],
                                }
                            ],
                            "max_tokens": 280,
                            "temperature": 0.2,
                        }
                        prompt_preview = f"[Image + Prompt] Extract tables & sanctions ({len(b64_img) // 1024} KB JPEG)"
                    else:
                        prompt_text = random.choice(SYNTHETIC_TEXT_PROMPTS)
                        payload = {
                            "model": self.target_model,
                            "messages": [{"role": "user", "content": prompt_text}],
                            "max_tokens": 300,
                            "temperature": 0.7,
                        }
                        prompt_preview = prompt_text[:70] + "..."

                    headers = {"Content-Type": "application/json"}
                    res = requests.post(
                        f"{self.target_url.rstrip('/')}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=45.0,
                    )

                    if res.status_code == 200:
                        status_ok = True
                        data = res.json()
                        usage = data.get("usage", {})
                        tokens_in = usage.get("prompt_tokens") or (800 if do_image else 250)
                        tokens_out = usage.get("completion_tokens") or 200
                    elif res.status_code == 400 and ("image" in res.text.lower() or "multimodal" in res.text.lower()):
                        # Model is text-only (e.g. Sarvam-30B) - disable image attempts
                        self.multimodal_supported = False
                        logger.warning("Local model server is text-only; disabled image requests.")
                        err_msg = "Model server text-only (switched to text loop)"
                    else:
                        err_msg = f"HTTP {res.status_code}: {res.text[:120]}"

            except requests.exceptions.ConnectionError:
                err_msg = f"Connection refused at {self.target_url}. Ensure local model server is running."
                self.last_error = err_msg
                time.sleep(2.0)
            except Exception as e:
                err_msg = str(e)[:120]
                self.last_error = err_msg
                time.sleep(1.0)

            dur_ms = round((time.time() - t0) * 1000.0, 1)

            # Record stats safely
            with self._lock:
                self.total_requests += 1
                if status_ok:
                    if do_image:
                        self.image_requests += 1
                    else:
                        self.text_requests += 1
                    self.total_prompt_tokens += tokens_in
                    self.total_completion_tokens += tokens_out
                    self._recent_latencies.append(dur_ms)
                    self._tokens_window.append((time.time(), tokens_out))
                else:
                    self.failed_requests += 1

                log_entry = {
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "worker": f"W{worker_id}",
                    "type": req_type,
                    "prompt": prompt_preview,
                    "tokens": tokens_out,
                    "latency_ms": dur_ms,
                    "success": status_ok,
                    "error": err_msg,
                }
                self._recent_logs.append(log_entry)

    def _sampler_loop(self):
        """Background thread updating telemetry rates and rolling stats every 1 second."""
        while not self._stop_event.is_set():
            time.sleep(1.0)
            with self._lock:
                now = time.time()
                # Compute token generation rate over last 10 seconds
                cutoff = now - 10.0
                recent_tokens = [tok for t, tok in self._tokens_window if t >= cutoff]
                span = max(1.0, min(10.0, (now - self.start_time) if self.start_time else 1.0))
                self.tokens_per_second = round(sum(recent_tokens) / span, 1)

                if self.start_time:
                    total_dur = max(1.0, now - self.start_time)
                    self.requests_per_second = round(self.total_requests / total_dur, 2)

                if self._recent_latencies:
                    self.avg_latency_ms = round(sum(self._recent_latencies) / len(self._recent_latencies), 1)

            # Sample GPU hardware stats
            self.telemetry.sample(
                is_workload_active=self.is_running,
                active_workers=self.concurrency if self.is_running else 0,
                tokens_per_sec=self.tokens_per_second if self.is_running else 0.0,
            )

    def start(self, concurrency: int = 2, include_images: bool = True) -> dict[str, Any]:
        """Start the continuous synthetic request loop."""
        with self._lock:
            if self.is_running:
                return self.get_status()

            self.concurrency = max(1, min(8, int(concurrency)))
            self.include_images = bool(include_images)
            self._stop_event.clear()
            self.start_time = time.time()
            self.is_running = True
            self.last_error = None
            self.total_requests = 0
            self.text_requests = 0
            self.image_requests = 0
            self.failed_requests = 0
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self._recent_latencies.clear()
            self._recent_logs.clear()
            self._tokens_window.clear()

        # Discover model before launching workers
        self.discover_target_model()

        self._workers = []
        for i in range(1, self.concurrency + 1):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True, name=f"gpu-workload-{i}")
            self._workers.append(t)
            t.start()

        if self._sampler_thread is None or not self._sampler_thread.is_alive():
            self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True, name="gpu-sampler")
            self._sampler_thread.start()

        logger.info(f"Continuous GPU synthetic workload started: {self.concurrency} workers, model={self.target_model}")
        return self.get_status()

    def stop(self) -> dict[str, Any]:
        """Instantly stop all synthetic request loops."""
        with self._lock:
            if not self.is_running:
                return self.get_status()
            self.is_running = False
            self._stop_event.set()

        # Join workers with short timeout
        for t in self._workers:
            t.join(timeout=0.2)
        self._workers.clear()

        # Final telemetry sample to reflect idle state
        self.telemetry.sample(is_workload_active=False, active_workers=0, tokens_per_sec=0.0)

        logger.info("Continuous GPU synthetic workload stopped.")
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        """Return snapshot of workload state and throughput metrics."""
        with self._lock:
            uptime = int(time.time() - self.start_time) if (self.is_running and self.start_time) else 0
            return {
                "running": self.is_running,
                "concurrency": self.concurrency,
                "include_images": self.include_images,
                "uptime_seconds": uptime,
                "target_model": self.target_model,
                "target_url": self.target_url,
                "multimodal_supported": self.multimodal_supported,
                "total_requests": self.total_requests,
                "text_requests": self.text_requests,
                "image_requests": self.image_requests,
                "failed_requests": self.failed_requests,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "completion_tokens": self.total_completion_tokens,
                "tokens_per_second": self.tokens_per_second,
                "requests_per_second": self.requests_per_second,
                "avg_latency_ms": self.avg_latency_ms,
                "last_error": self.last_error,
            }

    def get_recent_logs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._recent_logs))


# ── Global Singleton & Flask Blueprint ────────────────────────────────────────
_telemetry = GpuTelemetry(history_len=60)
_runner = SyntheticWorkloadRunner(_telemetry)

# Warm up initial telemetry reading
_telemetry.sample(is_workload_active=False, active_workers=0, tokens_per_sec=0.0)


def blueprint() -> Blueprint:
    """Blueprint for IndiaAI GPU demonstration endpoints."""
    bp = Blueprint("gpu_benchmark", __name__)

    @bp.route("/api/admin/gpu/status", methods=["GET"])
    def get_gpu_status():
        """Combined status: runner stats, live telemetry, and rolling history."""
        # Ensure fresh telemetry reading
        status = _runner.get_status()
        telem = _telemetry.sample(
            is_workload_active=status["running"],
            active_workers=status["concurrency"] if status["running"] else 0,
            tokens_per_sec=status["tokens_per_second"],
        )
        return jsonify({
            "success": True,
            "workload": status,
            "telemetry": telem,
            "history": _telemetry.get_history(),
            "recent_logs": _runner.get_recent_logs(),
        })

    @bp.route("/api/admin/gpu/start", methods=["POST"])
    def start_workload():
        """Start continuous loop of synthetic text & image requests (ONE button)."""
        body = request.get_json(silent=True) or {}
        concurrency = body.get("concurrency", 2)
        include_images = body.get("include_images", True)
        status = _runner.start(concurrency=concurrency, include_images=include_images)
        return jsonify({
            "success": True,
            "message": "Continuous GPU synthetic workload started.",
            "workload": status,
            "telemetry": _telemetry.get_latest(),
        })

    @bp.route("/api/admin/gpu/stop", methods=["POST"])
    def stop_workload():
        """Stop synthetic loop instantly (ONE button)."""
        status = _runner.stop()
        return jsonify({
            "success": True,
            "message": "Continuous GPU synthetic workload stopped.",
            "workload": status,
            "telemetry": _telemetry.get_latest(),
        })

    return bp
