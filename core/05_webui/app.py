"""
RAG Pipeline Web UI
Simple Flask-based interface for document question-answering
"""

# Windows OpenMP and Torchvision fix
import sys
import os
sys.modules['torchvision'] = None
sys.modules['torchvision.transforms'] = None
sys.modules['torchvision.io'] = None
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from importlib import import_module
import os
import sys
import importlib.util
import json
import time
import threading
import queue
import re
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from datetime import datetime
from pathlib import Path

for stream_name in ('stdout', 'stderr'):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')



# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # = core/
REPO_ROOT = PROJECT_ROOT.parent   # = repo root
for _p in [str(REPO_ROOT), str(PROJECT_ROOT), str(PROJECT_ROOT / '04_embeddings_and_kg' / 'scripts')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pipeline_config as pc
import query_debug as qd

# Dynamically find RAG pipeline module - works for both local and Docker
def _find_rag_module():
    """Find and import RAG pipeline module from various possible locations."""
    # Reuse an already-imported instance (e.g. loaded by the document-indexing
    # path) so there is exactly ONE rag_pipeline module and ONE Qdrant client in
    # the process. Re-executing it would open a second client on the same local
    # store and fail with "already accessed by another instance".
    existing = sys.modules.get('rag_pipeline')
    if existing is not None and hasattr(existing, 'retrieve_context'):
        return existing

    _candidate_paths = [
        PROJECT_ROOT / '04_embeddings_and_kg' / 'scripts' / 'rag_pipeline.py',
    ]

    for path in _candidate_paths:
        path = Path(path)
        if path.exists():
            print(f"[Web UI] Found RAG module at: {path}")
            spec = importlib.util.spec_from_file_location("rag_pipeline", str(path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["rag_pipeline"] = module
                spec.loader.exec_module(module)
                return module
    
    raise ImportError("rag_pipeline.py not found in expected locations")


def _build_initialize_pipeline(rag_module):
    """Create a compatibility initializer when rag_pipeline.py does not expose one."""
    def _initialize_pipeline():
        status = {
            'initialized': False,
            'qdrant_connected': False,
            'collection_exists': False,
            'embeddings_loaded': False,
            'graph_connected': False,
            'error': None,
        }

        # 1. Check Qdrant collection
        client = getattr(rag_module, 'client', None)
        ensure_qdrant_client = getattr(rag_module, 'ensure_qdrant_client', None)
        collection_name = getattr(rag_module, 'COLLECTION_NAME', None)
        if collection_name is None and hasattr(rag_module, 'CFG'):
            collection_name = rag_module.CFG.get('collection')

        try:
            if client is None and callable(ensure_qdrant_client):
                client = ensure_qdrant_client()

            if client is not None:
                client.get_collections()
                status['qdrant_connected'] = True
                if collection_name and client.collection_exists(collection_name):
                    status['collection_exists'] = True
        except Exception as e:
            print(f"[Web UI] Qdrant status notice: {e}")

        # 2. Check Temporal Graph Knowledge Store
        try:
            import graph_store
            gs = graph_store.get_graph_store()
            stats = gs.get_stats()
            status['graph_connected'] = True
            status['graph_nodes'] = stats.get('total_nodes', 0)
            status['graph_edges'] = stats.get('total_edges', 0)
            status['graph_chunks'] = stats.get('indexed_chunks', 0)
        except Exception as e:
            print(f"[Web UI] Graph store notice: {e}")

        # If either Qdrant collection exists or Temporal Graph is ready, mark pipeline initialized
        if status.get('collection_exists') or status.get('graph_connected'):
            status['embeddings_loaded'] = True
            status['initialized'] = True
            status['collection_exists'] = True  # Satisfies frontend app.js check
            return status

        status['error'] = 'Neither vector collection nor temporal graph database could be verified'
        return status

    return _initialize_pipeline


def _build_get_db_status(rag_module):
    """Create a compatibility DB status helper when rag_pipeline.py does not expose one."""
    def _get_db_status():
        status = {
            'db_connected': False,
            'collection_exists': False,
            'collection_name': None,
            'points_count': 0,
            'error': None,
        }

        client = getattr(rag_module, 'client', None)
        ensure_qdrant_client = getattr(rag_module, 'ensure_qdrant_client', None)
        collection_name = getattr(rag_module, 'COLLECTION_NAME', None)
        if collection_name is None and hasattr(rag_module, 'CFG'):
            collection_name = rag_module.CFG.get('collection')
        status['collection_name'] = collection_name

        try:
            if client is None and callable(ensure_qdrant_client):
                client = ensure_qdrant_client()

            if client is None:
                status['error'] = 'RAG client is not initialized'
                return status

            client.get_collections()
            status['db_connected'] = True
        except Exception as e:
            status['error'] = f'Cannot connect to Qdrant: {e}'
            return status

        try:
            if collection_name and client.collection_exists(collection_name):
                status['collection_exists'] = True
                collection_info = client.get_collection(collection_name)
                status['points_count'] = collection_info.points_count
            else:
                status['error'] = f"Collection '{collection_name}' not found" if collection_name else 'Collection name unavailable'
        except Exception as e:
            status['error'] = f'Collection check failed: {e}'

        return status

    return _get_db_status


_rag_module = None
retrieve_context = None
generate_answer = None
initialize_pipeline = None
get_db_status = None
RAG_AVAILABLE = False
_rag_import_error = None


def _load_rag_module():
    """Import the heavy RAG module on demand so the Flask server can start first."""
    global _rag_module, retrieve_context, generate_answer
    global initialize_pipeline, get_db_status, RAG_AVAILABLE, _rag_import_error

    if _rag_module is not None:
        return _rag_module

    try:
        _rag_module = _find_rag_module()

        retrieve_context = _rag_module.retrieve_context
        generate_answer = _rag_module.generate_answer
        initialize_pipeline = getattr(_rag_module, 'initialize_pipeline', None) or _build_initialize_pipeline(_rag_module)
        get_db_status = getattr(_rag_module, 'get_db_status', None) or _build_get_db_status(_rag_module)
        RAG_AVAILABLE = True
        _rag_import_error = None
        return _rag_module
    except Exception as e:
        print(f"Warning: Could not import RAG pipeline: {e}")
        _rag_module = None
        retrieve_context = None
        generate_answer = None
        initialize_pipeline = None
        get_db_status = None
        RAG_AVAILABLE = False
        _rag_import_error = str(e)
        return None

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max request body

# Document verification workspace (review -> verify -> gated indexing). Documents come
# from the ingest run; the panel never OCRs (no upload, no re-OCR) — see doc_pipeline.
import doc_registry
import doc_pipeline

def _bg(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()

# Store settings
pipeline_initialized = False
qdrant_retry_attempted = False
num_results = 8          # passages per answer (the evidence token budget decides what fits)
MAX_NUM_RESULTS = 20


def _store_diagnostics():
    """Vector-store state for health payloads: mode, target, and who owns it.

    A contended embedded store used to be invisible from outside — /api/init just
    returned an opaque error and the UI sat on "Preparing…" forever. Reporting the
    holder here turns that into a one-glance diagnosis. Non-invasive: it never
    opens the store, so calling a health endpoint cannot itself take the lock.
    """
    if _rag_module is None:
        return {'error': 'rag module not loaded'}
    try:
        store_status = getattr(_rag_module, 'store_status', None)
        if callable(store_status):
            return store_status()
        return {'error': 'store diagnostics unavailable'}
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def _is_qdrant_connection_failure(error_text):
    """Return True when the error likely indicates Qdrant connectivity failure."""
    if not error_text:
        return False

    msg = str(error_text).lower()
    keywords = [
        "qdrant connection failed",
        "cannot connect to qdrant",
        "connection refused",
        "failed to connect",
        # Contention on the embedded store. Worth exactly one retry: the only benign
        # cause is a backend restart whose predecessor is still letting go, and that
        # clears in well under a second. A persistent holder is reported, not retried
        # forever — qdrant_store has already waited out the lock budget by this point.
        "is in use by",
        "already accessed by another instance",
    ]
    return any(k in msg for k in keywords)


def _retry_initialize_once_on_qdrant_failure(error_text):
    """Attempt a single re-initialization after a Qdrant connection failure."""
    global qdrant_retry_attempted, pipeline_initialized

    if qdrant_retry_attempted:
        return None
    if not _is_qdrant_connection_failure(error_text):
        return None
    if not RAG_AVAILABLE or initialize_pipeline is None:
        return None

    qdrant_retry_attempted = True
    print("[Web UI] Qdrant connection failed. Retrying pipeline initialization once...")

    retry_result = initialize_pipeline()
    if retry_result.get('initialized'):
        pipeline_initialized = True
        print("[Web UI] One-time retry succeeded.")
    else:
        print(f"[Web UI] One-time retry failed: {retry_result.get('error')}")

    return retry_result

@app.route('/api/health', methods=['GET'])
def health():
    """Check system health"""
    return jsonify({
        'status': 'ok',
        'rag_pipeline': 'available',
        'rag_module_loaded': _rag_module is not None,
        'pipeline_initialized': pipeline_initialized,
        'store': _store_diagnostics(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/init', methods=['POST'])
def init():
    """Initialize RAG pipeline and verify database connection"""
    global pipeline_initialized
    
    if _load_rag_module() is None or initialize_pipeline is None:
        return jsonify({
            'success': False,
            'error': _rag_import_error or 'RAG pipeline not available. Check imports and configuration.',
            'details': {}
        }), 503
    
    try:
        print("\n[Web UI] Initializing RAG pipeline...")
        init_result = initialize_pipeline()
        
        if init_result.get('initialized'):
            pipeline_initialized = True
            print("[Web UI] RAG pipeline initialization successful")
            return jsonify({
                'success': True,
                'message': 'RAG pipeline initialized successfully',
                'details': init_result,
            }), 200

        retry_result = _retry_initialize_once_on_qdrant_failure(init_result.get('error'))
        if retry_result and retry_result.get('initialized'):
            return jsonify({
                'success': True,
                'message': 'RAG pipeline initialized successfully after one retry',
                'details': retry_result,
                'retried_once': True,
            }), 200

        else:
            print(f"[Web UI] RAG pipeline initialization failed: {init_result.get('error')}")
            return jsonify({
                'success': False,
                'error': init_result.get('error', 'Initialization failed'),
                'details': init_result,
                'retried_once': retry_result is not None,
                'retry_details': retry_result,
                # So the caller can say WHY it cannot start (e.g. "store held by
                # pid 8123") instead of leaving the UI on an unexplained spinner.
                'store': _store_diagnostics(),
            }), 400
            
    except Exception as e:
        print(f"[Web UI] Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Initialization error: {str(e)}',
            'details': {}
        }), 500

@app.route('/api/db-status', methods=['GET'])
def db_status():
    """Check database connection and collection status"""
    
    if _rag_module is None:
        return jsonify({
            'success': True,
            'error': 'RAG module has not been loaded yet',
            'db_connected': False,
            'collection_exists': False,
            'collection_name': None,
            'points_count': 0
        }), 200
    
    try:
        status = get_db_status()

        if not status.get('db_connected', False):
            retry_result = _retry_initialize_once_on_qdrant_failure(status.get('error'))
            if retry_result and retry_result.get('initialized'):
                status = get_db_status()

        return jsonify({
            'success': True,
            'db_connected': status.get('db_connected', False),
            'collection_exists': status.get('collection_exists', False),
            'collection_name': status.get('collection_name'),
            'points_count': status.get('points_count', 0),
            'error': status.get('error'),
            'retried_once': qdrant_retry_attempted,
            'store': _store_diagnostics(),
        }), 200
        
    except Exception as e:
        print(f"[Web UI] Error checking DB status: {e}")
        return jsonify({
            'success': False,
            'error': f'DB status check failed: {str(e)}',
            'db_connected': False,
            'collection_exists': False
        }), 500

# ─── Temporal Knowledge Graph & Retriever Experiment Routes ───────────────────
import graph_store
import retrievers

@app.route('/api/docs/refresh', methods=['GET', 'POST'])
def refresh_docs():
    """Trigger automated system documentation regeneration."""
    try:
        import subprocess
        docs_script = REPO_ROOT / 'scripts' / 'update_docs.py'
        res = subprocess.run([sys.executable, str(docs_script)], capture_output=True, text=True, check=True)
        return jsonify({
            'success': True,
            'message': 'Documentation updated successfully.',
            'output': res.stdout.strip()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/graph/stats', methods=['GET'])
def graph_stats():
    """Return high-level statistics of the GAD temporal knowledge graph."""
    try:
        store = graph_store.get_graph_store()
        return jsonify({
            'success': True,
            'stats': store.get_stats()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph/categories', methods=['GET'])
def graph_categories():
    """Return high-level governance domain categories with document counts."""
    try:
        store = graph_store.get_graph_store()
        return jsonify({
            'success': True,
            'categories': store.get_categories()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph/hierarchical', methods=['GET'])
def graph_hierarchical():
    """Return static, hierarchical hub-and-spoke graph layout (Macro Categories -> Micro Radial Documents)."""
    try:
        expanded = request.args.get('expanded')
        limit = request.args.get('limit', default=120, type=int)
        expanded_list = [x.strip() for x in expanded.split(',') if x.strip()] if expanded else []
        store = graph_store.get_graph_store()
        data = store.get_hierarchical_graph(
            expanded_categories=expanded_list,
            max_children_per_cat=limit
        )
        return jsonify({'success': True, **data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph/subgraph', methods=['GET'])
def graph_subgraph():
    """Return nodes and edges formatted for visual rendering (Vis.js / Cytoscape)."""
    try:
        center = request.args.get('center')
        query = request.args.get('query')
        year = request.args.get('year', type=int)
        limit = request.args.get('limit', default=100, type=int)
        store = graph_store.get_graph_store()
        subgraph = store.get_subgraph_for_visualization(
            center_node_id=center,
            query=query,
            year_filter=year,
            max_nodes=limit
        )
        return jsonify({'success': True, **subgraph})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph/history/<doc_id>', methods=['GET'])
def graph_history(doc_id):
    """Traverse document history across time (predecessors, successors, lineage)."""
    try:
        store = graph_store.get_graph_store()
        history = store.traverse_history(doc_id)
        if 'error' in history:
            return jsonify({'success': False, 'error': history['error']}), 404
        return jsonify({'success': True, **history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph/document/<doc_id>', methods=['GET'])
def graph_document_full(doc_id):
    """Return the entire full OCRed document text for a given document node."""
    try:
        store = graph_store.get_graph_store()
        full_text = store.get_document_full_ocr(doc_id)
        if not full_text:
            return jsonify({'success': False, 'error': f'Document {doc_id} OCR content not found'}), 404
        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'full_ocr_text': full_text,
            'character_count': len(full_text),
            'line_count': len(full_text.splitlines())
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph/search', methods=['GET'])
def graph_search():
    """Search nodes by order number, title, or keywords."""
    try:
        q = request.args.get('q', '').strip()
        limit = request.args.get('limit', default=20, type=int)
        if not q:
            return jsonify({'success': True, 'results': []})
        store = graph_store.get_graph_store()
        results = store.search_nodes(q, limit=limit)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph/semantic_search', methods=['GET'])
def graph_semantic_search():
    """Intelligent semantic, category-routed, and keyword/number/date search across documents."""
    try:
        q = request.args.get('q', '').strip()
        limit = request.args.get('limit', default=50, type=int)
        if not q:
            return jsonify({
                'success': True,
                'query': '',
                'matched_category': None,
                'matched_node_ids': [],
                'matched_hub_ids': [],
                'results': []
            })
        store = graph_store.get_graph_store()
        search_data = store.semantic_search(q, limit=limit)
        return jsonify({'success': True, **search_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/officers', methods=['GET'], strict_slashes=False)
@app.route('/api/officers/', methods=['GET'], strict_slashes=False)
def list_officers():
    """List or search officers from the officer store."""
    try:
        from gad_kg.src.officer_store import OfficerStore
        store = OfficerStore()
        q = request.args.get('q', '').strip()
        category = request.args.get('category', '').strip()
        limit = request.args.get('limit', default=50, type=int)

        if q:
            results = store.search_officers(q, limit=limit)
        elif category:
            conn = store._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT * FROM officers WHERE category = ? LIMIT ?", (category, limit))
            results = [dict(r) for r in cur.fetchall()]
            conn.close()
        else:
            conn = store._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT * FROM officers LIMIT ?", (limit,))
            results = [dict(r) for r in cur.fetchall()]
            conn.close()
        return jsonify({'success': True, 'count': len(results), 'officers': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/officer/<officer_id>', methods=['GET'], strict_slashes=False)
def get_officer_detail(officer_id):
    """Get full officer profile and linked notifications."""
    try:
        from gad_kg.src.officer_store import OfficerStore
        store = OfficerStore()
        off = store.get_officer(officer_id)
        if not off:
            return jsonify({'success': False, 'error': 'Officer not found'}), 404
        return jsonify({'success': True, 'officer': off})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/officers/secretaries', methods=['GET'], strict_slashes=False)
def get_secretaries_list():
    """Get current Secretaries, ACS, and Principal Secretaries of Govt of CG."""
    try:
        from gad_kg.src.officer_store import OfficerStore
        store = OfficerStore()
        secs = store.get_secretaries()
        return jsonify({'success': True, 'count': len(secs), 'secretaries': secs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/officers/retired_ias', methods=['GET'], strict_slashes=False)
def get_retired_ias_list():
    """Get official list of retired IAS officers from superannuation circulars."""
    try:
        from gad_kg.src.officer_store import OfficerStore
        store = OfficerStore()
        ret = store.get_retired_ias()
        return jsonify({'success': True, 'count': len(ret), 'retired_ias': ret})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/officers/chips_ceos', methods=['GET'], strict_slashes=False)
def get_chips_ceos_list():
    """Get CHiPS CEOs and leadership history."""
    try:
        from gad_kg.src.officer_store import OfficerStore
        store = OfficerStore()
        ceos = store.get_chips_ceos()
        return jsonify({'success': True, 'count': len(ceos), 'chips_ceos': ceos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/officers/ias_establishment', methods=['GET'], strict_slashes=False)
def get_ias_establishment_details():
    """Get IAS establishment section details and work allocation."""
    try:
        from gad_kg.src.officer_store import OfficerStore
        store = OfficerStore()
        est = store.get_ias_establishment()
        return jsonify({'success': True, 'ias_establishment': est})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/retriever/experiment', methods=['POST'])
def retriever_experiment():
    """Run comparative experiment across Lexical and Temporal Graph retrievers."""
    try:
        data = request.get_json(silent=True) or {}
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 5))
        if not query:
            return jsonify({'success': False, 'error': 'Query text is required'}), 400

        harness = retrievers.RetrieverExperimentHarness()
        exp_result = harness.run_experiment(query, top_k=top_k)

        # If LLM generation is requested, also generate a grounded answer using our vLLM endpoint
        include_answer = data.get('generate_answer', True)
        answer_text = None
        if include_answer and exp_result['temporal_graph']['results']:
            try:
                import llm_client
                llm = llm_client.client()
                
                # Format context with temporal warnings
                top_chunks = exp_result['temporal_graph']['results'][:3]
                context_parts = []
                for c in top_chunks:
                    status_note = f"[STATUS: {c.get('effective_status', '').upper()}]"
                    if c.get('superseded_by'):
                        status_note += f" (SUPERSEDED BY: {c.get('superseded_by')})"
                    context_parts.append(
                        f"Order No: {c.get('order_no', 'N/A')} | Date: {c.get('doc_date', 'N/A')} {status_note}\n"
                        f"{c.get('content', '')}"
                    )
                context_str = "\n\n---\n\n".join(context_parts)

                warnings_str = ""
                if exp_result['temporal_graph']['warnings']:
                    warnings_str = "IMPORTANT TEMPORAL NOTES:\n" + "\n".join(f"- {w}" for w in exp_result['temporal_graph']['warnings']) + "\n\n"

                system_prompt = (
                    "You are the GAD Governance Assistant for Chhattisgarh General Administration Department. "
                    "Answer the question accurately based on the provided government orders. "
                    "Pay special attention to whether an order is active, amended, or superseded. "
                    "If an older order is superseded, explicitly warn the user and cite the latest standing order."
                )
                user_prompt = f"{warnings_str}EVIDENCE FROM GAD RECORDS:\n{context_str}\n\nQUESTION: {query}\n\nANSWER:"

                ans, _ = llm.chat(system_prompt, user_prompt, max_tokens=600)
                answer_text = ans
            except Exception as e:
                print(f"[Web UI] LLM answer generation failed: {e}")
                answer_text = f"(Answer generation fallback: {e})"

        exp_result['llm_answer'] = answer_text
        return jsonify({'success': True, 'experiment': exp_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def _format_results(context_results, query_text):
    """Shape retrieved chunks for the frontend (PDF name + highlighted excerpt).

    Shared by the single-shot path and thinking mode so both render identically in
    the sources drawer.
    """
    formatted = []
    query_words = [w for w in query_text.lower().split() if len(w) > 3]
    excerpt_fn = getattr(_rag_module, 'extract_highlighted_excerpt', None)

    for result in context_results:
        point = result.get('point', {})
        payload = getattr(point, 'payload', None) or {}
        source = payload.get('source', '')
        text = payload.get('text', '')

        actual_pdf = f"{source}.pdf"
        # Readable label + the ORIGINAL source URL. The documents are scraped web
        # pages with no local PDF on disk, so the drawer opens the live source.
        label, source_url = actual_pdf, None
        if _rag_module and hasattr(_rag_module, 'source_label'):
            try:
                label, source_url = _rag_module.source_label(source)
            except Exception:
                pass
        correction = payload.get('verified_correction')
        if correction:
            label, source_url = correction['sourceTitle'], correction['sourceUrl']
        if excerpt_fn:
            excerpt = excerpt_fn(text, query_words, max_length=250)
        else:
            excerpt = text[:250] + "..." if len(text) > 250 else text

        formatted.append({
            'rank': result.get('rank', 0),
            'source': source,
            'actual_pdf': actual_pdf,
            'label': label,
            'source_url': source_url,
            'score': result.get('score', 0),
            'text': text,
            'excerpt': excerpt,
            'parent_id': result.get('parent_id', ''),
            # Thinking mode pulls in chunks adjacent to a hit for continuity; label
            # them so the drawer can show they are supporting rather than matched.
            'supporting': 'neighbor_of' in result,
        })
    return formatted


# Thinking mode lives next to rag_pipeline and is imported lazily, so a failure to
# load it (experimental module) degrades to normal query instead of 503-ing the app.
_thinking_module = None
_thinking_import_error = None


def _load_thinking_module():
    global _thinking_module, _thinking_import_error
    if _thinking_module is not None or _thinking_import_error is not None:
        return _thinking_module
    try:
        if _load_rag_module() is None:
            raise RuntimeError('RAG pipeline unavailable')
        import thinking_mode                      # same dir as rag_pipeline (already on sys.path)
        _thinking_module = thinking_mode
    except Exception as e:
        _thinking_import_error = f'{type(e).__name__}: {e}'
        print(f'[Web UI] thinking mode unavailable: {_thinking_import_error}')
    return _thinking_module


_GREETING = (r'hi+|hello+|hey+|good\s+(?:morning|afternoon|evening)|namaste|namaskar|pranam|'
             r'नमस्ते|नमस्कार|प्रणाम|हेलो|हैलो|हाय')
# Small talk is answered here, never by the answer model: its prompt has no small-talk
# rule (one with a reply to copy turned "what is gad" into a greeting, 2026-10-15).
_THANKS = (r'thanks?|thank\s+you|thx|dhanyavaa?d|dhanyawaa?d|shukriya|'
           r'धन्यवाद|शुक्रिया|आभार')
_HOW_ARE_YOU = r'how\s+are\s+you|(?:aap\s+)?kaise\s+(?:ho|hain|hai)|(?:आप\s+)?कैसे\s+(?:हो|हैं|है)'
_BYE = r'bye|good\s*bye|good\s+night|alvida|अलविदा'
_FILLER = r'ok|okay|ji|sir|madam|mam|जी'
_SMALL_TALK = rf'{_GREETING}|{_THANKS}|{_HOW_ARE_YOU}|{_BYE}'
_ONLY_GREETING = re.compile(rf'(?:(?:{_SMALL_TALK}|{_FILLER})(?:\s+|$))+', re.I)
_EDGE_L, _EDGE_R = r'(?<![\w\u0900-\u097f])', r'(?![\w\u0900-\u097f])'
_SMALL_TALK_KINDS = [(kind, re.compile(_EDGE_L + rf'(?:{rx})' + _EDGE_R, re.I)) for kind, rx in
                     (('thanks', _THANKS), ('bye', _BYE), ('how', _HOW_ARE_YOU), ('hello', _GREETING))]
_SMALL_TALK_REPLY = {   # (English, Hindi); the first kind found in the message decides
    'thanks': ("You're welcome! Ask me anything about your department's documents.",
               'आपका स्वागत है! विभाग के दस्तावेज़ों से जुड़ा कोई भी सवाल पूछें।'),
    'bye': ("Goodbye! Come back any time with a question about your department's documents.",
            'धन्यवाद! जब भी ज़रूरत हो, विभाग के दस्तावेज़ों के बारे में पूछें।'),
    'how': ("I'm here and ready to help. What would you like to know from your department's documents?",
            'मैं आपकी मदद के लिए तैयार हूँ। आप विभाग के दस्तावेज़ों के बारे में क्या जानना चाहते हैं?'),
    'hello': ("Hello! How can I help you with your department's documents?",
              'नमस्ते! मैं आपके विभाग के दस्तावेज़ों से जुड़े सवालों में आपकी कैसे मदद कर सकता हूँ?'),
}


def _greeting_result(text, thinking=False):
    """Reply only to a whole greeting or small talk ('thanks', 'how are you', 'bye');
    'hi, what is the pension rule?' still searches."""
    normalized = re.sub(r'[^\w\s\u0900-\u097f]', ' ', text).strip()
    if not _ONLY_GREETING.fullmatch(normalized):
        return None
    kind = next((k for k, rx in _SMALL_TALK_KINDS if rx.search(normalized)), None)
    if kind is None:                 # fillers alone ("ok", "ji"): not small talk to answer
        return None
    hindi = bool(re.search(r'[\u0900-\u097f]', normalized))
    answer = _SMALL_TALK_REPLY[kind][1 if hindi else 0]
    return {'success': True, 'query': text, 'answer': answer, 'results': [], 'result_count': 0,
            'thinking': bool(thinking), 'trace': [], 'hops': 0, 'timeline': [],
            'standalone': None, 'execution_time': '0.00s'}


@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json(silent=True) or {}
    debug = _new_query_debug(data)
    with qd.capture(debug):
        response = app.make_response(_query_response())
        body = response.get_json(silent=True)
        if debug and isinstance(body, dict):
            _finish_query_debug(debug, body)
            response.set_data(app.json.dumps(body))
        return response


def _debug_owner():
    # Express strips browser-supplied identity headers and inserts its verified
    # session role. Flask is the existing internal, loopback-only backend.
    owner = request.headers.get('X-Gad-User', '')
    if (request.remote_addr in ('127.0.0.1', '::1')
            and request.headers.get('X-Gad-Role') == 'admin'
            and re.fullmatch(r'[a-zA-Z0-9._-]{1,64}', owner)):
        return owner
    return None


def _new_query_debug(data):
    owner = _debug_owner()
    if not owner:
        return None
    return qd.Recorder(pc.DATA_ROOT, owner, pc.PIPELINE, {
        'query': data.get('query'), 'thinking': bool(data.get('thinking')),
        'history_received': data.get('history') or [], 'num_context': _num_context(data)})


def _finish_query_debug(debug, body):
    if debug:
        debug.add('Response delivered to chat', {
            'answer': body.get('answer'), 'error': body.get('error'),
            'success': body.get('success', body.get('type') != 'error'),
            'sources_shown': body.get('results', []), 'standalone': body.get('standalone')})
        trace_id = debug.save()
        if trace_id:
            body['debugId'] = trace_id
        else:
            body['debugError'] = 'Processing details could not be saved on the server.'


@app.route('/api/admin/query-debug/<trace_id>', methods=['GET'])
def query_debug_detail(trace_id):
    owner = _debug_owner()
    if not owner:
        return jsonify({'success': False, 'error': 'Admin access required.'}), 403
    try:
        file = qd.trace_path(pc.DATA_ROOT, owner, trace_id)
        data = json.loads(file.read_text(encoding='utf-8'))
        return jsonify({'success': True, 'debug': data}), 200, {'Cache-Control': 'no-store'}
    except (ValueError, FileNotFoundError):
        return jsonify({'success': False, 'error': 'Processing details not found for this account.'}), 404


def _query_response():
    """Process a query.

    Body: {query, num_results?, thinking?, history?}
    `thinking: true` runs the multi-hop orchestrator and additionally returns a
    `trace` (what it searched, what it found, how precedence was decided).
    `history`: the chat's previous turns [{question, answer}] — a follow-up is
    rewritten into a standalone question for the search, returned as `standalone`.
    """
    data = request.get_json(silent=True) or {}
    query_text = str(data.get('query') or '').strip()

    if not query_text:
        return jsonify({
            'success': False,
            'error': 'Query cannot be empty',
            'query': ''
        }), 400

    greeting = _greeting_result(query_text, data.get('thinking'))
    if greeting is not None:
        qd.record('Greeting shortcut', {'answer': greeting['answer'], 'retrieval': 'skipped', 'llm': 'skipped'})
        return jsonify(greeting)

    # ── GAD-KG Master Spec Pipeline Integration (Phases 0-15) ──────────────
    try:
        from gad_kg_adapter import handle_gad_query
        gad_resp = handle_gad_query(
            query_text,
            thinking=bool(data.get('thinking')),
            history=data.get('history'),
            num_context=_num_context(data)
        )
        return jsonify(gad_resp), 200
    except Exception as e:
        print(f"[GAD-KG Adapter Fallback] {e}")

    if _load_rag_module() is None or retrieve_context is None or generate_answer is None:
        return jsonify({'success': False, 'query': query_text,
                        'error': _rag_import_error or 'RAG pipeline not available. Check imports and configuration.'}), 503
    history, search_q = _chat_context(data, query_text)
    body, status, _hits = _query_result(query_text, history, search_q,
                                        bool(data.get('thinking')), _num_context(data))
    return jsonify(body), status


def _query_result(query_text, history, search_q, thinking, num_context):
    """(response body, HTTP status, retrieved hits) for one question — the chat's
    blocking path (/api/query) and the admins' batch test runs (batch_runs.py), so
    a test run answers exactly as the chat does."""
    greeting = _greeting_result(query_text, thinking)
    if greeting is not None:
        return greeting, 200, []

    # ── GAD-KG Master Spec Pipeline Integration (Phases 0-15) ──────────────
    try:
        from gad_kg_adapter import handle_gad_query
        gad_resp = handle_gad_query(
            query_text,
            thinking=bool(thinking),
            history=history,
            num_context=num_context
        )
        if gad_resp.get('success'):
            return gad_resp, 200, gad_resp.get('results', [])
    except Exception as e:
        print(f"[GAD-KG _query_result Fallback] {e}")

    query_start_time = time.time()
    standalone = search_q if search_q != query_text else None

    # ── Thinking mode: multi-hop retrieve/assess, with a reasoning trace ─────
    if thinking:
        tm = _load_thinking_module()
        if tm is None:
            return {
                'success': False,
                'error': f'Thinking mode unavailable: {_thinking_import_error}',
                'query': query_text, 'results': []
            }, 503, []
        try:
            print(f"\n🧠 [FLASK] Thinking mode: {query_text}")
            out = tm.deep_query(search_q)
            total_time = time.time() - query_start_time
            print(f"⏱️ [FLASK] Thinking mode total {total_time:.2f}s\n")
            return {
                'success': True,
                'query': query_text,
                'answer': out['answer'],
                'results': _format_results(out['results'], query_text),
                'result_count': len(out['results']),
                'thinking': True,
                'trace': out['trace'],
                'hops': out['hops'],
                'timeline': out['timeline'],
                'standalone': standalone,
                'execution_time': f"{total_time:.2f}s",
            }, 200, out['results']
        except Exception as e:
            print(f"[Web UI] Thinking mode error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Thinking mode failed: {e}',
                'query': query_text, 'results': []
            }, 500, []

    try:
        print(f"\n⏱️ [FLASK] Total request start")
        print(f"[Web UI] Processing query: {query_text}")
        
        # Step 1: Retrieve context (parent chunks)
        print(f"[Web UI] Retrieving context...")
        retrieval_start = time.time()
        context_results = retrieve_context(search_q, num_context=num_context)
        retrieval_time = time.time() - retrieval_start
        print(f"⏱️ [FLASK] Retrieval completed in {retrieval_time:.2f}s")
        
        if context_results is None or len(context_results) == 0:
            failed = _search_failed()
            return {
                'success': False,
                'error': failed or 'No context documents found for this query',
                'query': query_text,
                'results': []
            }, (503 if failed else 200), []
        
        # Step 2: Generate answer. The sources panel gets exactly the documents the
        # answer was given, numbered like its [n] markers — none under a "not in the
        # documents" reply, and none under a model failure.
        print(f"[Web UI] Generating answer...")
        answer_start = time.time()
        try:
            answer, display_results = _rag_module.answer_question(
                search_q, context_results, history=history, asked=query_text)
        except _rag_module.LLMError as e:
            return {'success': False, 'error': chat_model.friendly_error(e) or f'The answer model failed: {e}',
                    'query': query_text, 'results': []}, 502, context_results
        answer_time = time.time() - answer_start
        print(f"⏱️ [FLASK] Answer generation completed in {answer_time:.2f}s")
        formatted_results = _format_results(display_results, query_text)

        total_time = time.time() - query_start_time
        print(f"[Web UI] Query processed successfully")
        print(f"⏱️ [FLASK] TOTAL PIPELINE TIME: {total_time:.2f}s\n")

        return {
            'success': True,
            'query': query_text,
            'answer': answer,
            'results': formatted_results,
            'result_count': len(formatted_results),
            'standalone': standalone,
            'execution_time': f"{total_time:.2f}s"
        }, 200, context_results
    
    except Exception as e:
        print(f"[Web UI] Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': f'Error processing query: {str(e)}',
            'query': query_text,
            'results': []
        }, 500, []

def _sse(event):
    """One SSE message. ensure_ascii=False keeps Hindi readable on the wire."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.route('/api/query/stream', methods=['POST'])
def query_stream():
    """Streaming twin of /api/query (Server-Sent Events).

    Emits, as `data: {json}` lines:
      {"type":"status","stage":"retrieving"|"answering",   coarse phase (normal mode);
       "standalone": "..."}                                 the rewritten follow-up, if any
      {"type":"step","step":{...}}                          a completed thinking-mode step
      {"type":"token","text":"..."}                         one answer delta
      {"type":"done", ...}                                  final results/sources/trace
      {"type":"error","error":"..."}

    deep_query / call_llm_stream push into callbacks synchronously, so a worker
    thread runs the work and feeds a queue that this generator drains — the standard
    bridge from a push model to a pull generator.
    """
    data = request.get_json(silent=True) or {}
    query_text = str(data.get('query') or '').strip()
    num_context = _num_context(data)
    thinking = bool(data.get('thinking'))
    debug = _new_query_debug(data)
    if not query_text:
        return jsonify({'success': False, 'error': 'Query cannot be empty'}), 400

    greeting = _greeting_result(query_text, thinking)
    if greeting is not None:
        if debug:
            debug.add('Greeting shortcut', {'retrieval': 'skipped', 'llm': 'skipped'})
            _finish_query_debug(debug, greeting)
        events = _sse({'type': 'token', 'text': greeting['answer']}) + _sse({'type': 'done', **greeting})
        return Response(events, mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache, no-transform', 'X-Accel-Buffering': 'no'})

    # ── GAD-KG Master Spec Pipeline Integration (Phases 0-15) ──────────────
    try:
        from gad_kg_adapter import handle_gad_query_stream
        q = queue.Queue()
        DONE = object()
        def emit(event):
            if event.get('type') in ('done', 'error'):
                _finish_query_debug(debug, event)
            q.put(event)

        def worker():
            with qd.capture(debug):
                try:
                    handle_gad_query_stream(
                        query_text,
                        emit,
                        thinking=thinking,
                        history=data.get('history'),
                        num_context=num_context
                    )
                except Exception as e:
                    print(f"[GAD-KG Stream Worker Error] {e}")
                    import traceback
                    traceback.print_exc()
                    emit({'type': 'error', 'error': str(e)})
                finally:
                    q.put(DONE)

        threading.Thread(target=worker, daemon=True).start()

        def generate():
            yield ": stream open\n\n"
            while True:
                try:
                    item = q.get(timeout=60)
                    if item is DONE:
                        break
                    yield _sse(item)
                except queue.Empty:
                    yield ": keepalive\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache, no-transform', 'X-Accel-Buffering': 'no'})
    except Exception as e:
        print(f"[GAD-KG Stream Fallback] {e}")

    if _load_rag_module() is None or retrieve_context is None:
        body = {'success': False, 'error': _rag_import_error or 'RAG pipeline not available.'}
        _finish_query_debug(debug, body)
        return jsonify(body), 503

    q = queue.Queue()
    DONE = object()
    def emit(event):
        if event.get('type') in ('done', 'error'):
            _finish_query_debug(debug, event)
        q.put(event)

    def worker():
        with qd.capture(debug):
            process_query()

    def process_query():
        t0 = time.time()
        try:
            # A follow-up in the current chat becomes a standalone question for the
            # search (rag_pipeline.standalone_question); only this chat's turns are sent.
            history, search_q = _chat_context(data, query_text)
            standalone = search_q if search_q != query_text else None
            if thinking:
                tm = _load_thinking_module()
                if tm is None:
                    emit({'type': 'error', 'error': f'Thinking mode unavailable: {_thinking_import_error}'})
                    return
                out = tm.deep_query(
                    search_q,
                    on_step=lambda step: emit({'type': 'step', 'step': step}),
                    on_token=lambda text: emit({'type': 'token', 'text': text}),
                )
                emit({'type': 'done', 'success': True, 'thinking': True,
                      'answer': out['answer'],
                      'results': _format_results(out['results'], query_text),
                      'result_count': len(out['results']),
                      'hops': out.get('hops'), 'trace': out.get('trace'),
                      'timeline': out.get('timeline'), 'standalone': standalone,
                      'execution_time': f"{time.time() - t0:.2f}s"})
            else:
                emit({'type': 'status', 'stage': 'retrieving', 'standalone': standalone})
                ctx = retrieve_context(search_q, num_context=num_context)
                if not ctx:
                    emit({'type': 'done', 'success': False,
                          'error': _search_failed() or 'No context documents found for this query',
                          'answer': '', 'results': [], 'result_count': 0})
                    return
                emit({'type': 'status', 'stage': 'answering'})
                rp = _rag_module
                system, user, sources, lang = rp.build_answer_messages(
                    search_q, ctx, history=history, asked=query_text)
                buf = []
                for delta in rp.call_llm_stream(system, user, temperature=rp.ANSWER_TEMPERATURE):
                    buf.append(delta)
                    emit({'type': 'token', 'text': delta})
                # The same finishing step as the blocking path (rag_pipeline.finish_answer):
                # our numbered Sources list as the last token, the panel = the documents
                # the answer was given.
                answer, tail, display = rp.finish_answer(
                    ''.join(buf), user=user, question=search_q, sources=sources, lang=lang,
                    context_results=ctx)
                if tail:
                    emit({'type': 'token', 'text': tail})
                emit({'type': 'done', 'success': True, 'answer': answer,
                      'results': _format_results(display, query_text), 'result_count': len(display),
                      'standalone': standalone, 'execution_time': f"{time.time() - t0:.2f}s"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            emit({'type': 'error', 'error': chat_model.friendly_error(e) or str(e)})
        finally:
            q.put(DONE)

    def gen():
        threading.Thread(target=worker, daemon=True).start()
        # An initial comment flushes headers immediately so the client shows activity.
        yield ": stream open\n\n"
        while True:
            ev = q.get()
            if ev is DONE:
                break
            yield _sse(ev)

    return Response(stream_with_context(gen()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache, no-transform',
                             'X-Accel-Buffering': 'no',
                             'Connection': 'keep-alive'})


def _search_failed():
    """The reader's message when the document search itself failed (the index server
    down, a bug) — not "no documents found", which is what it used to look like."""
    why = getattr(_rag_module, 'last_retrieval_error', lambda: None)()
    if not why:
        return None
    print(f"[Web UI] document search failed: {why}")
    return ('The document search is not working right now, so this question could not be '
            'answered. Please try again in a few minutes; if it keeps happening, tell the administrator.')


def _chat_context(data, query_text):
    """(history, search question) for one request: the chat's recent turns, and the
    question rewritten as a standalone question when it is a follow-up (both from
    rag_pipeline; the question itself when there is no history)."""
    history = _rag_module.clean_history(data.get('history'))
    qd.record('Conversation context', {'history_used': history, 'original_question': query_text})
    search_q = _rag_module.standalone_question(query_text, history)
    qd.record('Search question selected', {'original_question': query_text, 'search_question': search_q,
                                         'rewritten': search_q != query_text})
    return history, search_q


def _num_context(data):
    """Passages to retrieve for one question: the request's num_results, kept to
    1..MAX_NUM_RESULTS (a bad or huge value must not slow every answer down)."""
    try:
        n = int(data.get('num_results', num_results))
    except (TypeError, ValueError):
        n = num_results
    return max(1, min(MAX_NUM_RESULTS, n))


# ── Batch test runs (admins; batch_runs.py, the Batch test panel) ─────────────
def _labels(hits):
    """Readable document names of retrieved hits, in rank order, each once."""
    out = []
    for r in hits or []:
        source = (getattr(r.get('point'), 'payload', None) or {}).get('source', '')
        label = f"{source}.pdf"
        try:
            label = _rag_module.source_label(source)[0]
        except Exception:
            pass
        if label not in out:
            out.append(label)
    return out


def _batch_answer(question, mode):
    """One test question, answered as the chat answers it (no chat history)."""
    greeting = _greeting_result(question, mode == 'thinking')
    if greeting is not None:
        return {'answer': greeting['answer'], 'sources': [], 'retrieved': [], 'model': None}

    # ── GAD-KG Master Spec Pipeline Integration (Phases 0-15) ──────────────
    try:
        from gad_kg_adapter import handle_gad_query
        gad_resp = handle_gad_query(question, thinking=(mode == 'thinking'), num_context=num_results)
        if gad_resp.get('success'):
            answer = gad_resp.get('answer', '')
            results = gad_resp.get('results') or []
            sources = []
            for s in results:
                lbl = s.get('label') or s.get('title')
                url = s.get('source_url') or s.get('url')
                if lbl and not any(x['label'] == lbl for x in sources):
                    sources.append({'label': lbl, 'url': url})
            retrieved = [s.get('label') for s in results if s.get('label')]
            model = 'gad-kg-vllm'
            return {'answer': answer, 'sources': sources, 'retrieved': retrieved, 'model': model}
    except Exception as e:
        print(f"[Batch Answer GAD-KG Error] {e}")

    if _load_rag_module() is None or retrieve_context is None:
        raise RuntimeError(_rag_import_error or 'RAG pipeline not available')
    history, search_q = _chat_context({}, question)
    body, _status, hits = _query_result(question, history, search_q, mode == 'thinking', num_results)
    if not body.get('success'):
        raise RuntimeError(body.get('error') or 'no answer')
    strip = getattr(getattr(_rag_module, 'llm_client', None), 'strip_thinking', None)
    sources = []
    for s in body.get('results') or []:
        if not any(x['label'] == s.get('label') for x in sources):
            sources.append({'label': s.get('label'), 'url': s.get('source_url')})
    try:
        model = _rag_module.llm_client.client().model      # which model answered (runs compare them)
    except Exception:
        model = None
    return {'answer': strip(body['answer']) if strip else body['answer'],
            'sources': sources, 'retrieved': _labels(hits), 'model': model}


import batch_runs
app.register_blueprint(batch_runs.blueprint(batch_runs.Runs(pc.DATA_ROOT / 'batch_tests', _batch_answer)))

import eval_routes
app.register_blueprint(eval_routes.create_eval_blueprint())

# Correction embeddings are needed by every bot, using its own loaded encoder.
import dept_service
app.register_blueprint(dept_service.correction_blueprint())

# Department bots (dept_service.py): reading, embedding and answering for the
# department platform, on the bot that hosts it (PLATFORM_PIPELINE, default chips).
# Internal routes — the Node server calls them with a key and never forwards them.
if pc.PIPELINE == (os.getenv('PLATFORM_PIPELINE') or 'chips').strip().lower():
    app.register_blueprint(dept_service.blueprint())

# The answer model, switched by an admin on the chat page (chat_model.py)
import chat_model
app.register_blueprint(chat_model.blueprint())

# IndiaAI GPU Capacity demonstration & continuous workload runner (Yotta)
import gpu_benchmark
app.register_blueprint(gpu_benchmark.blueprint())


@app.route('/api/settings', methods=['GET'])
def settings():
    """Read-only: settings are per request (num_results), not changed for everyone."""
    return jsonify({'num_results': num_results, 'max_num_results': MAX_NUM_RESULTS}), 200

@app.route('/api/examples', methods=['GET'])
def examples():
    """Return example queries"""
    examples_list = [
        "What approval was given in the recent meeting?",
        "Who leads the committee?",
        "What are the key financial decisions?",
        "Summarize the meeting agenda",
        "What are the next action items?",
        "What entities are mentioned in the documents?",
        "What was discussed about budget allocation?",
        "Tell me about the committee members",
    ]
    
    return jsonify({
        'examples': examples_list
    }), 200

@app.route('/01_preprocessing/used_files/<filename>', methods=['GET'])
def serve_pdf(filename):
    """Serve PDF files from used_files directory"""
    import os
    
    # Security: Only allow PDF files and alphanumeric filenames
    if not filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 403
    
    # Prevent directory traversal attacks
    if '..' in filename or '/' in filename:
        return jsonify({'error': 'Invalid filename'}), 403
    
    pdf_path = pc.paths.used_files / filename

    if not pdf_path.exists():
        return jsonify({'error': f'PDF not found: {filename}'}), 404
    
    try:
        return send_file(str(pdf_path), mimetype='application/pdf')
    except Exception as e:
        print(f"[Web UI] Error serving PDF {filename}: {e}")
        return jsonify({'error': f'Error serving PDF: {str(e)}'}), 500


# ════════════════════════════════════════════════════════════════════
# Document verification workspace
# Upload PDF -> background OCR -> .md -> human verify -> gated RAG indexing.
# Only VERIFIED documents are embedded; editing a verified .md re-indexes it.
# ════════════════════════════════════════════════════════════════════

def _reviewer(source):
    """Extract the reviewer/uploader name from a request body/form."""
    return (source.get('verified_by') or source.get('uploaded_by')
            or source.get('edited_by') or source.get('user') or 'unknown')


def _reconcile_processing():
    """Flip scraped PDFs from 'processing' to 'processed' once their OCR text lands.

    scrape_import registers scraped PDFs immediately as 'processing' (so they show in
    the panel with the source PDF openable, without waiting for the whole OCR pass).
    OCR then writes stage2/<doc_dir>/structured.md but does NOT touch the registry —
    so this app stays the single registry writer and there is no cross-process race
    during the long OCR run. On each panel load we cheaply check the handful of
    'processing' docs and promote any whose OCR output now exists."""
    try:
        for d in doc_registry.all():
            # Only promote FIRST-time OCR of a scraped PDF (md_path still null). A
            # re-process keeps md_path set and manages its own status, so this never
            # races it back to 'processed' with stale text.
            if d.get('status') != 'processing' or not d.get('doc_dir') or d.get('md_path'):
                continue
            md = pc.paths.stage2 / d['doc_dir'] / 'structured.md'
            if not md.exists():
                continue
            fields = {'status': 'processed', 'md_path': str(md),
                      'md_hash': doc_pipeline.md_hash(md.read_text(encoding='utf-8')),
                      'error': None}
            used = pc.paths.used_files / f"{d['doc_dir']}.pdf"
            if used.exists():   # OCR's --archive-source moved the source PDF here
                fields['pdf_path'] = str(used)
            doc_registry.update(d['id'], **fields)
    except Exception as e:  # reconcile is best-effort; never fail the list call
        print(f"[Web UI] processing-reconcile skipped: {e}")


# The panel filters and badges by OCR method, which means every row needs it — but
# _processing_summary reads a JSON sidecar per document, and the registry runs to
# thousands. Memoised on (doc_dir, status): a confidence sidecar is written once by
# stage2 and never changes afterwards, so the only thing that can invalidate an entry
# is the document moving out of 'processing'.
_method_cache = {}


def _method_for(rec):
    """(method, label) for a document — cheap enough to call for every row."""
    key = (rec.get('doc_dir'), rec.get('status'), rec.get('text_source'))
    hit = _method_cache.get(key)
    if hit is None:
        summary = _processing_summary(rec)
        hit = (summary.get('method'), summary.get('label'))
        _method_cache[key] = hit
    return hit


@app.route('/api/documents', methods=['GET'])
def documents_list():
    _reconcile_processing()   # promote scraped PDFs whose OCR has since completed
    docs = []
    for rec in doc_registry.all():
        method, label = _method_for(rec)
        # Surfaced on the row itself so the reviewer can see "this was Sarvam" while
        # triaging, and filter to it, without opening each document first.
        docs.append({**rec, 'ocr_method': method, 'ocr_label': label})
    return jsonify({'success': True, 'documents': docs}), 200


@app.route('/api/documents/<doc_id>/spellcheck', methods=['GET'])
def documents_spellcheck(doc_id):
    """Words in this document's markdown that a reviewer should check.

    Read-only: returns character offsets + ranked replacement candidates. Nothing is
    rewritten here — the reviewer picks a suggestion and the edit goes through the
    existing PUT /md, so every change stays an explicit human action.
    """
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'success': False, 'error': 'Document not found'}), 404
    md_path = rec.get('md_path')
    if not md_path or not Path(md_path).is_file():
        return jsonify({'success': False, 'error': 'This document has no extracted text yet.'}), 400

    try:
        spellcheck = import_module("02_optimization.spellcheck")
    except Exception as e:
        return jsonify({'success': False, 'error': f'Spellcheck unavailable: {e}'}), 503

    try:
        text = Path(md_path).read_text(encoding='utf-8', errors='replace')
        # Page scores let the UI point at the weakest pages; absent for extracted text.
        pages = []
        d = pc.paths.stage2 / (rec.get('doc_dir') or '')
        if d.is_dir():
            for f in d.glob('*_confidence.json'):
                try:
                    pages = (json.loads(f.read_text(encoding='utf-8')) or {}).get('pages') or []
                    break
                except Exception:
                    continue
        result = spellcheck.find_issues(text, page_scores=pages)
        result['success'] = True
        return jsonify(result), 200
    except Exception as e:
        print(f'[Web UI] spellcheck failed for {doc_id}: {e}')
        return jsonify({'success': False, 'error': f'Spellcheck failed: {e}'}), 500


@app.route('/api/documents/<doc_id>/pdf', methods=['GET'])
def documents_pdf(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    # pdf_path can be null (e.g. OCR-imported docs that kept no source PDF) — skip
    # it rather than Path(None), which would raise and 500. Fall back to the
    # conventional on-disk locations before reporting a clean 404.
    candidates = []
    if rec.get('pdf_path'):
        candidates.append(Path(rec['pdf_path']))
    candidates.append(pc.paths.used_files / f"{rec['doc_dir']}.pdf")
    candidates.append(pc.paths.input_pdfs / f"{rec['doc_dir']}.pdf")
    for p in candidates:
        if p.exists():
            return send_file(str(p), mimetype='application/pdf')
    return jsonify({'error': 'No source PDF available for this document'}), 404


_RAW_MIME = {
    'pdf': 'application/pdf',
    'html': 'text/html', 'htm': 'text/html',
    # Serve structured/text snapshots as plain text so they render (not download)
    # inside the review iframe instead of triggering a save dialog.
    'json': 'text/plain; charset=utf-8', 'xml': 'text/plain; charset=utf-8',
    'txt': 'text/plain; charset=utf-8',
}


@app.route('/api/documents/<doc_id>/raw', methods=['GET'])
def documents_raw(doc_id):
    """Serve a scraped document's raw source snapshot (the content that produced
    its markdown). PDFs render in the browser viewer, HTML renders, JSON/XML/text
    display as plain text — so the reviewer sees exactly what was fetched."""
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    raw_path = rec.get('raw_path')
    if not raw_path or not Path(raw_path).exists():
        return jsonify({'error': 'No raw snapshot for this document'}), 404
    ext = (rec.get('raw_ext') or Path(raw_path).suffix.lstrip('.')).lower()
    return send_file(str(raw_path), mimetype=_RAW_MIME.get(ext, 'text/plain; charset=utf-8'))


@app.route('/api/documents/<doc_id>/md', methods=['GET'])
def documents_md_get(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    md_path = rec.get('md_path')
    if not md_path or not Path(md_path).exists():
        return jsonify({'success': False, 'status': rec.get('status'),
                        'error': 'Document is still processing or has no output yet'}), 200
    return jsonify({'success': True, 'markdown': Path(md_path).read_text(encoding='utf-8'),
                    'document': rec}), 200


@app.route('/api/documents/<doc_id>/md', methods=['PUT'])
def documents_md_put(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    if not rec.get('md_path'):
        return jsonify({'success': False, 'error': 'Document has no .md yet'}), 400

    data = request.get_json(silent=True) or {}
    md = data.get('markdown')
    if md is None:
        return jsonify({'success': False, 'error': 'markdown field is required'}), 400

    Path(rec['md_path']).write_text(md, encoding='utf-8')
    updated = doc_registry.update(doc_id, md_hash=doc_pipeline.md_hash(md),
                                  edited_by=_reviewer(data), edited_at=datetime.now().isoformat())

    # Re-index when the document is IN the index — verified in the panel, or indexed
    # by the ingest run (the GPU VM indexes every document without the verify step).
    # Keying this on `verified` alone left an admin's correction out of search: the
    # old text kept answering.
    reindexing = bool(rec.get('verified') or rec.get('indexed'))
    if reindexing:
        _bg(doc_pipeline.index_document, doc_id)
    return jsonify({'success': True, 'reindexing': reindexing, 'document': updated}), 200


def _processing_summary(rec: dict) -> dict:
    """How this document's markdown was produced — text extraction vs OCR, and for
    OCR the per-page Docling/Sarvam split (read live from the stage2 confidence
    sidecar). Drives the method badge in the panel and works for old docs too."""
    out = {'text_source': rec.get('text_source'), 'method': None, 'label': None,
           'pages': None, 'docling_pages': 0, 'sarvam_pages': 0}
    if rec.get('status') == 'processing':
        out['label'] = 'Processing…'
        return out
    if rec.get('text_source') == 'extract':
        src = 'Scraped web text' if rec.get('source_type') == 'web' else 'Embedded PDF text'
        out['method'] = 'extract'
        out['label'] = f'Text extraction ({src})'
        return out
    # OCR: read the per-page source split from stage2/<doc_dir>/*_confidence.json
    d = pc.paths.stage2 / (rec.get('doc_dir') or '')
    sidecar = None
    if d.is_dir():
        for f in d.glob('*_confidence.json'):
            try:
                sidecar = json.loads(f.read_text(encoding='utf-8'))
                break
            except Exception:
                continue
    # No confidence sidecar means the OCR pipeline never ran on this document. If
    # ocr_scanned kept the PDF's own text layer instead, it leaves extraction.json —
    # report that truthfully rather than inheriting the registry's historical "ocr"
    # label, which put an OCR badge on text no engine had touched.
    if sidecar is None and (d / 'extraction.json').is_file():
        try:
            marker = json.loads((d / 'extraction.json').read_text(encoding='utf-8'))
        except Exception:
            marker = {}
        if marker.get('method') == 'text_layer':
            out['method'] = 'extract'
            out['label'] = 'Text extraction (Embedded PDF text — no OCR was run)'
            return out

    pages = (sidecar or {}).get('pages') or []
    sarvam = sum(1 for p in pages if p.get('source') == 'sarvam')
    docling = sum(1 for p in pages if p.get('source') == 'docling')
    out.update(pages=len(pages) or None, docling_pages=docling, sarvam_pages=sarvam)
    if (sidecar or {}).get('engine') == 'vlm':
        # Local GPU vision-language model (core/01_preprocessing/vlm_ocr.py). Reported as method
        # 'ocr' — local OCR, no Sarvam — so the panel's filters and badges apply.
        reread = sum(1 for p in pages if p.get('source') == 'vlm-fallback')
        dropped = sum(1 for p in pages if p.get('status') == 'reject')
        model = ((sidecar.get('models') or {}).get('primary') or {}).get('model') or 'local model'
        extra = []
        if reread:
            extra.append(f'{reread} re-read')
        if dropped:
            extra.append(f'{dropped} dropped')
        out['method'] = 'ocr'
        out['label'] = f'OCR — {model} on GPU' + (f' ({", ".join(extra)})' if extra else '')
        return out
    if pages:
        if sarvam and docling:
            out['method'], out['label'] = 'auto', f'OCR — Docling + Sarvam ({sarvam}/{len(pages)} pages)'
        elif sarvam:
            out['method'], out['label'] = 'ocr_sarvam', 'OCR — Sarvam (all pages)'
        else:
            out['method'], out['label'] = 'ocr', 'OCR — Docling / EasyOCR'
    else:
        out['label'] = 'OCR' if rec.get('text_source') == 'ocr' else 'Unknown'
    return out


@app.route('/api/documents/<doc_id>/processing', methods=['GET'])
def documents_processing(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    return jsonify({'success': True, 'processing': _processing_summary(rec)}), 200


@app.route('/api/documents/<doc_id>/verify', methods=['POST'])
def documents_verify(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    if not rec.get('md_path'):
        return jsonify({'success': False, 'error': 'Document is not processed yet'}), 400

    data = request.get_json(silent=True) or {}
    updated = doc_registry.update(doc_id, verified=True, verified_by=_reviewer(data),
                                  verified_at=datetime.now().isoformat())
    _bg(doc_pipeline.index_document, doc_id)   # only verified docs get embedded
    return jsonify({'success': True, 'document': updated}), 200


@app.route('/api/documents/verify-bulk', methods=['POST'])
def documents_verify_bulk():
    """Verify many documents at once. Body: {"ids": [...]} or {"all_pending": true}.
    Marks each processed-but-unverified doc verified, then indexes them in ONE
    background worker (index_document serializes on a lock, so a single sequential
    thread is correct — hundreds of threads would just contend)."""
    data = request.get_json(silent=True) or {}
    reviewer = _reviewer(data)
    docs = doc_registry.all()
    if data.get('all_pending'):
        targets = [d['id'] for d in docs if d.get('status') == 'processed' and not d.get('verified')]
    else:
        wanted = set(data.get('ids') or [])
        targets = [d['id'] for d in docs
                   if d['id'] in wanted and d.get('status') == 'processed' and not d.get('verified')]
    now = datetime.now().isoformat()
    for doc_id in targets:
        doc_registry.update(doc_id, verified=True, verified_by=reviewer, verified_at=now)

    def _index_all(ids):
        for i in ids:
            doc_pipeline.index_document(i)
    if targets:
        _bg(_index_all, list(targets))
    return jsonify({'success': True, 'verified': len(targets)}), 200


@app.route('/api/documents/<doc_id>/unverify', methods=['POST'])
def documents_unverify(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    updated = doc_registry.update(doc_id, verified=False, verified_by=None, verified_at=None)
    _bg(doc_pipeline.remove_document_vectors, doc_id)   # pull it back out of the RAG
    return jsonify({'success': True, 'document': updated}), 200


@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def documents_delete(doc_id):
    """Discard a document from the review workspace: remove its RAG vectors (if any)
    and drop it from the registry, clearing it from the queue. Source/OCR files on
    disk are left untouched (a later re-import can bring it back) — this is a
    'reject', not a hard file delete."""
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    # Remove vectors BEFORE deregistering (the remover looks the record up by id).
    if rec.get('indexed'):
        try:
            doc_pipeline.remove_document_vectors(doc_id)
        except Exception as e:
            print(f"[Web UI] delete: vector removal failed for {doc_id}: {e}")
    doc_registry.remove(doc_id)
    return jsonify({'success': True, 'id': doc_id}), 200


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CHiPS-RAG Pipeline (Internal Backend)")
    print("="*70)
    print("\nRAG Pipeline Status: Deferred (loaded on demand)")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print("\nThis Flask server is INTERNAL ONLY.")
    print("Authentication is handled by Express.js at :3000")
    
    # INTERNAL ONLY: the Node UI proxies to us over loopback, and this backend has
    # NO auth of its own (auth lives at the Express layer). Bind to 127.0.0.1 by
    # default so :500x is never exposed on a public interface even if the firewall
    # is misconfigured. Override with FLASK_HOST only if you know you need it.
    flask_host = os.getenv('FLASK_HOST', '127.0.0.1')
    flask_port = int(os.getenv('FLASK_PORT', '5000'))
    flask_debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\nStarting Flask server on http://{flask_host}:{flask_port} (debug: {flask_debug})")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=flask_debug, host=flask_host, port=flask_port, use_reloader=False)
