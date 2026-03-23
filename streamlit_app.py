"""
🧠 RAG Intelligence — Streamlit UI v2
Multi-User Document AI | GPT-4o-mini + Qdrant + BGE + Hybrid Search + Re-ranking

Features:
  • ChatGPT-style persistent conversation history (per user, JSON-backed)
  • Sidebar with native Streamlit toggle + conversation list grouped by date
  • Streaming responses with full Markdown / link / table / image rendering
  • Anti-hallucination fallback, source citations, hybrid search + re-ranking

Run:  streamlit run streamlit_app.py
"""

import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Load .env BEFORE any library reads env vars (works both locally and on cloud)
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)  # override=False: real env vars win
except ImportError:
    pass  # python-dotenv not installed — rely on system env vars

import streamlit as st

# set_page_config MUST be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="🧠 RAG Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "**RAG Intelligence** — Multi-User AI Document Assistant"},
)

# ─────────────────────────────────────────────────────────────────────────────
# Path / backend init
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

CONV_DIR = ROOT_DIR / "data" / "conversations"
CONV_DIR.mkdir(parents=True, exist_ok=True)


def _inject_secrets() -> None:
    """Copy Streamlit Cloud secrets → env vars on first run."""
    try:
        for key in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        pass


_inject_secrets()


@st.cache_resource(show_spinner=False)
def _init_app():
    from config.config import get_config
    from src.utils import setup_logging
    c = get_config()
    os.makedirs(c.get_logs_dir(), exist_ok=True)
    setup_logging(
        log_level=c.log_level,
        log_file=os.path.join(c.get_logs_dir(), "streamlit.log"),
    )
    return c


cfg = _init_app()

from src.rag_pipeline import get_pipeline, invalidate_pipeline        # noqa: E402
from src.company_manager import get_company_manager                    # noqa: E402
from src.multi_format_processor import load_file, load_url, load_api   # noqa: E402

logger = logging.getLogger(__name__)
ALLOWED_EXT = {"pdf", "xlsx", "xls", "csv", "docx", "doc", "txt", "md", "markdown"}

# ─────────────────────────────────────────────────────────────────────────────
# CSS  —  fixes sidebar toggle visibility, ChatGPT-like conversation list,
#          proper table / link / image styling inside chat bubbles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide Streamlit chrome we don't need ────────────────────────────────── */
#MainMenu                              { visibility: hidden; }
footer                                 { visibility: hidden; }
/* Hide the top-right Deploy button */
[data-testid="stDeployButton"]         { display: none !important; }
/* Hide the top-right toolbar Actions (share / star etc.) */
[data-testid="stToolbarActions"]       { display: none !important; }

/* Minimise the default Streamlit header height but keep it in DOM so the
   sidebar collapse/expand chevron still works. */
header[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
    background: transparent !important;
}
/* The collapsed-sidebar expand arrow — always visible & clickable */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    top: 16px !important;
    left: 16px !important;
    z-index: 999999 !important;
    position: fixed !important;
    width: 40px !important;
    height: 40px !important;
    background: #4338ca !important;
    border: 1px solid #818cf8 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(67,56,202,0.45) !important;
    padding: 0 !important;
}
[data-testid="collapsedControl"] button {
    width: 100% !important;
    height: 100% !important;
    color: #fff !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
}
[data-testid="collapsedControl"] button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
}
[data-testid="collapsedControl"] svg {
    color: #ffffff !important;
    stroke: #ffffff !important;
}
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
}
/* Sidebar collapse button (the << inside the sidebar) */
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(79,70,229,0.25) !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 700 !important;
}

/* ── Sidebar dark gradient ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f0f1e 0%, #1a1240 100%) !important;
}
[data-testid="stSidebar"] * { color: #d4d4f4; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3  { color: #ffffff !important; }
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small { color: #8888aa !important; }
[data-testid="stSidebar"] ::-webkit-scrollbar       { width: 4px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: #4f46e5; border-radius: 4px;
}
/* Sidebar collapse chevron */
[data-testid="stSidebarCollapseButton"] button {
    background: #1a1240 !important;
    color: #d4d4f4 !important;
    border: 1px solid #4f46e580 !important;
}

/* ── Conversation history items ─────────────────────────────────────────── */
.conv-group-label {
    font-size: 0.70em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #6666aa;
    padding: 10px 2px 3px;
    margin: 0;
}

/* ── User badge ─────────────────────────────────────────────────────────── */
.user-badge {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 10px;
    padding: 10px 13px;
    color: #fff !important;
    margin: 4px 0 10px 0;
    font-size: 0.83em;
    line-height: 1.75;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35);
}
.user-badge strong { color: #fff !important; }

/* ── Welcome screen ─────────────────────────────────────────────────────── */
.welcome-box {
    text-align: center;
    padding: 70px 20px 40px;
    color: #9CA3AF;
}
.welcome-box h1 {
    font-size: 2.5em;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.welcome-tag {
    background: #ede9fe;
    border: 1px solid #c4b5fd;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82em;
    color: #4c1d95;
    margin: 4px;
    display: inline-block;
}

/* ── Chat header bar ────────────────────────────────────────────────────── */
.chat-header {
    font-size: 1.05em;
    font-weight: 700;
    color: #4f46e5;
    padding: 6px 0 8px;
    border-bottom: 2px solid #e0e7ff;
    margin-bottom: 12px;
}

/* ── Source chips ───────────────────────────────────────────────────────── */
.src-chip {
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78em;
    color: #3730a3;
    margin: 3px 4px;
    display: inline-block;
    word-break: break-all;
}
.src-chip a { color: #3730a3 !important; text-decoration: none; }
.src-chip a:hover { text-decoration: underline; }

/* ── Tables inside chat messages ────────────────────────────────────────── */
[data-testid="stChatMessage"] table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 0.9em;
}
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td {
    border: 1px solid #d1d5db;
    padding: 8px 12px;
    text-align: left;
}
[data-testid="stChatMessage"] th {
    background: #f3f4f6;
    font-weight: 600;
}
[data-testid="stChatMessage"] tr:nth-child(even) td { background: #f9fafb; }

/* ── Images inside chat messages ────────────────────────────────────────── */
[data-testid="stChatMessage"] img {
    max-width: 100%;
    border-radius: 8px;
    margin: 8px 0;
    display: block;
}

/* ── Links inside chat messages ─────────────────────────────────────────── */
[data-testid="stChatMessage"] a {
    color: #4f46e5;
    text-decoration: underline;
}
[data-testid="stChatMessage"] a:hover { color: #7c3aed; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div {
    color: #0f172a !important;
    line-height: 1.62;
    font-size: 1.01rem;
}

/* ── Primary buttons ────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.3);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(79,70,229,0.5) !important;
    transform: translateY(-1px);
}

/* Sidebar secondary/default buttons (chat history, user controls) */
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background: #241d56 !important;
    color: #e6e9ff !important;
    border: 1px solid rgba(167, 180, 255, 0.34) !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background: #30286c !important;
    color: #ffffff !important;
    border-color: rgba(196, 206, 255, 0.7) !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] summary { font-size: 0.88em; color: #6366f1; }

/* Keep sidebar expander states readable (avoid white active background). */
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background: #171339 !important;
    border: 1px solid rgba(129, 140, 248, 0.28) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: #171339 !important;
    color: #eef2ff !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: #211a50 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details[open] summary {
    background: #2a245f !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    color: #c7d2fe !important;
    stroke: #c7d2fe !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details[open] summary svg {
    color: #ffffff !important;
    stroke: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details > div {
    background: #171339 !important;
    color: #dbe3ff !important;
}

/* ── Code blocks in answers ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] pre {
    background: #111827;
    color: #f8fafc;
    border-radius: 8px;
    padding: 12px 16px;
    overflow-x: auto;
    font-size: 0.92em;
    line-height: 1.55;
}
[data-testid="stChatMessage"] code {
    background: #e2e8f0;
    color: #0f172a;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 0.88em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation persistence  (JSON files in data/conversations/)
# ─────────────────────────────────────────────────────────────────────────────

def _conv_path(user: str) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in user.lower()).strip("_")
    return CONV_DIR / f"{slug}.json"


def _load_convs(user: str) -> Dict:
    """Return {conv_id: conv_dict} from disk; empty dict if not found."""
    p = _conv_path(user)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_convs(user: str, convs: Dict) -> None:
    try:
        _conv_path(user).write_text(
            json.dumps(convs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Could not save conversations for '{user}': {e}")


def _new_conv(user: str) -> str:
    """Create a new empty conversation; return its ID."""
    cid = uuid.uuid4().hex[:12]
    convs = _load_convs(user)
    convs[cid] = {
        "id": cid,
        "title": "New Chat",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    _save_convs(user, convs)
    return cid


def _get_conv(user: str, cid: str) -> Optional[Dict]:
    return _load_convs(user).get(cid)


def _append_msg(user: str, cid: str, role: str, content: str,
                sources: Optional[List] = None, not_found: bool = False) -> None:
    """Append a message to a conversation and persist."""
    convs = _load_convs(user)
    if cid not in convs:
        return
    c = convs[cid]
    c["messages"].append({
        "role": role,
        "content": content,
        "sources": [s for s in (sources or []) if s and s not in ("unknown_source", "unknown_file")],
        "not_found": not_found,
        "ts": datetime.utcnow().isoformat(),
    })
    c["updated_at"] = datetime.utcnow().isoformat()
    # Auto-title from first user message
    if role == "user" and c["title"] == "New Chat":
        c["title"] = content[:55] + ("…" if len(content) > 55 else "")
    _save_convs(user, convs)


def _del_conv(user: str, cid: str) -> None:
    convs = _load_convs(user)
    convs.pop(cid, None)
    _save_convs(user, convs)


def _group_convs(convs: Dict) -> Dict[str, List]:
    """Group conversations into Today / Yesterday / Last 7 Days / Older."""
    today = date.today()
    groups: Dict[str, List] = {
        "Today": [], "Yesterday": [], "Last 7 Days": [], "Older": []
    }
    for c in sorted(convs.values(), key=lambda x: x.get("updated_at", ""), reverse=True):
        try:
            d = datetime.fromisoformat(c["updated_at"]).date()
        except Exception:
            d = date.min
        if d == today:
            groups["Today"].append(c)
        elif d == today - timedelta(days=1):
            groups["Yesterday"].append(c)
        elif d >= today - timedelta(days=7):
            groups["Last 7 Days"].append(c)
        else:
            groups["Older"].append(c)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────

for _k, _v in {"selected_user": None, "active_conv_id": None}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_sources(raw: list) -> list:
    out = []
    for s in raw:
        s = (s or "").strip()
        if not s:
            continue
        low = s.lower()
        if low in {"document", "document 1", "document 2", "source", "source 1", "source 2"}:
            continue
        if s.startswith("http"):
            label = s
        else:
            label = Path(s).name or s
            if label.lower().startswith("tmp"):
                continue
        if label not in out:
            out.append(label)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Ingest helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ingest_files(files) -> list:
    user = st.session_state.selected_user
    upload_dir = cfg.get_pdf_dir()
    os.makedirs(upload_dir, exist_ok=True)
    results = []
    for f in files:
        tmp_path = None
        try:
            suffix = Path(f.name).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=upload_dir) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            docs = load_file(tmp_path)
            if not docs:
                results.append(("❌", f.name, "No content could be extracted"))
                continue
            n = get_pipeline(user).ingest_documents(
                docs,
                source_type=suffix.lstrip(".").lower(),
                extra_metadata={"file_name": f.name},
            )
            results.append(("✅", f.name, f"{n:,} chunks indexed"))
        except Exception as e:
            results.append(("❌", f.name, str(e)))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    return results


def _scrape(url: str) -> tuple:
    user = st.session_state.selected_user
    try:
        docs = load_url(url, use_selenium=False)
        if not docs:
            return False, "Could not extract any content from the URL."
        n = get_pipeline(user).ingest_documents(
            docs, source_type="url", extra_metadata={"scraped_url": url}
        )
        return True, f"{len(docs)} page(s) → **{n:,} chunks** indexed"
    except Exception as e:
        return False, str(e)


def _scrape_api(
    url: str,
    method: str = "GET",
    auth_token: str = "",
    json_path: str = "",
    post_body: str = "",
) -> tuple:
    """Fetch data from a REST API endpoint and ingest it."""
    user = st.session_state.selected_user
    try:
        headers = {}
        if auth_token.strip():
            headers["Authorization"] = (
                auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
            )
        body = None
        if method == "POST" and post_body.strip():
            try:
                body = json.loads(post_body)
            except json.JSONDecodeError as je:
                return False, f"Invalid JSON body: {je}"
        docs = load_api(
            url,
            method=method,
            headers=headers,
            json_path=json_path.strip() or None,
            body=body,
        )
        if not docs:
            return False, "API returned no usable data."
        n = get_pipeline(user).ingest_documents(
            docs, source_type="api", extra_metadata={"api_url": url}
        )
        return True, f"{len(docs)} record(s) → **{n:,} chunks** indexed"
    except Exception as e:
        return False, str(e)


def _delete_user(user_name: str) -> None:
    mgr = get_company_manager()
    rec = mgr.get(user_name)
    if rec:
        try:
            from src.vector_store import QdrantVectorStore
            vsc = cfg.vectorstore
            cloud_url = vsc.get("cloud_url") or os.environ.get("QDRANT_URL")
            api_key   = vsc.get("api_key")   or os.environ.get("QDRANT_API_KEY")
            QdrantVectorStore(
                collection_name=rec["collection"],
                host=vsc.get("host", "localhost"),
                port=int(vsc.get("port", 6333)),
                use_cloud=bool(cloud_url),
                cloud_url=cloud_url,
                api_key=api_key,
            ).delete_collection()
        except Exception:
            pass
    mgr.delete(user_name)
    invalidate_pipeline(user_name)
    try:
        _conv_path(user_name).unlink(missing_ok=True)
    except Exception:
        pass
    st.session_state.selected_user = None
    st.session_state.active_conv_id = None
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧠 RAG Intelligence")
        st.caption("Multi-User · Document AI · GPT-4o-mini")
        st.divider()

        # ── User selector ─────────────────────────────────────────────────────
        st.markdown("### 👤 Select User")
        mgr   = get_company_manager()
        names = [u["name"] for u in mgr.list_companies()]
        opts  = ["— Select or create a user —"] + names

        cur_idx = 0
        if st.session_state.selected_user in names:
            cur_idx = names.index(st.session_state.selected_user) + 1

        sel = st.selectbox("User", opts, index=cur_idx, label_visibility="collapsed")
        if sel != opts[0]:
            if sel != st.session_state.selected_user:
                st.session_state.selected_user   = sel
                st.session_state.active_conv_id  = None
                st.rerun()
        elif st.session_state.selected_user not in names:
            st.session_state.selected_user  = None
            st.session_state.active_conv_id = None

        # ── Create new user ───────────────────────────────────────────────────
        with st.expander("➕ New User"):
            nu = st.text_input(
                "Name", placeholder="e.g. research, personal, work", key="nu_input"
            )
            if st.button("Create User", type="primary", key="nu_btn", use_container_width=True):
                if nu.strip():
                    mgr.get_or_create(nu.strip())
                    st.session_state.selected_user  = nu.strip()
                    st.session_state.active_conv_id = None
                    st.rerun()
                else:
                    st.warning("Please enter a user name.")

        if not st.session_state.selected_user:
            return

        user = st.session_state.selected_user
        rec  = mgr.get(user) or {}
        st.markdown(
            f"""<div class="user-badge">
                <strong>👤 {user}</strong><br>
                📄 <strong>{rec.get('doc_count', 0):,}</strong> chunks indexed
                &nbsp;·&nbsp; 📁 {', '.join(rec.get('sources', [])) or '—'}
            </div>""",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── New Chat ──────────────────────────────────────────────────────────
        if st.button("✏️ New Chat", type="primary", use_container_width=True, key="new_chat_btn"):
            cid = _new_conv(user)
            st.session_state.active_conv_id = cid
            try:
                get_pipeline(user).clear_memory()
            except Exception:
                pass
            st.rerun()

        # ── Conversation history ──────────────────────────────────────────────
        convs = _load_convs(user)
        if convs:
            active_cid = st.session_state.active_conv_id
            groups = _group_convs(convs)

            for group_label, items in groups.items():
                if not items:
                    continue
                st.markdown(
                    f'<div class="conv-group-label">{group_label}</div>',
                    unsafe_allow_html=True,
                )
                for c in items:
                    cid   = c["id"]
                    title = c.get("title", "New Chat")
                    icon  = "●" if cid == active_cid else "○"

                    col_btn, col_del = st.columns([7, 1])
                    with col_btn:
                        btn_type = "primary" if cid == active_cid else "secondary"
                        if st.button(
                            f"{icon} {title}",
                            key=f"conv_{cid}",
                            type=btn_type,
                            use_container_width=True,
                            help=title,
                        ):
                            if cid != active_cid:
                                st.session_state.active_conv_id = cid
                                # Reload pipeline memory with this conversation's turns
                                try:
                                    pl = get_pipeline(user)
                                    pl.clear_memory()
                                    for m in c.get("messages", []):
                                        pl.memory.add_message(m["role"], m["content"])
                                except Exception:
                                    pass
                                st.rerun()
                    with col_del:
                        if st.button("🗑", key=f"del_c_{cid}", help="Delete"):
                            _del_conv(user, cid)
                            if cid == active_cid:
                                st.session_state.active_conv_id = None
                            st.rerun()
        else:
            st.caption("No conversations yet — click ✏️ New Chat to start.")

        st.divider()

        # ── Upload documents ──────────────────────────────────────────────────
        with st.expander("📄 Upload Documents", expanded=False):
            st.caption("PDF · XLSX · CSV · DOCX · TXT · MD")
            upfiles = st.file_uploader(
                "Drop files here",
                type=list(ALLOWED_EXT),
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="uploader",
            )
            if upfiles:
                if st.button("⬆️ Ingest Files", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {len(upfiles)} file(s)…"):
                        results = _ingest_files(upfiles)
                    for icon, name, msg in results:
                        (st.success if icon == "✅" else st.error)(
                            f"{icon} **{name}** — {msg}"
                        )

        # ── Web scraping ──────────────────────────────────────────────────────
        with st.expander("🌐 Web Scraping", expanded=False):
            st.caption("Fetches & indexes an entire web page via URL.")
            sc_url = st.text_input(
                "Page URL", placeholder="https://example.com",
                label_visibility="collapsed", key="sc_url",
            )
            if st.button("🔍 Scrape & Ingest", use_container_width=True, key="sc_btn"):
                if sc_url.strip():
                    with st.spinner(f"Scraping {sc_url}…"):
                        ok, msg = _scrape(sc_url.strip())
                    (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")
                else:
                    st.warning("Please enter a URL.")

        # ── API Ingestion ─────────────────────────────────────────────────────
        with st.expander("🔌 API Ingestion", expanded=False):
            st.caption("Fetch JSON data from a REST API endpoint and ingest it.")
            api_url = st.text_input(
                "API Endpoint URL",
                placeholder="https://api.example.com/v1/articles",
                key="api_url",
            )
            api_method = st.radio(
                "Method", ["GET", "POST"], horizontal=True, key="api_method"
            )
            api_token = st.text_input(
                "Bearer Token (optional)",
                placeholder="Leave blank if not needed",
                type="password",
                key="api_token",
            )
            api_path = st.text_input(
                "JSON path to extract (optional)",
                placeholder="e.g.  data.results  or  items",
                key="api_path",
            )
            api_body = ""
            if api_method == "POST":
                api_body = st.text_area(
                    "Request body (JSON)",
                    placeholder='{"query": "test"}',
                    key="api_body",
                    height=80,
                )
            if st.button("⬇️ Fetch & Ingest", use_container_width=True, key="api_btn", type="primary"):
                if api_url.strip():
                    with st.spinner(f"Calling {api_url}…"):
                        ok, msg = _scrape_api(
                            api_url.strip(),
                            method=api_method,
                            auth_token=api_token,
                            json_path=api_path,
                            post_body=api_body,
                        )
                    (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")
                else:
                    st.warning("Please enter an API endpoint URL.")

        st.divider()

        # ── Danger zone ───────────────────────────────────────────────────────
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.caption("Permanently deletes this user and their entire vector index + history.")
            if st.checkbox("I understand — this is irreversible", key="del_confirm"):
                if st.button("🗑️ Delete User & All Data", type="secondary",
                             use_container_width=True):
                    _delete_user(user)


# ─────────────────────────────────────────────────────────────────────────────
# Source display
# ─────────────────────────────────────────────────────────────────────────────

def _show_sources(sources: list) -> None:
    fmt = _fmt_sources(sources)
    if not fmt:
        return
    unit = "source" if len(fmt) == 1 else "sources"
    with st.expander(f"📎 Sources · {len(fmt)} {unit}", expanded=False):
        chips = []
        for s in fmt:
            if s.startswith("http"):
                chips.append(
                    f'<span class="src-chip">🔗 <a href="{s}" target="_blank">{s}</a></span>'
                )
            else:
                chips.append(f'<span class="src-chip">📄 {s}</span>')
        st.markdown(" ".join(chips), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Render a single stored message
# ─────────────────────────────────────────────────────────────────────────────

def _render_msg(m: Dict) -> None:
    avatar = "🧑" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        # unsafe_allow_html=True → links, images, tables, and HTML from LLM render correctly
        st.markdown(m["content"], unsafe_allow_html=True)
        if m["role"] == "assistant":
            if m.get("sources"):
                _show_sources(m["sources"])
            if m.get("not_found"):
                st.caption("💡 Tip: Upload documents containing relevant information.")


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────

def _chat() -> None:
    user = st.session_state.selected_user

    # ── Welcome: no user selected ─────────────────────────────────────────────
    if not user:
        st.markdown("""
        <div class="welcome-box">
            <h1>🧠 RAG Intelligence</h1>
            <p style="font-size:1.15em;margin-bottom:6px">Multi-User AI Document Assistant</p>
            <p style="color:#6B7280">
                Select or create a <strong>User</strong> in the sidebar,<br>
                then click <strong>✏️ New Chat</strong> to start.
            </p>
            <div style="margin-top:24px">
                <span class="welcome-tag">📄 PDF</span>
                <span class="welcome-tag">📊 Excel / CSV</span>
                <span class="welcome-tag">📝 Word</span>
                <span class="welcome-tag">🌐 URLs</span>
                <span class="welcome-tag">🔍 Hybrid Search</span>
                <span class="welcome-tag">🔁 Re-ranking</span>
                <span class="welcome-tag">⚡ Streaming</span>
                <span class="welcome-tag">📎 Sources</span>
                <span class="welcome-tag">🛡️ Anti-hallucination</span>
                <span class="welcome-tag">💾 Chat History</span>
            </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── No active conversation yet ────────────────────────────────────────────
    cid = st.session_state.active_conv_id
    if not cid:
        st.markdown(
            f'<div class="chat-header">💬 {user}</div>'
            '<div style="text-align:center;padding:60px 20px;color:#9CA3AF">'
            '<p style="font-size:1.1em">Click <strong>✏️ New Chat</strong> in the sidebar to begin,<br>'
            'or select an existing conversation.</p></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Load conversation ─────────────────────────────────────────────────────
    conv = _get_conv(user, cid)
    if not conv:
        st.session_state.active_conv_id = None
        st.rerun()
        return

    title = conv.get("title", "New Chat")
    msgs  = conv.get("messages", [])

    # Header + clear button
    hcol1, hcol2 = st.columns([9, 1])
    with hcol1:
        st.markdown(f'<div class="chat-header">💬 {title}</div>', unsafe_allow_html=True)
    with hcol2:
        if st.button("🗑", key="clear_conv", help="Clear messages in this conversation"):
            convs = _load_convs(user)
            if cid in convs:
                convs[cid]["messages"] = []
                convs[cid]["title"]    = "New Chat"
                convs[cid]["updated_at"] = datetime.utcnow().isoformat()
                _save_convs(user, convs)
            try:
                get_pipeline(user).clear_memory()
            except Exception:
                pass
            st.rerun()

    # Render history
    for m in msgs:
        _render_msg(m)

    # Empty state
    if not msgs:
        st.markdown(
            '<div style="text-align:center;padding:50px 20px;color:#9CA3AF">'
            '<p style="font-size:1.05em">👋 Upload documents in the sidebar, '
            'then ask anything about them.</p>'
            '<p style="font-size:0.88em">Each user has a fully isolated vector index — '
            'your data stays private.</p></div>',
            unsafe_allow_html=True,
        )

    # Chat input
    if question := st.chat_input(f"Ask anything about {user}'s documents…"):
        _handle_question(user, cid, question)


# ─────────────────────────────────────────────────────────────────────────────
# Question handler  (streaming)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_question(user: str, conv_id: str, question: str) -> None:
    # Persist & immediately render user message
    _append_msg(user, conv_id, "user", question)
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question, unsafe_allow_html=True)

    # Stream assistant response
    with st.chat_message("assistant", avatar="🤖"):
        status_ph = st.empty()
        answer_ph = st.empty()
        status_ph.markdown("🔍 *Searching documents…*")

        full_text  = ""
        sources: List[str] = []
        not_found  = False
        first_tok  = True

        try:
            pipeline = get_pipeline(user)
            for event in pipeline.stream_answer(question):
                if "token" in event:
                    if first_tok:
                        status_ph.empty()
                        first_tok = False
                    full_text += event["token"]
                    # unsafe_allow_html ensures tables / links / images render
                    # while tokens are still streaming in
                    answer_ph.markdown(full_text + " ▌", unsafe_allow_html=True)

                elif "done" in event:
                    sources   = event.get("sources", [])
                    not_found = event.get("not_found", False)
                    break

                elif "error" in event:
                    full_text = f"❌ **Error:** {event['error']}"
                    status_ph.empty()
                    break

            status_ph.empty()
            # Final render without the streaming cursor ▌
            answer_ph.markdown(full_text, unsafe_allow_html=True)

            if sources:
                _show_sources(sources)
            if not_found:
                st.caption("💡 Tip: Upload relevant documents to improve answers.")

        except Exception as e:
            full_text = f"❌ **Unexpected error:** {e}"
            answer_ph.markdown(full_text, unsafe_allow_html=True)
            status_ph.empty()
            logger.error(f"Chat error for '{user}': {e}", exc_info=True)

    # Persist the assistant reply
    _append_msg(user, conv_id, "assistant", full_text, sources=sources, not_found=not_found)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _sidebar()
    _chat()


main()


