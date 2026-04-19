import time
import uuid

import streamlit as st
import streamlit.components.v1 as components

from api_client import api


@st.cache_data(ttl=8, show_spinner=False)
def _get_docs(project_id: str):
    return api.get_documents(project_id)

@st.cache_data(ttl=10, show_spinner=False)
def _is_online() -> bool:
    return api.is_online()

@st.cache_data(ttl=60, show_spinner=False)
def _get_guide(doc_id: str) -> dict:
    return api.get_source_guide(doc_id)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp { background-color: #0E1117; color: #E2E8F0; }

[data-testid="stSidebar"] {
    background-color: #13151F;
    border-right: 1px solid #1E2535;
}

/* ── Hide default Streamlit chrome ── */
footer { visibility: hidden; }

/* ── App header ── */
.app-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.app-subheader {
    color: #4A5568;
    font-size: 0.85rem;
    margin-top: -6px;
}

/* ── Status badge ── */
.badge-online  { background:#1C3A2A; color:#4ADE80; border:1px solid #22543D;
                 border-radius:20px; padding:3px 10px; font-size:11px; }
.badge-offline { background:#3A1C1C; color:#FC8181; border:1px solid #542222;
                 border-radius:20px; padding:3px 10px; font-size:11px; }

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #6C63FF, #4F46E5);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0 6px auto;
    max-width: 78%;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px #6C63FF30;
}
.assistant-bubble {
    background: #1A1F2E;
    color: #CBD5E0;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 6px auto 6px 0;
    max-width: 85%;
    font-size: 0.9rem;
    line-height: 1.6;
    border-left: 3px solid #6C63FF;
}

/* ── Citation card ── */
.citation-card {
    background: #141824;
    border: 1px solid #1E2535;
    border-left: 4px solid #00D4FF;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.82rem;
}
.citation-title { color: #00D4FF; font-weight: 600; }
.citation-meta  { color: #4A5568; font-size: 0.75rem; margin: 2px 0; }
.citation-text  { color: #718096; font-style: italic; margin-top: 6px; }
.citation-exact-quote {
    background: #0D1F2D;
    border-left: 3px solid #00D4FF;
    color: #E2E8F0;
    font-style: italic;
    font-size: 0.84rem;
    padding: 8px 12px;
    margin: 8px 0 4px 0;
    border-radius: 0 6px 6px 0;
    line-height: 1.6;
}
.citation-full-text {
    color: #4A5568;
    font-size: 0.78rem;
    margin-top: 4px;
    line-height: 1.5;
    padding: 4px 0;
}

/* ── Confidence bar ── */
.conf-high   { color: #4ADE80; }
.conf-medium { color: #FBBF24; }
.conf-low    { color: #FC8181; }

/* ── File status dots ── */
.dot-indexed    { color: #4ADE80; }
.dot-processing { color: #FBBF24; }
.dot-failed     { color: #FC8181; }
.dot-queued     { color: #718096; }

/* ── Follow-up buttons ── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #141824;
    color: #7C8FAC;
    border: 1px solid #1E2D45;
    border-radius: 20px;
    font-size: 0.78rem;
    padding: 4px 12px;
    transition: all 0.2s;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color: #6C63FF;
    color: #A5B4FC;
    background: #16192A;
}

/* ── Section labels ── */
.section-label {
    color: #4A5568;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 14px 0 6px 0;
}

/* ── Project pill ── */
.project-pill {
    background: #1A1F35;
    border: 1px solid #2D3A5E;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #818CF8;
    display: inline-block;
}

/* ── Doc row ── */
.doc-row {
    background: #141824;
    border: 1px solid #1E2535;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #2D3748;
}
.empty-icon { font-size: 3rem; }

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid #1E2535;
    margin: 12px 0;
}

/* ── Chat bubble animations ── */
.user-bubble {
    width: fit-content;
    margin-left: auto !important;
    animation: fadeSlideIn 0.25s ease;
}
.assistant-bubble {
    animation: fadeSlideIn 0.25s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Typing indicator (3 bouncing dots) ── */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 12px 18px;
    background: #1A1F2E;
    border-radius: 18px 18px 18px 4px;
    border-left: 3px solid #6C63FF;
    width: fit-content;
    margin: 8px 0;
}
.typing-dot {
    width: 8px; height: 8px;
    background: #6C63FF;
    border-radius: 50%;
    animation: typingBounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30%            { transform: translateY(-6px); opacity: 1; }
}

/* ── Premium chat input styling ── */
[data-testid="stChatInput"] {
    border: 1px solid #2D3A5E !important;
    border-radius: 14px !important;
    background: #141824 !important;
    box-shadow: 0 0 0 2px #6C63FF22 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 3px #6C63FF33 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────
def _init_state():
    defaults = {
        "project_id":    "default_project",
        "session_id":    str(uuid.uuid4()),
        "messages":      [],       # {"role": "user|assistant", "content": "...", "meta": {...}}
        "upload_jobs":   [],       # [{"filename": ..., "job_id": ..., "doc_id": ...}]
        "pending_input": "",       # pre-filled from follow-up button click
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + status — single row, no columns to avoid wrapping
    online = _is_online()
    badge  = "badge-online" if online else "badge-offline"
    label  = "● Online" if online else "● Offline"
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:4px 0 2px 0;">'
        f'  <div class="app-header" style="font-size:1.5rem;">🧠 DocuMind</div>'
        f'  <span class="{badge}" style="white-space:nowrap;">{label}</span>'
        f'</div>'
        f'<div class="app-subheader">Multi-PDF Intelligence</div>',
        unsafe_allow_html=True,
    )
    if not online:
        if st.button("🔄 Reconnect", key="reconnect_btn", use_container_width=True):
            st.rerun()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Project selector ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📁 Project</div>', unsafe_allow_html=True)

    col_proj, col_new = st.columns([3, 1])
    with col_proj:
        project_input = st.text_input(
            "Project ID",
            value=st.session_state["project_id"],
            label_visibility="collapsed",
            placeholder="project-name",
        )
        if project_input != st.session_state["project_id"]:
            st.session_state["project_id"]  = project_input
            st.session_state["messages"]    = []
            st.session_state["session_id"]  = str(uuid.uuid4())
            st.rerun()
    with col_new:
        if st.button("＋", help="New project"):
            st.session_state["project_id"]  = f"project-{str(uuid.uuid4())[:6]}"
            st.session_state["messages"]    = []
            st.session_state["session_id"]  = str(uuid.uuid4())
            st.rerun()

    st.markdown(
        f'<div class="project-pill">📂 {st.session_state["project_id"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── PDF Upload ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📤 Upload PDFs</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button("⬆ Upload & Index", use_container_width=True):
        with st.spinner("Sending to backend…"):
            result = api.upload_pdfs(uploaded, st.session_state["project_id"])

        for item in result.get("accepted", []):
            st.session_state["upload_jobs"].append(item)

        if result.get("rejected"):
            st.warning(f"Rejected (not PDF): {result['rejected']}")

    # ── Upload Progress ──────────────────────────────────────────────────────
    if st.session_state["upload_jobs"]:
        st.markdown('<div class="section-label">⏳ Processing</div>', unsafe_allow_html=True)
        still_running = []

        for job in st.session_state["upload_jobs"]:
            status = api.get_job_status(job["job_id"])
            pct    = status.get("progress", 0)
            stage  = status.get("status", "queued")

            if stage == "completed":
                st.markdown(
                    f'<span class="dot-indexed">✓</span> {job["filename"]}',
                    unsafe_allow_html=True,
                )
            elif stage == "failed":
                st.markdown(
                    f'<span class="dot-failed">✗</span> {job["filename"]}',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="dot-processing">⟳</span> {job["filename"]}',
                    unsafe_allow_html=True,
                )
                st.progress(pct / 100, text=f"{pct}% — {status.get('stage','…')}")
                still_running.append(job)

        st.session_state["upload_jobs"] = still_running

        if still_running:
            time.sleep(1.5)
            st.rerun()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Indexed Documents ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📄 Indexed Documents</div>', unsafe_allow_html=True)

    docs = _get_docs(st.session_state["project_id"])

    if not docs:
        st.markdown(
            '<p style="color:#2D3748;font-size:0.8rem;">No documents yet. Upload PDFs above.</p>',
            unsafe_allow_html=True,
        )
    else:
        for doc in docs:
            status_map = {
                "indexed":    ("dot-indexed",    "✓"),
                "processing": ("dot-processing", "⟳"),
                "failed":     ("dot-failed",     "✗"),
                "queued":     ("dot-queued",     "◌"),
            }
            css, icon = status_map.get(doc["status"], ("dot-queued", "◌"))
            pages = f'{doc["page_count"]}pg' if doc["page_count"] else ""
            label = (
                f'{icon} {doc["filename"][:26]} '
                f'<span style="color:#2D3748;font-size:0.72rem;">{pages}</span>'
            )

            if doc["status"] == "indexed":
                with st.expander(f'{icon} {doc["filename"][:26]}  {pages}'):
                    guide = _get_guide(doc["doc_id"])
                    if guide.get("summary"):
                        st.markdown(
                            f'<div style="color:#A0AEC0;font-size:0.82rem;margin-bottom:6px;">'
                            f'{guide["summary"]}</div>',
                            unsafe_allow_html=True,
                        )
                    if guide.get("key_topics"):
                        tags = " ".join(
                            f'<span style="background:#1A2535;color:#00D4FF;'
                            f'border-radius:4px;padding:2px 7px;font-size:0.72rem;">{t}</span>'
                            for t in guide["key_topics"]
                        )
                        st.markdown(tags, unsafe_allow_html=True)
                    if st.button("🗑 Delete", key=f"del_{doc['doc_id']}"):
                        api.delete_document(doc["doc_id"])
                        st.rerun()
            else:
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(
                        f'<span class="{css}">{icon}</span> '
                        f'<span style="font-size:0.82rem;color:#A0AEC0">{doc["filename"][:28]}</span>'
                        f'<span style="color:#2D3748;font-size:0.72rem;"> {pages}</span>',
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("🗑", key=f"del_{doc['doc_id']}", help="Delete"):
                        api.delete_document(doc["doc_id"])
                        st.rerun()

    # ── Collection Brief ─────────────────────────────────────────────────────
    indexed_count = len([d for d in docs if d["status"] == "indexed"]) if docs else 0
    if indexed_count > 0:
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        if st.button("📋 Generate Collection Brief", use_container_width=True):
            with st.spinner("Generating briefing…"):
                brief_data = api.generate_brief(st.session_state["project_id"])
            st.session_state["collection_brief"] = brief_data.get("brief", "")
        if st.session_state.get("collection_brief"):
            st.markdown(
                f'<div style="background:#0D1A0D;border:1px solid #2D4A2D;border-left:4px solid #4ADE80;'
                f'border-radius:8px;padding:12px 16px;margin-top:8px;">'
                f'<div style="color:#4ADE80;font-weight:700;font-size:0.78rem;margin-bottom:6px;">📋 COLLECTION BRIEF</div>'
                f'<div style="color:#A0AEC0;font-size:0.83rem;line-height:1.6;">'
                f'{st.session_state["collection_brief"]}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── New chat button ──────────────────────────────────────────────────────
    if st.button("💬 New Chat", use_container_width=True):
        st.session_state["messages"]   = []
        st.session_state["session_id"] = str(uuid.uuid4())
        st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────
tab_chat, tab_docs, tab_contra, tab_gaps, tab_questions, tab_insights, tab_timeline = st.tabs([
    "💬  Chat",
    "📄  Documents",
    "⚡  Contradictions",
    "🔍  Knowledge Gaps",
    "💡  Smart Questions",
    "🔮  Insights",
    "📅  Timeline",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT  (ChatGPT-style)
# ════════════════════════════════════════════════════════════════════════════

import re as _re

def _clean_answer(text: str) -> str:
    """Remove inline [1],[2]…[12] markers — citations shown separately below."""
    return _re.sub(r'\[\d+\]', '', text).strip()

with tab_chat:

    # ── pre-fill from follow-up button click ─────────────────────────────────
    prefill = st.session_state.pop("pending_input", "")

    # ════════════════════════════════════════════════════════════════════════
    # MESSAGE AREA — scrollable container (messages grow upward from bottom)
    # ════════════════════════════════════════════════════════════════════════
    if not st.session_state["messages"]:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🧠</div>
            <h3 style="color:#3D4A62;margin:12px 0 6px">DocuMind is ready</h3>
            <p style="color:#2D3748;font-size:0.85rem">
                Upload PDFs in the sidebar, then ask anything.<br>
                Answers include exact page citations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_box = st.container(height=555, border=False)
        with chat_box:
            for idx, msg in enumerate(st.session_state["messages"]):

                # ── User bubble (right-aligned purple) ───────────────────────
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Assistant reply ───────────────────────────────────────────
                else:
                    clean_text = _clean_answer(msg["content"])
                    st.markdown(
                        f'<div class="assistant-bubble">{clean_text}</div>',
                        unsafe_allow_html=True,
                    )

                    meta      = msg.get("meta", {})
                    citations = meta.get("citations", [])
                    confidence = meta.get("confidence", 0)
                    follow_ups = meta.get("follow_up_questions", [])

                    # ── Confidence bar ────────────────────────────────────────
                    if confidence:
                        conf_pct = int(confidence * 100)
                        conf_css = (
                            "conf-high"   if conf_pct >= 75 else
                            "conf-medium" if conf_pct >= 50 else
                            "conf-low"
                        )
                        col_conf, col_bar = st.columns([1, 4])
                        with col_conf:
                            st.markdown(
                                f'<span class="{conf_css}" style="font-size:0.8rem;font-weight:700">'
                                f'{conf_pct}% confidence</span>',
                                unsafe_allow_html=True,
                            )
                        with col_bar:
                            st.progress(confidence)

                    # ── Citations — CLICK TO EXPAND (like a box) ──────────────
                    if citations:
                        with st.expander(
                            f"📎 {len(citations)} Source{'s' if len(citations) > 1 else ''}  ·  click to view",
                            expanded=False,
                        ):
                            for c in citations:
                                exact_quote  = c.get("exact_quote", "")
                                text_snippet = c.get("text_snippet", "")
                                display_quote = exact_quote or text_snippet
                                quote_block = (
                                    f'<div style="border-left:3px solid #00D4FF;'
                                    f'background:#0D1A25;color:#A0AEC0;font-style:italic;'
                                    f'font-size:0.82rem;padding:8px 12px;margin-top:8px;'
                                    f'border-radius:0 6px 6px 0;line-height:1.6;">'
                                    f'"{display_quote}"</div>'
                                ) if display_quote else ""
                                st.markdown(f"""
                                <div style="background:#141824;border:1px solid #1E2535;
                                    border-left:4px solid #00D4FF;border-radius:8px;
                                    padding:10px 14px;margin:5px 0;">
                                    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                                        <span style="background:#00D4FF18;color:#00D4FF;font-weight:700;
                                            font-size:0.78rem;border-radius:4px;padding:2px 8px;">[{c['number']}]</span>
                                        <span style="color:#E2E8F0;font-weight:600;font-size:0.88rem;">
                                            📄 {c['filename']}</span>
                                        <span style="background:#1A2535;color:#6C63FF;font-weight:700;
                                            font-size:0.82rem;border-radius:4px;padding:2px 8px;margin-left:auto;">
                                            Page {c['page_number']}</span>
                                    </div>
                                    {quote_block}
                                </div>
                                """, unsafe_allow_html=True)

                    # ── Follow-up suggestion buttons ──────────────────────────
                    if follow_ups:
                        st.markdown(
                            '<div style="color:#2D3748;font-size:0.73rem;margin:8px 0 4px 0;">'
                            '💡 Follow-up suggestions</div>',
                            unsafe_allow_html=True,
                        )
                        cols = st.columns(len(follow_ups))
                        for i, (col, q) in enumerate(zip(cols, follow_ups)):
                            with col:
                                if st.button(f"↗ {q}", key=f"fu_{idx}_{i}"):
                                    st.session_state["pending_input"] = q
                                    st.rerun()

                    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Auto-scroll the chat container to the bottom ──────────────────────
        components.html("""
        <script>
        (function() {
            function scrollChat() {
                try {
                    var doc = window.parent.document;
                    // Target the scrollable div with most content (the chat container)
                    var divs = doc.querySelectorAll('div[data-testid="stVerticalBlockBorderWrapper"] div, div');
                    var best = null, bestH = 0;
                    for (var i = 0; i < divs.length; i++) {
                        var el = divs[i];
                        var oy = window.parent.getComputedStyle(el).overflowY;
                        if ((oy === 'scroll' || oy === 'auto') && el.scrollHeight > el.clientHeight + 20) {
                            if (el.scrollHeight > bestH) { bestH = el.scrollHeight; best = el; }
                        }
                    }
                    if (best) best.scrollTop = best.scrollHeight;
                } catch(e) {}
            }
            scrollChat();
            setTimeout(scrollChat, 120);
            setTimeout(scrollChat, 350);
            setTimeout(scrollChat, 700);
            setTimeout(scrollChat, 1200);
        })();
        </script>
        """, height=0, scrolling=False)

    # ════════════════════════════════════════════════════════════════════════
    # CHAT INPUT — placed AFTER the message container so Streamlit pins it
    #              to the BOTTOM of the page (ChatGPT behaviour)
    # ════════════════════════════════════════════════════════════════════════
    user_input = st.chat_input("Ask anything about your documents…")
    question   = user_input or prefill

    # ── Handle new question: save + fetch answer + rerun ─────────────────────
    if question:
        # 1. Store user message immediately
        st.session_state["messages"].append({"role": "user", "content": question})

        # 2. Show typing dots while waiting for backend
        typing_slot = st.empty()
        typing_slot.markdown(
            '<div class="typing-indicator">'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # 3. Query backend
        full_answer = ""
        metadata    = {}
        try:
            response = api.query_stream(
                question,
                st.session_state["project_id"],
                st.session_state["session_id"],
            )
            full_answer, metadata = api.parse_stream(response)
        except Exception as e:
            full_answer = f"⚠️ Backend unreachable. Make sure FastAPI is running.\n\n`{e}`"

        # 4. Clear typing indicator, save reply, rerun to render cleanly
        typing_slot.empty()
        st.session_state["messages"].append({
            "role":    "assistant",
            "content": full_answer,
            "meta":    metadata,
        })
        st.rerun()



# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENTS
# ════════════════════════════════════════════════════════════════════════════

with tab_docs:
    st.markdown("### 📄 Documents in this project")

    docs = _get_docs(st.session_state["project_id"])

    if not docs:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <p style="color:#2D3748;">No documents indexed yet.<br>Upload PDFs from the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary metrics
        total_pages  = sum(d.get("page_count", 0) for d in docs)
        indexed_docs = sum(1 for d in docs if d["status"] == "indexed")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total PDFs",    len(docs))
        c2.metric("Indexed",       indexed_docs)
        c3.metric("Total Pages",   total_pages)

        st.markdown("<br>", unsafe_allow_html=True)

        # Document table
        for doc in docs:
            status_map = {
                "indexed":    ("🟢", "Indexed"),
                "processing": ("🟡", "Processing"),
                "failed":     ("🔴", "Failed"),
                "queued":     ("⚪", "Queued"),
            }
            icon, label = status_map.get(doc["status"], ("⚪", doc["status"]))

            col_icon, col_name, col_pages, col_status, col_date, col_del = st.columns([1, 4, 2, 2, 3, 1])

            with col_icon:
                st.markdown(icon)
            with col_name:
                st.markdown(f"**{doc['filename']}**")
            with col_pages:
                st.markdown(f"`{doc['page_count']} pages`")
            with col_status:
                st.markdown(label)
            with col_date:
                upload_date = doc.get("upload_date", "")[:10] if doc.get("upload_date") else "—"
                st.markdown(f"_{upload_date}_")
            with col_del:
                if st.button("🗑", key=f"tab_del_{doc['doc_id']}"):
                    api.delete_document(doc["doc_id"])
                    st.rerun()

            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTRADICTION DETECTOR
# ════════════════════════════════════════════════════════════════════════════
with tab_contra:
    st.markdown("### ⚡ Contradiction Detector")
    st.markdown(
        '<p style="color:#4A5568;font-size:0.85rem;">'
        "Finds claims in different documents that directly contradict each other. "
        "Run the scan after uploading new PDFs."
        "</p>",
        unsafe_allow_html=True,
    )

    # Check unique filenames — same file uploaded twice doesn't count
    _all_docs  = _get_docs(st.session_state["project_id"])
    _indexed   = [d for d in _all_docs if d["status"] == "indexed"]
    _unique_names = set(d["filename"] for d in _indexed)

    if len(_unique_names) < 2:
        st.markdown(
            f'<div style="background:#1A1000;border:1px solid #7C4A00;border-left:4px solid #FBBF24;'
            f'border-radius:8px;padding:14px 18px;margin:10px 0;">'
            f'<div style="color:#FBBF24;font-weight:700;margin-bottom:4px;">⚠️ Need at least 2 different PDFs</div>'
            f'<div style="color:#A0AEC0;font-size:0.85rem;">'
            f'You currently have <b style="color:#FBBF24">{len(_unique_names)}</b> unique document(s): '
            f'{", ".join(_unique_names) or "none"}.<br>'
            f'Upload and index <b>at least 2 different PDFs</b> to compare them for contradictions.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Auto-load from DB (populated automatically after indexing)
        if "contra_cache" not in st.session_state:
            st.session_state["contra_cache"] = api.get_contradictions(st.session_state["project_id"])
        if st.button("🔄 Re-scan Contradictions", key="run_contra"):
            with st.spinner("Re-scanning for contradictions… (may take 20-40s)"):
                results = api.run_contradictions(st.session_state["project_id"])
            st.session_state["contra_cache"] = results
            st.rerun()

    contradictions = st.session_state.get("contra_cache") or []

    if not contradictions:
        if len(_unique_names) >= 2:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">✅</div>
                <p style="color:#2D3748;">No contradictions found yet.<br>
                Results appear automatically after indexing 2+ PDFs.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<p style="color:#4A5568;font-size:0.82rem;">'
            f'Found <b style="color:#FC8181">{len(contradictions)}</b> contradiction(s)</p>',
            unsafe_allow_html=True,
        )
        for contra in contradictions:
            conf_pct = int(contra["confidence"] * 100)
            with st.expander(f"⚡ {contra['topic']}  —  {conf_pct}% confidence", expanded=False):
                col_a, col_vs, col_b = st.columns([5, 1, 5])
                with col_a:
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<div class="citation-title">📄 {contra["doc_a"]}</div>'
                        f'<div class="citation-text">"{contra["text_a"]}"</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_vs:
                    st.markdown(
                        '<div style="text-align:center;padding-top:20px;color:#FC8181;font-weight:bold;">VS</div>',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.markdown(
                        f'<div class="citation-card" style="border-left-color:#FC8181;">'
                        f'<div class="citation-title" style="color:#FC8181;">📄 {contra["doc_b"]}</div>'
                        f'<div class="citation-text">"{contra["text_b"]}"</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<p style="color:#718096;font-size:0.83rem;margin-top:8px;">'
                    f'💬 {contra["explanation"]}</p>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — KNOWLEDGE GAPS
# ════════════════════════════════════════════════════════════════════════════
with tab_gaps:
    st.markdown("### 🔍 Knowledge Gap Finder")
    st.markdown(
        '<p style="color:#4A5568;font-size:0.85rem;">'
        "Identifies important topics your documents don't cover — "
        "so you know what to research next."
        "</p>",
        unsafe_allow_html=True,
    )

    _gaps_docs = _get_docs(st.session_state["project_id"])
    _gaps_indexed = [d for d in _gaps_docs if d["status"] == "indexed"]
    if not _gaps_indexed:
        st.markdown(
            '<div style="background:#1A1000;border:1px solid #7C4A00;border-left:4px solid #FBBF24;'
            'border-radius:8px;padding:14px 18px;">'
            '<div style="color:#FBBF24;font-weight:700;">⚠️ No indexed documents</div>'
            '<div style="color:#A0AEC0;font-size:0.85rem;margin-top:4px;">'
            'Upload and index at least 1 PDF first, then come back here.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Auto-load from DB (populated automatically after indexing)
        if "gaps_cache" not in st.session_state:
            st.session_state["gaps_cache"] = api.get_gaps(st.session_state["project_id"])
        if st.button("🔄 Re-analyse Gaps", use_container_width=False, key="run_gaps"):
            with st.spinner("Re-analysing document coverage… (15-30s)"):
                gaps = api.run_gaps(st.session_state["project_id"])
            st.session_state["gaps_cache"] = gaps
            st.rerun()

    gaps = st.session_state.get("gaps_cache") or []

    if not gaps and _gaps_indexed:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <p style="color:#2D3748;">No gaps found yet.<br>
            Results appear automatically after indexing.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, gap in enumerate(gaps, 1):
            st.markdown(
                f'<div class="citation-card" style="border-left-color:#FBBF24;">'
                f'<div class="citation-title" style="color:#FBBF24;">🔍 {i}. {gap["topic"]}</div>'
                f'<div class="citation-text" style="color:#A0AEC0;margin-top:6px;">{gap["description"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — SMART QUESTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_questions:
    st.markdown("### 💡 Smart Question Generator")
    st.markdown(
        '<p style="color:#4A5568;font-size:0.85rem;">'
        "Type a topic and get 5 smart questions tailored to your documents."
        "</p>",
        unsafe_allow_html=True,
    )

    _q_docs = _get_docs(st.session_state["project_id"])
    _q_indexed = [d for d in _q_docs if d["status"] == "indexed"]

    if not _q_indexed:
        st.markdown(
            '<div style="background:#1A1000;border:1px solid #7C4A00;border-left:4px solid #FBBF24;'
            'border-radius:8px;padding:14px 18px;">'
            '<div style="color:#FBBF24;font-weight:700;">⚠️ No indexed documents</div>'
            '<div style="color:#A0AEC0;font-size:0.85rem;margin-top:4px;">'
            'Upload and index at least 1 PDF first, then generate questions.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        q_topic = st.text_input(
            "Focus on a specific topic (optional)",
            placeholder="e.g. climate change, World War II, financial risk, chapter 3…",
            key="q_topic_input",
        )
        if st.button("✨ Generate Smart Questions", use_container_width=False, key="run_questions"):
            with st.spinner(f"Generating questions{' about: ' + q_topic if q_topic else ''}… (15-30s)"):
                questions = api.run_questions(st.session_state["project_id"], topic=q_topic)
            st.session_state["questions_cache"] = questions
            st.session_state["questions_topic"] = q_topic
            st.rerun()

    questions = st.session_state.get("questions_cache") or []

    if not questions and _q_indexed:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💡</div>
            <p style="color:#2D3748;">No questions generated yet.<br>
            Type a topic above and click Generate.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        topic_label = st.session_state.get("questions_topic", "")
        if topic_label:
            st.markdown(
                f'<div style="color:#6C63FF;font-size:0.82rem;font-weight:600;margin:4px 0 14px;">'
                f'📌 Questions about: {topic_label}</div>',
                unsafe_allow_html=True,
            )

        for idx, q in enumerate(questions, start=1):
            col_q, col_ask = st.columns([8, 2])
            with col_q:
                st.markdown(
                    f'<div style="background:#141824;border:1px solid #1E2535;'
                    f'border-left:4px solid #6C63FF;border-radius:8px;'
                    f'padding:12px 16px;margin:5px 0;display:flex;gap:12px;align-items:flex-start;">'
                    f'<span style="color:#6C63FF;font-weight:800;font-size:1rem;min-width:22px;">{idx}.</span>'
                    f'<span style="color:#CBD5E0;font-size:0.9rem;line-height:1.55;">{q["question"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_ask:
                if st.button("Ask →", key=f"ask_{q['id']}"):
                    with st.spinner("Getting answer…"):
                        st.session_state["messages"].append({
                            "role": "user", "content": q["question"]
                        })
                        try:
                            resp = api.query_stream(
                                q["question"],
                                st.session_state["project_id"],
                                st.session_state["session_id"],
                            )
                            full_answer, metadata = api.parse_stream(resp)
                        except Exception as e:
                            full_answer = f"⚠️ Error: {e}"
                            metadata = {}
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": full_answer,
                            "meta": metadata,
                        })
                    st.toast("Answer ready! Click the 💬 Chat tab.", icon="✅")
                    st.rerun()



# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — PROACTIVE INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("### 🔮 Proactive Insight Engine")
    st.markdown(
        '<p style="color:#4A5568;font-size:0.85rem;">'
        "Surfaces non-obvious patterns, trends, and anomalies you might not think to ask about."
        "</p>",
        unsafe_allow_html=True,
    )

    _ins_docs = _get_docs(st.session_state["project_id"])
    _ins_indexed = [d for d in _ins_docs if d["status"] == "indexed"]

    if not _ins_indexed:
        st.markdown(
            '<div style="background:#1A1000;border:1px solid #7C4A00;border-left:4px solid #FBBF24;'
            'border-radius:8px;padding:14px 18px;">'
            '<div style="color:#FBBF24;font-weight:700;">⚠️ No indexed documents</div>'
            '<div style="color:#A0AEC0;font-size:0.85rem;margin-top:4px;">'
            'Upload and index at least 1 PDF first, then mine insights.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        i_topic = st.text_input(
            "Focus on a specific topic (optional)",
            placeholder="e.g. economic impact, historical events, key findings, chapter 5…",
            key="i_topic_input",
        )
        if st.button("🔮 Mine Insights", use_container_width=False, key="run_insights"):
            with st.spinner(f"Analysing for hidden patterns{' about: ' + i_topic if i_topic else ''}… (15-30s)"):
                insights = api.run_insights(st.session_state["project_id"], topic=i_topic)
            st.session_state["insights_cache"] = insights
            st.session_state["insights_topic"] = i_topic
            st.rerun()

    insights = st.session_state.get("insights_cache") or []

    if not insights:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔮</div>
            <p style="color:#2D3748;">No insights mined yet.<br>
            Click "Mine Insights" after indexing documents.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        type_meta = {
            "pattern": ("🔄", "#6C63FF"),
            "trend":   ("📈", "#4ADE80"),
            "anomaly": ("⚠️",  "#FC8181"),
        }

        for insight in insights:
            icon, color = type_meta.get(insight["insight_type"], ("💡", "#00D4FF"))
            supporting = ", ".join(insight.get("supporting_docs", []))
            st.markdown(
                f'<div class="citation-card" style="border-left-color:{color};margin-bottom:10px;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'  <span style="font-size:1.1rem;">{icon}</span>'
                f'  <span style="color:{color};font-weight:700;font-size:0.9rem;">{insight["title"]}</span>'
                f'  <span style="background:{color}20;color:{color};border-radius:10px;'
                f'    padding:1px 8px;font-size:0.7rem;">{insight["insight_type"].upper()}</span>'
                f'</div>'
                f'<div style="color:#A0AEC0;font-size:0.85rem;line-height:1.5;">{insight["description"]}</div>'
                f'{"<div style=color:#4A5568;font-size:0.75rem;margin-top:6px;>📄 " + supporting + "</div>" if supporting else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — DOCUMENT EVOLUTION TIMELINE
# ════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown("### 📅 Document Evolution Timeline")
    st.markdown(
        '<p style="color:#4A5568;font-size:0.85rem;">'
        "See how research findings and methodologies evolved chronologically across your documents."
        "</p>",
        unsafe_allow_html=True,
    )

    if st.button("📅 Build Timeline", use_container_width=False, key="run_timeline"):
        with st.spinner("Building timeline… (may take 20-40s for many documents)"):
            timeline_events = api.get_timeline(st.session_state["project_id"])
            st.session_state["timeline_events"] = timeline_events

    # Display cached timeline if available
    timeline_events = st.session_state.get("timeline_events", [])

    if not timeline_events:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📅</div>
            <p style="color:#2D3748;">No timeline built yet.<br>
            Click "Build Timeline" after indexing documents.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        importance_colors = {"high": "#FC8181", "medium": "#FBBF24", "low": "#4A5568"}

        current_year = None
        for event in timeline_events:
            year = event.get("year", "Unknown")

            # Year header
            if year != current_year:
                current_year = year
                st.markdown(
                    f'<div style="background:linear-gradient(90deg,#6C63FF20,transparent);'
                    f'border-left:4px solid #6C63FF;padding:8px 16px;margin:20px 0 10px;'
                    f'border-radius:0 8px 8px 0;">'
                    f'<span style="color:#818CF8;font-weight:800;font-size:1.1rem;">{year}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            color = importance_colors.get(event.get("importance", "medium"), "#4A5568")
            st.markdown(
                f'<div style="display:flex;gap:12px;margin:6px 0 6px 20px;">'
                f'  <div style="width:3px;background:{color};border-radius:2px;flex-shrink:0;"></div>'
                f'  <div class="citation-card" style="border-left:none;flex:1;margin:0;'
                f'       border-left-color:transparent;">'
                f'    <div style="color:#4A5568;font-size:0.73rem;margin-bottom:3px;">'
                f'      📄 {event["filename"]}'
                f'    </div>'
                f'    <div style="color:#CBD5E0;font-size:0.88rem;">{event["claim"]}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
