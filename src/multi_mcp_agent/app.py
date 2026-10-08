import hashlib
import json
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from multi_mcp_agent.agent import AgentRuntime, message_to_text
from multi_mcp_agent.rag import save_uploaded_file

# =========================================================
# Streamlit Configuration
# =========================================================

st.set_page_config(
    page_title="Multi MCP Agent",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Session State
# =========================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "main"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "last_tools_used" not in st.session_state:
    st.session_state.last_tools_used = []

if "last_agent_state" not in st.session_state:
    st.session_state.last_agent_state = "Ready"

if "indexed_upload_keys" not in st.session_state:
    st.session_state.indexed_upload_keys = set()

if "active_upload_keys" not in st.session_state:
    st.session_state.active_upload_keys = {}

if "upload_errors" not in st.session_state:
    st.session_state.upload_errors = {}


# =========================================================
# Theme
# =========================================================

if st.session_state.dark_mode:
    THEME = {
        "bg": "#090b12",
        "bg_soft": "#10131d",
        "surface": "rgba(20, 24, 36, 0.92)",
        "surface_strong": "#161b28",
        "sidebar": "rgba(13, 16, 26, 0.96)",
        "border": "rgba(255, 255, 255, 0.10)",
        "border_strong": "rgba(143, 115, 255, 0.34)",
        "text": "#f5f7fb",
        "muted": "#9ca6ba",
        "input": "#171b26",
        "user_bubble": "#22283a",
        "assistant_bubble": "#151a26",
        "card": "rgba(24, 29, 44, 0.90)",
        "shadow": "rgba(0, 0, 0, 0.26)",
        "hero_glow": "rgba(132, 91, 255, 0.20)",
    }
else:
    THEME = {
        "bg": "#f7f8fc",
        "bg_soft": "#ffffff",
        "surface": "rgba(255, 255, 255, 0.90)",
        "surface_strong": "#ffffff",
        "sidebar": "rgba(251, 251, 254, 0.95)",
        "border": "rgba(37, 44, 69, 0.10)",
        "border_strong": "rgba(105, 76, 255, 0.22)",
        "text": "#25283a",
        "muted": "#70778c",
        "input": "#ffffff",
        "user_bubble": "#f1efff",
        "assistant_bubble": "#ffffff",
        "card": "rgba(255, 255, 255, 0.92)",
        "shadow": "rgba(59, 46, 112, 0.10)",
        "hero_glow": "rgba(132, 91, 255, 0.13)",
    }

st.markdown(
    f"""
    <style>
        :root {{
            --app-bg: {THEME['bg']};
            --bg-soft: {THEME['bg_soft']};
            --surface: {THEME['surface']};
            --surface-strong: {THEME['surface_strong']};
            --sidebar: {THEME['sidebar']};
            --border: {THEME['border']};
            --border-strong: {THEME['border_strong']};
            --text: {THEME['text']};
            --muted: {THEME['muted']};
            --input: {THEME['input']};
            --user-bubble: {THEME['user_bubble']};
            --assistant-bubble: {THEME['assistant_bubble']};
            --card: {THEME['card']};
            --shadow: {THEME['shadow']};
            --hero-glow: {THEME['hero_glow']};
            --accent-a: #6c5ce7;
            --accent-b: #9b5de5;
            --accent-c: #41c7f0;
            --success: #22c55e;
            --danger: #ff5c6c;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 88% -8%, rgba(255, 118, 179, 0.16), transparent 31%),
                radial-gradient(circle at 74% 4%, rgba(132, 91, 255, 0.18), transparent 34%),
                radial-gradient(circle at 55% -4%, rgba(65, 199, 240, 0.15), transparent 28%),
                var(--app-bg) !important;
            color: var(--text) !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stToolbar"] {
            color: var(--text) !important;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.25rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            width: 286px !important;
            background: var(--sidebar) !important;
            border-right: 1px solid var(--border);
            backdrop-filter: blur(22px);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.1rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--border) !important;
        }

        .app-brand {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            margin-bottom: 0.3rem;
        }

        .app-logo {
            width: 36px;
            height: 36px;
            border-radius: 11px;
            display: grid;
            place-items: center;
            color: white;
            font-weight: 800;
            font-size: 1rem;
            background: linear-gradient(135deg, #22d3ee, #7c3aed 52%, #ec4899);
            box-shadow: 0 9px 26px rgba(124, 58, 237, 0.22);
        }

        .app-name {
            font-size: 1.05rem;
            font-weight: 760;
            letter-spacing: -0.02em;
        }

        .app-subtle {
            font-size: 0.75rem;
            color: var(--muted) !important;
        }

        .section-label {
            margin: 1.15rem 0 0.52rem 0;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-weight: 760;
            font-size: 0.68rem;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.25rem 0 0.95rem 0;
        }

        .topbar-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text);
        }

        .topbar-status {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.38rem 0.62rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--muted);
            font-size: 0.73rem;
            box-shadow: 0 8px 26px var(--shadow);
        }

        .ready-dot {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.50);
        }

        .hero-shell {
            max-width: 860px;
            margin: 7vh auto 1.35rem auto;
            text-align: center;
            position: relative;
        }

        .hero-shell::before {
            content: "";
            position: absolute;
            width: 390px;
            height: 220px;
            border-radius: 50%;
            filter: blur(65px);
            background: var(--hero-glow);
            left: 50%;
            top: -55px;
            transform: translateX(-50%);
            pointer-events: none;
        }

        .hero-mark {
            width: 54px;
            height: 54px;
            margin: 0 auto 1rem auto;
            border-radius: 16px;
            display: grid;
            place-items: center;
            color: white;
            font-size: 1.35rem;
            font-weight: 800;
            background: linear-gradient(135deg, #22d3ee, #7c3aed 52%, #ec4899);
            box-shadow: 0 16px 42px rgba(124, 58, 237, 0.24);
            position: relative;
        }

        .hero-title {
            position: relative;
            font-size: clamp(2rem, 4vw, 3.15rem);
            font-weight: 760;
            letter-spacing: -0.045em;
            color: var(--text);
            line-height: 1.08;
        }

        .hero-copy {
            position: relative;
            max-width: 650px;
            margin: 0.7rem auto 0 auto;
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.55;
        }

        .capability-grid {
            max-width: 900px;
            margin: 1.1rem auto 1.35rem auto;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .capability-card {
            min-height: 108px;
            padding: 0.92rem;
            border: 1px solid var(--border);
            background: var(--card);
            border-radius: 18px;
            text-align: left;
            box-shadow: 0 14px 42px var(--shadow);
            backdrop-filter: blur(16px);
        }

        .capability-icon {
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            border-radius: 10px;
            background: rgba(108, 92, 231, 0.10);
            margin-bottom: 0.66rem;
            font-size: 1.05rem;
        }

        .capability-title {
            color: var(--text);
            font-size: 0.84rem;
            font-weight: 720;
        }

        .capability-copy {
            color: var(--muted);
            font-size: 0.72rem;
            line-height: 1.42;
            margin-top: 0.2rem;
        }

        .chat-shell {
            max-width: 900px;
            margin: 0 auto;
        }

        div[data-testid="stChatMessage"] {
            background: transparent !important;
            border: 0 !important;
            padding: 0 !important;
            margin-bottom: 0.85rem !important;
        }

        div[data-testid="stChatMessage"] > div {
            align-items: flex-start !important;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            color: var(--text) !important;
            line-height: 1.58;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: var(--text) !important;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
            color: #7c3aed !important;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
            background: var(--user-bubble);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.82rem 1rem;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
            background: var(--assistant-bubble);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.82rem 1rem;
            box-shadow: 0 8px 24px var(--shadow);
        }

        .composer-wrap {
            max-width: 900px;
            margin: 0.65rem auto 0 auto;
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--border-strong) !important;
            border-radius: 20px !important;
            background: var(--surface) !important;
            box-shadow: 0 16px 50px var(--shadow) !important;
            padding: 0.36rem 0.44rem !important;
            backdrop-filter: blur(20px);
        }

        div[data-testid="stTextInput"] input {
            background: var(--input) !important;
            color: var(--text) !important;
            caret-color: #7c3aed !important;
            border: 0 !important;
            border-radius: 14px !important;
            min-height: 46px !important;
            font-size: 0.95rem !important;
            box-shadow: none !important;
            -webkit-text-fill-color: var(--text) !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
            -webkit-text-fill-color: var(--muted) !important;
        }

        div[data-testid="stTextInput"] label {
            display: none !important;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            background: var(--surface-strong) !important;
            color: var(--text) !important;
            min-height: 2.45rem;
            box-shadow: none !important;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            border-color: rgba(124, 58, 237, 0.38) !important;
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6c5ce7, #8b5cf6) !important;
            color: white !important;
            border-color: transparent !important;
        }

        [data-testid="stFileUploader"] {
            border: 1px dashed var(--border-strong) !important;
            border-radius: 14px !important;
            background: var(--surface) !important;
            padding: 0.25rem !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
        }

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] label {
            color: var(--muted) !important;
        }

        .doc-summary {
            padding: 0.72rem 0.78rem;
            border-radius: 13px;
            border: 1px solid var(--border);
            background: var(--surface);
            margin-bottom: 0.52rem;
        }

        .doc-title {
            color: var(--text);
            font-weight: 700;
            font-size: 0.82rem;
            overflow-wrap: anywhere;
        }

        .doc-meta {
            color: var(--muted);
            font-size: 0.70rem;
            margin-top: 0.2rem;
        }

        .auto-index {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.48rem;
            border-radius: 999px;
            border: 1px solid rgba(34, 197, 94, 0.20);
            background: rgba(34, 197, 94, 0.08);
            color: #22a957;
            font-size: 0.69rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }

        .status-strip {
            max-width: 900px;
            margin: 0.35rem auto 0.8rem auto;
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
        }

        .status-pill {
            padding: 0.32rem 0.54rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--surface);
            color: var(--muted);
            font-size: 0.70rem;
        }

        .tool-note {
            color: var(--muted);
            font-size: 0.72rem;
            margin-top: 0.25rem;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            background: var(--surface) !important;
            border-radius: 14px !important;
        }

        @media (max-width: 1050px) {
            .capability-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 700px) {
            .block-container {
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }

            .hero-shell {
                margin-top: 2.5rem;
            }

            .capability-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Persistent Runtime
# =========================================================


@st.cache_resource(show_spinner=False)
def get_runtime():
    return AgentRuntime()


try:
    runtime = get_runtime()
except Exception as exc:
    st.error(f"Unable to start agent: {exc}")
    st.stop()


# =========================================================
# Helpers
# =========================================================


def render_agent_stream(event_stream):
    response_placeholder = st.empty()
    streamed_text = ""

    # Show the assistant immediately while the model/tool is working.
    response_placeholder.markdown(
        """
        <div class="ai-thinking">
            <span class="ai-thinking-robot">🤖</span>
            <span>Thinking</span>
            <span class="thinking-dots">...</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.last_agent_state = "Thinking"

    for event in event_stream:
        event_type = event.get("type")

        if event_type == "token":
            token = event.get("text", "")
            streamed_text += token

            response_placeholder.markdown(
                streamed_text + "▌"
            )

        elif event_type == "approval":
            response_placeholder.empty()
            st.session_state.last_agent_state = "Approval required"
            return "approval"

        elif event_type == "error":
            response_placeholder.empty()

            kind = event.get("kind")
            message = event.get("message", "Unknown error")
            st.session_state.last_agent_state = "Error"

            if kind == "quota":
                st.error("Gemini quota/rate limit has been reached.")
            elif kind == "timeout":
                st.error("The request timed out. Please try again.")
            elif kind == "unavailable":
                st.error(
                    "Gemini is temporarily unavailable because of high demand. "
                    "Please try again shortly."
                )
            else:
                st.error(f"Agent error: {message}")

            return "error"

        elif event_type == "done":
            final_response = event.get("response", "")

            response_placeholder.markdown(
                final_response
                or streamed_text
                or "Request completed."
            )

            tools_used = event.get("tools_used", [])
            st.session_state.last_tools_used = tools_used
            st.session_state.last_agent_state = "Ready"

            if tools_used:
                st.markdown(
                    '<div class="tool-note">Tools used: '
                    + ", ".join(tools_used)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            if event.get("draft_created"):
                st.toast("Gmail draft created successfully.")

            return "done"

    return None


def get_thread_documents():
    try:
        result = runtime.call_tool(
            "list_documents",
            {"thread_id": st.session_state.thread_id},
        )

        if isinstance(result, dict):
            return result

        return {"success": False, "error": str(result)}

    except Exception as exc:
        return {"success": False, "error": str(exc)}


def start_new_chat():
    st.session_state.thread_id = "chat-" + uuid.uuid4().hex[:12]
    st.session_state.last_tools_used = []
    st.session_state.last_agent_state = "Ready"


def auto_index_uploads(uploaded_files):
    thread_id = st.session_state.thread_id
    previous_keys = set(st.session_state.active_upload_keys.get(thread_id, set()))

    current_keys = set()

    for uploaded_pdf in uploaded_files or []:
        digest = hashlib.sha256(uploaded_pdf.getvalue()).hexdigest()
        current_keys.add(f"{thread_id}:{digest}")

    # If a selected file is cleared from the uploader, forget only the
    # UI-side processed marker. The persistent RAG index is unchanged.
    # Re-selecting that file later can therefore index it again if needed.
    for removed_key in previous_keys - current_keys:
        st.session_state.indexed_upload_keys.discard(removed_key)
        st.session_state.upload_errors.pop(removed_key, None)

    if not uploaded_files:
        st.session_state.active_upload_keys.pop(thread_id, None)
        return

    st.session_state.active_upload_keys[thread_id] = current_keys

    for uploaded_pdf in uploaded_files:
        file_bytes = uploaded_pdf.getvalue()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        digest = hashlib.sha256(file_bytes).hexdigest()
        upload_key = f"{thread_id}:{digest}"

        if upload_key in st.session_state.indexed_upload_keys:
            continue

        if upload_key in st.session_state.upload_errors:
            continue

        if file_size_mb > 25:
            st.session_state.upload_errors[upload_key] = (
                f"{uploaded_pdf.name} is larger than 25 MB."
            )
            continue

        try:
            with st.spinner(f"Indexing {uploaded_pdf.name}…"):
                saved = save_uploaded_file(
                    uploaded_pdf.name,
                    file_bytes,
                )

                result = runtime.call_tool(
                    "index_document",
                    {
                        "file_path": saved["stored_path"],
                        "thread_id": st.session_state.thread_id,
                    },
                )

            if isinstance(result, dict) and result.get("success"):
                st.session_state.indexed_upload_keys.add(upload_key)
                st.session_state.upload_errors.pop(upload_key, None)
                st.session_state.upload_notice = (
                    f"{uploaded_pdf.name} indexed automatically."
                )
                st.rerun()

            error_message = (
                result.get("error", "Unable to index PDF.")
                if isinstance(result, dict)
                else str(result)
            )
            st.session_state.upload_errors[upload_key] = error_message

        except Exception as exc:
            st.session_state.upload_errors[upload_key] = str(exc)


# =========================================================
# Current State
# =========================================================

document_result = get_thread_documents()
current_documents = (
    document_result.get("documents", []) if document_result.get("success") else []
)

try:
    saved_messages = runtime.load_messages(st.session_state.thread_id)
except Exception as exc:
    saved_messages = []
    st.error(f"Unable to load conversation: {exc}")

try:
    pending_approval = runtime.get_pending_approval(st.session_state.thread_id)
except Exception:
    pending_approval = None


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="app-brand">
            <div class="app-logo">✦</div>
            <div>
                <div class="app-name">Multi MCP Agent</div>
                <div class="app-subtle">Local tool-grounded workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch between day and night appearance.",
    )

    if st.button(
        "＋ New chat",
        type="primary",
        use_container_width=True,
    ):
        start_new_chat()
        st.rerun()

    st.markdown('<div class="section-label">Chats</div>', unsafe_allow_html=True)

    try:
        threads = runtime.list_threads(limit=30)
    except Exception:
        threads = []

    if not threads:
        st.caption("No saved conversations yet.")

    for thread in threads:
        thread_id = thread["thread_id"]
        title = thread["title"]
        active = thread_id == st.session_state.thread_id
        label = ("● " if active else "") + title

        if st.button(
            label,
            key=("thread_" + thread_id),
            use_container_width=True,
        ):
            st.session_state.thread_id = thread_id
            st.session_state.last_tools_used = []
            st.session_state.last_agent_state = "Ready"
            st.rerun()

    st.divider()
    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)

    uploaded_pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=("pdf_upload_" + st.session_state.thread_id),
        label_visibility="collapsed",
        help="PDFs are indexed automatically after upload.",
    )

    auto_index_uploads(uploaded_pdfs)

    if uploaded_pdfs:
        st.markdown(
            '<div class="auto-index">✓ Auto indexing enabled</div>',
            unsafe_allow_html=True,
        )

        for uploaded_pdf in uploaded_pdfs:
            digest = hashlib.sha256(uploaded_pdf.getvalue()).hexdigest()
            upload_key = f"{st.session_state.thread_id}:{digest}"
            error = st.session_state.upload_errors.get(upload_key)

            if error:
                st.error(f"{uploaded_pdf.name}: {error}")
                if st.button(
                    "Retry",
                    key=("retry_" + digest),
                    use_container_width=True,
                ):
                    st.session_state.upload_errors.pop(upload_key, None)
                    st.rerun()

    if current_documents:
        st.caption(f"Indexed in this chat · {len(current_documents)}")

    for document in current_documents:
        document_id = document["document_id"]
        file_name = document["file_name"]

        st.markdown(
            f"""
            <div class="doc-summary">
                <div class="doc-title">📄 {file_name}</div>
                <div class="doc-meta">
                    {document['page_count']} pages · {document['chunk_count']} chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Remove",
            key=("remove_document_" + document_id),
            use_container_width=True,
        ):
            try:
                result = runtime.call_tool(
                    "delete_document",
                    {
                        "document_id": document_id,
                        "thread_id": st.session_state.thread_id,
                    },
                )

                if isinstance(result, dict) and result.get("success"):
                    st.rerun()

                st.error(
                    result.get("error", "Unable to remove document.")
                    if isinstance(result, dict)
                    else str(result)
                )

            except Exception as exc:
                st.error(f"Unable to remove document: {exc}")

    if not document_result.get("success"):
        st.caption("Document index is unavailable.")

    st.divider()

    if st.button(
        "Delete current chat",
        use_container_width=True,
    ):
        try:
            for document in current_documents:
                runtime.call_tool(
                    "delete_document",
                    {
                        "document_id": document["document_id"],
                        "thread_id": st.session_state.thread_id,
                    },
                )

            runtime.delete_thread(st.session_state.thread_id)
            start_new_chat()
            st.rerun()

        except Exception as exc:
            st.error(f"Unable to delete chat: {exc}")


# =========================================================
# One-time Notices
# =========================================================

upload_notice = st.session_state.pop("upload_notice", None)
if upload_notice:
    st.toast(upload_notice)


# =========================================================
# Main Top Bar
# =========================================================

st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-title">Chat</div>
        <div class="topbar-status">
            <span class="ready-dot"></span>
            {runtime.tool_count} MCP tools · {len(current_documents)} PDF(s)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Main Chat Area
# =========================================================

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

if not saved_messages and not pending_approval:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-mark">✦</div>
            <div class="hero-title">Hi, how can I help you?</div>
            <div class="hero-copy">
                Work with your connected MCP tools, ask questions about indexed PDFs,
                and keep every answer grounded in the active capability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="capability-grid">
            <div class="capability-card">
                <div class="capability-icon">🌦️</div>
                <div class="capability-title">Weather</div>
                <div class="capability-copy">Current conditions and forecasts.</div>
            </div>
            <div class="capability-card">
                <div class="capability-icon">✉️</div>
                <div class="capability-title">Gmail</div>
                <div class="capability-copy">Search, read, and create drafts.</div>
            </div>
            <div class="capability-card">
                <div class="capability-icon">📅</div>
                <div class="capability-title">Calendar</div>
                <div class="capability-copy">Inspect events and protected updates.</div>
            </div>
            <div class="capability-card">
                <div class="capability-icon">📄</div>
                <div class="capability-title">PDF RAG</div>
                <div class="capability-copy">Hybrid retrieval with page citations.</div>
            </div>
            <div class="capability-card">
                <div class="capability-icon">🌐</div>
                <div class="capability-title">Fetch</div>
                <div class="capability-copy">Read public web content safely.</div>
            </div>
            <div class="capability-card">
                <div class="capability-icon">⑂</div>
                <div class="capability-title">Git</div>
                <div class="capability-copy">Inspect and manage repository actions.</div>
            </div>
            <div class="capability-card">
                <div class="capability-icon">▶️</div>
                <div class="capability-title">YouTube</div>
                <div class="capability-copy">Find and open requested music.</div>
            </div>
            <div class="capability-card">
                <div class="capability-icon">🧠</div>
                <div class="capability-title">Memory</div>
                <div class="capability-copy">Persistent thread context with SQLite.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if saved_messages:
    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-pill">● {st.session_state.last_agent_state}</div>
            <div class="status-pill">{runtime.tool_count} tools connected</div>
            <div class="status-pill">{len(current_documents)} indexed PDF(s)</div>
            <div class="status-pill">Human approval enabled</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in saved_messages:
    if isinstance(message, HumanMessage):
        text = message_to_text(message.content)
        if text:
            with st.chat_message("user", avatar="👤"):
                st.markdown(text)

    elif isinstance(message, AIMessage):
        text = message_to_text(message.content)
        if text:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(text)

# Reserve a slot ABOVE the composer so newly submitted turns appear
# in the correct order while streaming.
live_turn_slot = st.container()


# =========================================================
# Human Approval UI
# =========================================================

if pending_approval:
    st.warning("Human approval required")
    st.write(
        pending_approval.get(
            "message",
            "The agent wants to perform a protected action.",
        )
    )

    tool_calls = pending_approval.get("tool_calls", [])

    for index, tool_call in enumerate(tool_calls, start=1):
        name = tool_call.get("name", "Unknown tool")
        args = tool_call.get("args", {})

        with st.expander(f"Action {index} · {name}", expanded=True):
            st.code(
                json.dumps(args, indent=2, default=str),
                language="json",
            )

    approve_column, cancel_column = st.columns(2)

    with approve_column:
        if st.button(
            "Approve action",
            type="primary",
            use_container_width=True,
        ):
            with live_turn_slot:
                with st.chat_message("assistant", avatar="🤖"):
                    result = render_agent_stream(
                        runtime.resume_approval(
                            st.session_state.thread_id,
                            approved=True,
                        )
                    )

            if result in {"done", "approval"}:
                st.rerun()

    with cancel_column:
        if st.button(
            "Cancel action",
            use_container_width=True,
        ):
            with live_turn_slot:
                with st.chat_message("assistant", avatar="🤖"):
                    result = render_agent_stream(
                        runtime.resume_approval(
                            st.session_state.thread_id,
                            approved=False,
                        )
                    )

            if result in {"done", "approval"}:
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# =========================================================
# Composer
# =========================================================

st.markdown('<div class="composer-wrap">', unsafe_allow_html=True)

with st.form(
    key=("composer_form_" + st.session_state.thread_id),
    clear_on_submit=True,
    border=False,
):
    input_column, send_column = st.columns([12, 1])

    with input_column:
        user_prompt = st.text_input(
            "Message",
            placeholder=(
                "Message Multi MCP Agent — ask about PDFs, weather, Gmail, "
                "Calendar, Git, Fetch, or YouTube"
            ),
            label_visibility="collapsed",
            key=("composer_input_" + st.session_state.thread_id),
        )

    with send_column:
        submitted = st.form_submit_button(
            "➤",
            type="primary",
            use_container_width=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Process User Turn
# =========================================================

if submitted and user_prompt.strip():
    clean_prompt = user_prompt.strip()

    with live_turn_slot:
        with st.chat_message("user", avatar="👤"):
            st.markdown(clean_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            result = render_agent_stream(
                runtime.stream_turn(
                    clean_prompt,
                    st.session_state.thread_id,
                )
            )

    # Always rerun after a completed turn so persisted history is rendered
    # above the composer. This fixes the old question/input/answer ordering bug.
    if result in {"done", "approval"}:
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
