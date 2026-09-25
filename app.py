import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

load_dotenv()

# On Streamlit Cloud the key comes from Secrets; locally it comes from .env.
if not os.environ.get("GOOGLE_API_KEY"):
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

if not os.environ.get("GOOGLE_API_KEY"):
    st.set_page_config(page_title="ScholarChat AI", page_icon="🎓", layout="centered")
    st.error(
        "Google API key not found. Add `GOOGLE_API_KEY` to your Streamlit Secrets "
        "(.streamlit/secrets.toml locally, or the Cloud Secrets editor) or to a `.env` file."
    )
    st.stop()

st.set_page_config(page_title="ScholarChat AI", page_icon="🎓", layout="centered")

PAGE_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #f4f1ff;
    --muted: #c9bff2;
    --violet: #7c4dff;
    --pink: #ff2e88;
    --cyan: #22d3ee;
    --card: rgba(255, 255, 255, 0.06);
    --card-border: rgba(255, 255, 255, 0.14);
}

.stApp {
    font-family: 'Poppins', system-ui, -apple-system, sans-serif;
    color: var(--ink);
}

#MainMenu,
footer,
[data-testid="stToolbar"] {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 520px at 8% -8%, rgba(124, 77, 255, 0.55), transparent 60%),
        radial-gradient(820px 500px at 92% -12%, rgba(255, 46, 136, 0.45), transparent 60%),
        radial-gradient(760px 540px at 50% 112%, rgba(34, 211, 238, 0.32), transparent 62%),
        linear-gradient(160deg, #0c0322 0%, #160a3a 45%, #25074a 100%);
    background-attachment: fixed;
}

/* ---------- Navbar ---------- */
.scholar-nav {
    position: sticky;
    top: 8px;
    z-index: 99;
    backdrop-filter: blur(16px);
    background: rgba(14, 4, 38, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    padding: 14px 22px;
    margin: 6px 0 20px;
    box-shadow: 0 12px 44px rgba(0, 0, 0, 0.35);
}

.scholar-nav-inner {
    display: flex;
    align-items: center;
    gap: 16px;
}

.nav-logo {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    flex: 0 0 auto;
    background: linear-gradient(135deg, #7c4dff, #ff2e88 60%, #22d3ee);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 20px;
    color: #fff;
    box-shadow: 0 8px 26px rgba(255, 46, 136, 0.4);
    letter-spacing: 0.5px;
}

.nav-text { line-height: 1.25; }

.nav-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 0.3px;
    background: linear-gradient(90deg, #c4b5fd, #f9a8d4, #67e8f9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.nav-tag {
    color: var(--muted);
    font-size: 13px;
    margin-top: 2px;
}

/* ---------- Hero ---------- */
.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    border: 1px solid var(--card-border);
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.03));
    backdrop-filter: blur(14px);
    padding: 34px 32px 30px;
    margin: 10px 0 30px;
    box-shadow: 0 26px 64px rgba(0, 0, 0, 0.42);
}

.hero-glow {
    position: absolute;
    top: -130px;
    right: -90px;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 46, 136, 0.5), transparent 70%);
    pointer-events: none;
}

.hero-glow--left {
    top: auto;
    bottom: -140px;
    left: -90px;
    right: auto;
    background: radial-gradient(circle, rgba(124, 77, 255, 0.5), transparent 70%);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 15px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    background: rgba(255, 255, 255, 0.06);
    color: #e9e2ff;
    font-size: 12.5px;
    font-weight: 600;
}

.hero-badge .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.22);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.25); }
    50% { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0.05); }
}

.hero-title {
    font-size: 42px;
    line-height: 1.12;
    font-weight: 800;
    margin: 20px 0 10px;
}

.hero-title span {
    background: linear-gradient(90deg, #a78bfa, #f472b6, #22d3ee);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-desc {
    color: var(--muted);
    font-size: 15px;
    max-width: 640px;
    margin: 0 0 26px;
}

.specs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
    gap: 12px;
}

.spec-pill {
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.05);
    padding: 12px 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.spec-pill:hover {
    transform: translateY(-3px);
    border-color: rgba(255, 46, 136, 0.6);
}

.spec-label {
    display: block;
    font-size: 10.5px;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #9d93c7;
    font-weight: 600;
}

.spec-value {
    display: block;
    margin-top: 4px;
    font-size: 14.5px;
    font-weight: 600;
    color: #efeafe;
}

/* ---------- Section headings ---------- */
.section-kicker {
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #f472b6;
    font-weight: 700;
    margin-bottom: 4px;
}

.section-title {
    font-size: 26px;
    font-weight: 800;
    margin: 0;
}

/* ---------- Bordered container (search + answer card) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 22px !important;
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02)) !important;
    backdrop-filter: blur(12px);
    padding: 10px 14px 14px !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.3);
}

/* ---------- Widgets ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1.5px dashed rgba(197, 178, 255, 0.55) !important;
    border-radius: 18px !important;
    transition: border-color 0.25s, background 0.25s;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--pink) !important;
    background: rgba(255, 46, 136, 0.08) !important;
}

[data-testid="stFileUploaderDropzone"] div {
    color: #d9d1f5 !important;
}

div[data-testid="stFileUploaderFile"] {
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.16);
    overflow: hidden;
}

[data-testid="stTextInput"] > div > div {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    transition: border-color 0.2s;
}

[data-testid="stTextInput"]:hover > div > div {
    border-color: rgba(255, 46, 136, 0.6) !important;
}

[data-testid="stTextInput"] input {
    color: var(--ink) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #8f84b8 !important;
}

[data-testid="stAlert"] {
    background: rgba(34, 211, 238, 0.1) !important;
    border: 1px solid rgba(34, 211, 238, 0.4) !important;
    border-radius: 14px !important;
    color: #cffafe !important;
}

[data-testid="stAlert"] p { color: #cffafe !important; }

[data-testid="stCaptionContainer"] { color: #9d93c7 !important; }

[data-testid="stMarkdownContainer"] p { color: #ded7f7; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 10px 14px;
    margin-bottom: 4px;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #7c4dff, #ff2e88) !important;
    color: #ffffff !important;
    font-weight: 700;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #22d3ee, #7c4dff) !important;
    color: #ffffff !important;
    font-weight: 700;
}

[data-testid="stChatMessageAvatarUser"]::before,
[data-testid="stChatMessageAvatarAssistant"]::before {
    content: "";
}

/* ---------- Ask button ---------- */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #7c4dff, #ff2e88) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-family: 'Poppins', sans-serif !important;
    height: 48px;
    box-shadow: 0 6px 18px rgba(255, 46, 136, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stButton button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(255, 46, 136, 0.45);
}

/* ---------- Footer ---------- */
.scholar-footer {
    margin-top: 46px;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.02));
    backdrop-filter: blur(12px);
    padding: 26px 26px 18px;
    box-shadow: 0 20px 54px rgba(0, 0, 0, 0.3);
}

.f-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 19px;
    font-weight: 800;
}

.nav-logo.sm {
    width: 38px;
    height: 38px;
    font-size: 15px;
    border-radius: 12px;
}

.f-tag {
    color: var(--muted);
    font-size: 13px;
    margin: 8px 0 0;
}

.footer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 22px;
    margin: 20px 0;
}

.f-col-title {
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    color: #f472b6;
    margin-bottom: 9px;
}

.f-col a {
    color: #c4b5fd;
    text-decoration: none;
    font-weight: 600;
    font-size: 13.5px;
    display: block;
    margin-bottom: 5px;
    word-break: break-all;
}

.f-col a:hover {
    color: #f9a8d4;
    text-decoration: underline;
}

.f-col span,
.f-col p {
    color: #a99fd0;
    font-size: 13px;
    display: block;
    line-height: 1.6;
    margin: 0;
}

.footer-bottom {
    border-top: 1px dashed rgba(255, 255, 255, 0.16);
    padding-top: 14px;
    margin-top: 12px;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    color: #9d93c7;
    font-size: 12.5px;
}

@media (max-width: 640px) {
    .hero-title { font-size: 30px; }
    .nav-title { font-size: 21px; }
    .hero-card { padding: 26px 22px 24px; }
    .specs-grid { grid-template-columns: 1fr; }
}
</style>"""

st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ---- session state ----
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.file_name = None

if "stats" not in st.session_state:
    st.session_state.stats = {"answered": 0, "last_time": None}

if "history" not in st.session_state:
    st.session_state.history = []

if "qbox" not in st.session_state:
    st.session_state.qbox = ""

if st.session_state.pop("pending_clear_qbox", False):
    st.session_state.qbox = ""

# ---- Navbar ----
st.markdown(
    """
    <header class="scholar-nav">
        <div class="scholar-nav-inner">
            <div class="nav-logo">SC</div>
            <div class="nav-text">
                <div class="nav-title">ScholarChat AI</div>
                <div class="nav-tag">Your smart RAG assistant &mdash; answers grounded in your syllabus</div>
            </div>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)

# ---- Hero card with app specs ----
answered = st.session_state.stats["answered"]
last_time = st.session_state.stats["last_time"]
response_text = f"{last_time:.2f} s" if last_time is not None else "Live &mdash; measured on your first question"
answered_text = f"<b>{answered}</b> answered" if answered else "Waiting for your first question"

st.markdown(
    f"""
    <section class="hero-card">
        <div class="hero-glow"></div>
        <div class="hero-glow hero-glow--left"></div>
        <div class="hero-badge"><span class="dot"></span> Retrieval-Augmented Generation &middot; Powered by Google Gemini</div>
        <h1 class="hero-title">Ask anything from your <span>syllabus</span>.</h1>
        <p class="hero-desc">
            Drop in a PDF, and ScholarChat AI answers with context taken straight from the
            document &mdash; not guesses. Type or speak your question and get a grounded
            response in seconds.
        </p>
        <div class="specs-grid">
            <div class="spec-pill"><span class="spec-label">LLM</span><span class="spec-value">Gemini 2.5 Flash</span></div>
            <div class="spec-pill"><span class="spec-label">Embeddings</span><span class="spec-value">gemini-embedding-001</span></div>
            <div class="spec-pill"><span class="spec-label">Vector DB</span><span class="spec-value">In-memory vector store</span></div>
            <div class="spec-pill"><span class="spec-label">Retrieval</span><span class="spec-value">Semantic search &middot; Top 3 chunks</span></div>
            <div class="spec-pill"><span class="spec-label">Chunking</span><span class="spec-value">1200 tokens &middot; 200 overlap</span></div>
            <div class="spec-pill"><span class="spec-label">Voice input</span><span class="spec-value">Web Speech API &middot; Chrome/Edge</span></div>
            <div class="spec-pill"><span class="spec-label">Response time</span><span class="spec-value">{response_text}</span></div>
            <div class="spec-pill"><span class="spec-label">Questions</span><span class="spec-value">{answered_text}</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ---- Chat area ----
st.markdown(
    """
    <div class="section-kicker">Start here</div>
    <div class="section-title">Ask your syllabus</div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    uploaded_file = st.file_uploader("Upload a syllabus PDF to begin", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.file_name != uploaded_file.name:
            st.session_state.vector_store = None
            st.session_state.file_name = uploaded_file.name
            st.session_state.history = []
            st.session_state.stats = {"answered": 0, "last_time": None}
            st.session_state.qbox = ""

        if st.session_state.vector_store is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200,
                )
                chunks = splitter.split_documents(documents)

                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

                st.session_state.vector_store = InMemoryVectorStore.from_documents(
                    chunks, embeddings
                )
            except GoogleGenerativeAIError as exc:
                st.error(
                    "Embedding quota reached. Wait about 20 seconds and try again, "
                    "or upload a smaller PDF."
                )
                st.caption(str(exc))
                st.stop()
            finally:
                Path(pdf_path).unlink(missing_ok=True)

        db = st.session_state.vector_store
        st.success("PDF processed successfully! Your syllabus is ready for questions.")

        # ---- Render chat history (all Q&A stays visible) ----
        if not st.session_state.history:
            with st.chat_message("assistant"):
                st.markdown("Upload your syllabus PDF above, then ask your first question. Every answer stays in this chat.")
        else:
            for entry in st.session_state.history:
                with st.chat_message("user"):
                    st.markdown(entry["question"])
                with st.chat_message("assistant"):
                    st.markdown(entry["answer"])
                    st.caption(
                        f"Retrieved {entry['chunks']} chunks in {entry['retrieval_ms']:.0f} ms &middot; "
                        f"generated in {entry['gen_s']:.2f} s &middot; total {entry['total_s']:.2f} s"
                    )

        # ---- Mic Button (Web Speech API) ----
        components.html(
            """
            <div style="display:flex; align-items:center; gap:12px; margin:6px 0 10px;">
                <button onclick="startListening()" id="micBtn"
                    style="padding:10px 20px; border-radius:12px;
                           background:linear-gradient(135deg,#7c4dff,#ff2e88);
                           color:white; border:none; cursor:pointer; font-size:14px;
                           font-weight:600; font-family:'Poppins',sans-serif;
                           box-shadow:0 6px 18px rgba(255,46,136,.35);">
                     Speak Question
                </button>
                <span id="status" style="color:#a99fd0; font-size:13px;">
                    Click mic and speak (Chrome/Edge only)
                </span>
            </div>

            <script>
            function startListening() {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SpeechRecognition) {
                    document.getElementById("status").innerText = "Not supported. Use Chrome or Edge.";
                    return;
                }

                const recognition = new SpeechRecognition();
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;

                document.getElementById("micBtn").innerText = "Listening...";
                document.getElementById("status").innerText = "Speak now...";

                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    document.getElementById("status").innerText = "Heard: " + transcript;
                    document.getElementById("micBtn").innerText = "Speak Question";

                    const url = new URL(window.parent.location.href);
                    url.searchParams.set("transcript", transcript);
                    window.parent.location.href = url.toString();
                };

                recognition.onerror = (e) => {
                    let msg = "Error: " + e.error;
                    if (e.error === "network") {
                        msg += " — Check your internet connection. Firewalls/VPNs may block speech recognition.";
                    } else if (e.error === "not-allowed") {
                        msg += " — Please allow microphone access in your browser.";
                    } else if (e.error === "no-speech") {
                        msg += " — No speech detected. Try speaking louder or closer to the mic.";
                    }
                    document.getElementById("status").innerText = msg;
                    document.getElementById("micBtn").innerText = "Speak Question";
                };

                recognition.onend = () => {
                    if (document.getElementById("micBtn").innerText === "Listening...") {
                        document.getElementById("micBtn").innerText = "Speak Question";
                    }
                };

                recognition.start();
            }
            </script>
            """,
            height=60,
        )

        # ---- Pre-fill question box from mic transcript ----
        transcript = st.query_params.get("transcript", "")
        if transcript:
            st.session_state.qbox = transcript
            st.query_params.clear()

        # ---- Bottom bar: question input + Ask button ----
        col_input, col_button = st.columns([6, 1], vertical_alignment="bottom")

        with col_input:
            question = st.text_input(
                "Ask anything from the syllabus",
                key="qbox",
                placeholder="Ask anything from the syllabus...",
                label_visibility="collapsed",
            )

        with col_button:
            asked = st.button("Ask", key="ask_btn", type="primary", use_container_width=True)

        # ---- Generate answer only on Ask click ----
        if asked and question.strip():
            start_retrieval = time.perf_counter()
            docs = db.similarity_search(question, k=3)
            retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
            You are a syllabus assistant.
            Answer only using the provided context.

            Context:
            {context}

            Question:
            {question}
            """

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

            start_gen = time.perf_counter()
            with st.spinner("Thinking..."):
                response = llm.invoke(prompt)
            gen_s = time.perf_counter() - start_gen

            st.session_state.history.append(
                {
                    "question": question,
                    "answer": response.content,
                    "chunks": len(docs),
                    "retrieval_ms": retrieval_ms,
                    "gen_s": gen_s,
                    "total_s": gen_s + retrieval_ms / 1000,
                }
            )
            st.session_state.stats["answered"] += 1
            st.session_state.stats["last_time"] = gen_s + retrieval_ms / 1000
            st.session_state.pending_clear_qbox = True
            st.rerun()

# ---- Footer ----
st.markdown(
    """
    <footer class="scholar-footer">
        <div class="f-brand">
            <div class="nav-logo sm">SC</div>
            <div>ScholarChat AI</div>
        </div>
        <p class="f-tag">Turn any syllabus PDF into a smart study buddy you can chat with.</p>

        <div class="footer-grid">
            <div class="f-col">
                <div class="f-col-title">Creator</div>
                <a href="https://github.com/Seelam-Mohith" target="_blank">github.com/Seelam-Mohith</a>
                <p>Built by Seelam Mohith</p>
            </div>
            <div class="f-col">
                <div class="f-col-title">Report an issue</div>
                <a href="https://github.com/Seelam-Mohith/ScholarChatAI/issues" target="_blank">
                    github.com/Seelam-Mohith/ScholarChatAI/issues
                </a>
                <p>Found a bug or have a feature idea? Open an issue.</p>
            </div>
            <div class="f-col">
                <div class="f-col-title">Powered by</div>
                <p>Streamlit &middot; LangChain &middot; Google Gemini &middot; In-memory vector store</p>
            </div>
            <div class="f-col">
                <div class="f-col-title">Why you can trust it</div>
                <p>Every answer is grounded in your uploaded document via RAG to reduce hallucinations.</p>
            </div>
        </div>

        <div class="footer-bottom">
            <span>&copy; 2026 ScholarChat AI &middot; Open source</span>
            <span>Chunking: 1200 tokens &middot; overlap 200 &middot; Top-3 retrieval</span>
        </div>
    </footer>
    """,
    unsafe_allow_html=True,
)