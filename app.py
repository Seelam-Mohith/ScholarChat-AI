import streamlit as st
import streamlit.components.v1 as components
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai._common import GoogleGenerativeAIError

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

load_dotenv()

st.set_page_config(page_title="ScholarChat AI", page_icon="🎓")

st.title("ScholarChat AI")
st.write("Upload a syllabus PDF and ask questions.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.file_name = None

uploaded_file = st.file_uploader("Upload Syllabus PDF", type="pdf")

if uploaded_file:
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.vector_store = None
        st.session_state.file_name = uploaded_file.name

    if st.session_state.vector_store is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
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
    st.success("PDF processed successfully!")

    # --- Mic Button (Web Speech API) ---
    components.html("""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
            <button onclick="startListening()" id="micBtn"
                style="padding:9px 18px; border-radius:8px; background:#e74c3c;
                       color:white; border:none; cursor:pointer; font-size:15px;">
                🎤 Speak Question
            </button>
            <span id="status" style="color:gray; font-size:13px;">
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
                document.getElementById("micBtn").innerText = "🎤 Speak Question";

                // Pass transcript back to Streamlit via URL
                const url = new URL(window.parent.location.href);
                url.searchParams.set("transcript", transcript);
                window.parent.location.href = url.toString();
            };

            recognition.onerror = (e) => {
                let msg = "❌ Error: " + e.error;
                if (e.error === "network") {
                    msg += " — Check your internet connection. Firewalls/VPNs may block speech recognition.";
                } else if (e.error === "not-allowed") {
                    msg += " — Please allow microphone access in your browser.";
                } else if (e.error === "no-speech") {
                    msg += " — No speech detected. Try speaking louder or closer to the mic.";
                }
                document.getElementById("status").innerText = msg;
                document.getElementById("micBtn").innerText = "🎤 Speak Question";
            };

            recognition.onend = () => {
                if (document.getElementById("micBtn").innerText === "⏳ Listening...") {
                    document.getElementById("micBtn").innerText = "🎤 Speak Question";
                }
            };

            recognition.start();
        }
        </script>
    """, height=60)

    # --- Get transcript from URL if mic was used ---
    transcript = st.query_params.get("transcript", "")

    question = st.text_input(
        "Ask anything from the syllabus",
        value=transcript,
        placeholder="Type or speak your question above..."
    )

    # Clear transcript from URL after it's loaded into the input
    if transcript:
        st.query_params.clear()

    if question:
        docs = db.similarity_search(question, k=3)

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

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)