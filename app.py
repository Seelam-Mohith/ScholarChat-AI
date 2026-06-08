import streamlit as st
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

st.set_page_config(page_title="PaperPilot AI")

st.title("ScholarChat AI")
st.write("Upload a syllabus PDF and ask questions.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.file_name = None

uploaded_file = st.file_uploader(
    "Upload Syllabus PDF",
    type="pdf"
)

if uploaded_file:
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.vector_store = None
        st.session_state.file_name = uploaded_file.name

    if st.session_state.vector_store is None:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Use larger chunks to reduce embedding requests.
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            # Gemini Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001"
            )

            # Keep retrieval in memory for the uploaded PDF.
            st.session_state.vector_store = InMemoryVectorStore.from_documents(
                chunks,
                embeddings
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

    question = st.text_input(
        "Ask anything from the syllabus"
    )

    if question:

        docs = db.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
        You are a syllabus assistant.

        Answer only using the provided context.

        Context:
        {context}

        Question:
        {question}
        """

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)
