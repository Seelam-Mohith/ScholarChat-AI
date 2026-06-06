import streamlit as st
import tempfile

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

load_dotenv()

st.set_page_config(page_title="PaperPilot AI")

st.title("PaperPilot AI")
st.write("Upload a syllabus PDF and ask questions.")

uploaded_file = st.file_uploader(
    "Upload Syllabus PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # Vector Store
    db = Chroma.from_documents(
        chunks,
        embeddings
    )

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