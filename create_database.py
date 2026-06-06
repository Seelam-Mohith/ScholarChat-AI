from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os
import shutil

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Delete old database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    db.persist()

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
    main()
