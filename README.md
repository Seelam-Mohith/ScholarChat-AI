# ScholarChat AI

ScholarChat AI is an AI-powered academic assistant that helps students interact with syllabus documents using natural language. Users can upload a PDF syllabus and ask questions through text or voice, receiving answers grounded in the document content.

The application uses Retrieval-Augmented Generation (RAG) with Google's Gemini models to retrieve relevant information from uploaded PDFs and generate context-aware responses.

## Features

* Upload and analyze PDF documents
* Ask questions using text or voice input
* Retrieval-Augmented Generation (RAG)
* Semantic search using Gemini embeddings
* Context-aware responses powered by Gemini 2.5 Flash
* In-memory vector storage for fast retrieval
* Session-based caching to avoid repeated processing
* Automatic temporary file cleanup

## Overview

ScholarChat AI follows a lightweight RAG pipeline:

1. Upload a PDF document.
2. Extract text using PyPDFLoader.
3. Split content into manageable chunks.
4. Generate embeddings using Gemini Embeddings.
5. Store embeddings in an in-memory vector database.
6. Retrieve relevant content through similarity search.
7. Generate answers using Gemini 2.5 Flash based on the retrieved context.

This approach helps reduce hallucinations by grounding responses in the uploaded document.

## System Architecture

```text
PDF Upload
     │
     ▼
PyPDFLoader
     │
     ▼
Text Chunking
     │
     ▼
Gemini Embeddings
     │
     ▼
InMemoryVectorStore
     │
     ▼
Similarity Search
     │
     ▼
Gemini 2.5 Flash
     │
     ▼
Answer Generation
```

## Voice Assistant Workflow

```text
User Speech
     │
     ▼
Web Speech API
     │
     ▼
Speech-to-Text Conversion
     │
     ▼
Question Input
     │
     ▼
Similarity Search
     │
     ▼
Gemini 2.5 Flash
     │
     ▼
Answer Generation
```

The voice assistant uses the browser's Web Speech API to convert spoken questions into text, allowing users to interact with the application hands-free.

## Technology Stack

### Frontend

* Streamlit
* HTML
* JavaScript

### AI & RAG

* Google Gemini 2.5 Flash
* Google Generative AI Embeddings
* LangChain

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

### Vector Storage

* InMemoryVectorStore

### Voice Input

* Web Speech API

## Installation

Clone the repository:

```bash
git clone https://github.com/Seelam-Mohith/ScholarChatAI.git
cd ScholarChatAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the application in your browser, upload a syllabus PDF, and begin asking questions.

## Future Enhancements

* Multi-document support
* Quiz and MCQ generation
* Study plan generation
* Chat history and conversation memory
* Important topic extraction
* Cloud deployment and authentication

## Author

**Seelam Mohith**

GitHub: https://github.com/Seelam-Mohith
