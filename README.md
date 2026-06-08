# ScholarChatAI

ScholarChatAI is an AI-powered document assistant that enables students to interact with their academic materials through natural language. By combining Retrieval-Augmented Generation (RAG) with Google's Gemini models, the application allows users to upload PDF documents and ask questions directly about their content.

The system retrieves relevant information from the uploaded document and generates context-aware responses, helping students understand topics, revise concepts, and quickly locate important information.

## Overview

ScholarChatAI follows a lightweight RAG pipeline optimized for simplicity and efficiency. Users upload a PDF, the document is processed and converted into semantic embeddings, and relevant content is retrieved whenever a question is asked. Responses are generated using Gemini 2.5 Flash based only on the retrieved document context.

Unlike traditional chatbots, ScholarChatAI grounds its responses in the uploaded document, reducing hallucinations and improving answer relevance.

## Features

* Upload and analyze PDF documents
* Natural language question answering
* Retrieval-Augmented Generation (RAG) workflow
* Semantic search using vector embeddings
* Context-aware responses powered by Gemini 2.5 Flash
* In-memory vector storage for fast processing
* Session-based caching to avoid redundant embedding generation
* Automatic PDF processing and cleanup

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

## Technology Stack

### Frontend

* Streamlit

### AI & RAG

* Google Gemini 2.5 Flash
* Google Generative AI Embeddings
* LangChain

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

### Vector Storage

* InMemoryVectorStore

## How It Works

1. A user uploads a PDF document through the Streamlit interface.
2. The document is temporarily stored and loaded using PyPDFLoader.
3. The extracted text is divided into manageable chunks.
4. Gemini Embeddings convert each chunk into vector representations.
5. Vectors are stored in an in-memory vector database.
6. When a question is asked, the system retrieves the most relevant chunks using similarity search.
7. The retrieved context is combined with the user's query and sent to Gemini 2.5 Flash.
8. The generated answer is displayed in the application.

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

The application will launch in your browser, allowing you to upload a PDF and begin asking questions immediately.

## Future Enhancements

* Study plan generation
* Question paper generation
* Quiz and MCQ generation
* Important topic extraction
* Multi-document support
* Chat history and conversation memory
* Cloud deployment and authentication

## Author

**Seelam Mohith**

GitHub: https://github.com/Seelam-Mohith
