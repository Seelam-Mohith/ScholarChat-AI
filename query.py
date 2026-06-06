import argparse

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context:

{question}
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query_text",
        type=str,
        help="The query text."
    )

    args = parser.parse_args()
    query_text = args.query_text

    # Gemini Embeddings
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search Chroma
    results = db.similarity_search_with_relevance_scores(
        query_text,
        k=3
    )

    if len(results) == 0:
        print("No matching results found.")
        return

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results]
    )

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )

    # Gemini Model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    response = model.invoke(prompt)

    sources = [
        doc.metadata.get("source", None)
        for doc, _score in results
    ]

    print("\nResponse:")
    print(response.content)

    print("\nSources:")
    print(sources)


if __name__ == "__main__":
    main()
