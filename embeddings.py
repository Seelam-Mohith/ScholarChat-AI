from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))

    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5

    return dot_product / (norm1 * norm2)


def main():
    # Create Gemini embedding model
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # Generate embedding
    vector = embedding_function.embed_query("apple")

    print("Vector for 'apple':")
    print(vector[:10])  # First 10 values only
    print(f"\nVector length: {len(vector)}")

    # Compare two words
    word1 = "apple"
    word2 = "iphone"

    embedding1 = embedding_function.embed_query(word1)
    embedding2 = embedding_function.embed_query(word2)

    similarity = cosine_similarity(embedding1, embedding2)

    print(f"\nSimilarity between '{word1}' and '{word2}':")
    print(similarity)


if __name__ == "__main__":
    main()