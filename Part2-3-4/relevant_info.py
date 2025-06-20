from openai import OpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeRerank
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError(
        "API keys not found. Ensure OPENAI_API_KEY and PINECONE_API_KEY are set in your .env."
    )
# Initialize clients
client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-chunking-eval")

# Optional: set up a reranker
reranker = PineconeRerank(model="bge-reranker-v2-m3")

def extract_relevant_chunks(query: str, top_k: int = 10, rerank_top_n: int = 5, rerank: bool = False):
    """
    Given a query string, returns top_k relevant chunks from Pinecone.
    
    If rerank=True, it returns the reranked top_n results using PineconeRerank.
    """
    # 1. Embed the query
    resp = client.embeddings.create(input=[query], model="text-embedding-ada-002")
    q_emb = resp.data[0].embedding

    # 2. Retrieve candidates
    results = index.query(vector=q_emb, top_k=top_k, namespace="semantic", include_metadata=True)
    docs = [
        Document(page_content=match['metadata']['text'], metadata=match['metadata'])
        for match in results.matches
    ]

    # 3. Rerank if enabled
    if rerank:
        docs = reranker.compress_documents(docs, query, top_n=rerank_top_n)

    # 4. Print retrieved passages
    for i, doc in enumerate(docs, start=1):
        print(f"\n[{i}] ID: {doc.metadata.get('chunk_id', 'N/A')} | Section: {doc.metadata.get('section', 'N/A')}\n")
        print(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

    return docs

# Example usage
if __name__ == "__main__":
    query = "What is a tangent and its property regarding radius?"
    relevant = extract_relevant_chunks(query, top_k=4, rerank_top_n=2, rerank=False)
