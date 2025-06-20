from langchain_core.documents import Document
from langchain_pinecone import PineconeRerank

client = OpenAI(api_key=OPENAI_API_KEY)
pine_index = pc.Index(INDEX_NAME)
reranker = PineconeRerank(model="bge-reranker-v2-m3")

def retrieve_chunks(query: str, namespace: str = "semantic", use_rerank: bool = True, top_k: int = 10) -> list[Document]:
    q_emb = client.embeddings.create(input=[query], model="text-embedding-ada-002").data[0].embedding
    results = pine_index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=namespace)
    docs = [Document(page_content=r["metadata"]["text"], metadata=r["metadata"]) for r in results["matches"]]
    return reranker.compress_documents(docs, query) if use_rerank else docs

