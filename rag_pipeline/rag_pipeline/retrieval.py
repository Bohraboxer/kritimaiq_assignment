from openai import OpenAI
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_pinecone import PineconeRerank
from rag_pipeline.config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME_PIPELINE

def retrieve_chunks(query: str, namespace: str, use_rerank: bool = True, top_k: int = 10) -> list[Document]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME_PIPELINE)
    reranker = PineconeRerank(model="bge-reranker-v2-m3")

    q_emb = client.embeddings.create(input=[query], model="text-embedding-ada-002").data[0].embedding
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=namespace)
    docs = [Document(page_content=r["metadata"]["text"], metadata=r["metadata"]) for r in results["matches"]]
    return reranker.compress_documents(docs, query) if use_rerank else docs

def generate_answer_with_context(question: str, context_docs: list, model: str = "gpt-3.5-turbo") -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.\n

Context:
{context}
\n
Question:
{question}
\n
Answer:"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Use only the context provided."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
