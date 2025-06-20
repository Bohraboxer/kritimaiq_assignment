from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from tqdm import tqdm
from pypdf import PdfReader
from rag_pipeline.config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME_PIPELINE

def store_chunks(chunks: list[dict], namespace: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME_PIPELINE not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME_PIPELINE,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(INDEX_NAME_PIPELINE)
    print(f"\n📦 Uploading {len(chunks)} chunks to namespace '{namespace}'")
    for i, chunk in enumerate(tqdm(chunks, desc=namespace)):
        embedding = client.embeddings.create(input=[chunk["text"]], model="text-embedding-ada-002").data[0].embedding
        vector = {
            "id": f"{namespace}-{i}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                **chunk.get("metadata", {})
            }
        }
        index.upsert(vectors=[vector], namespace=namespace)

def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text
