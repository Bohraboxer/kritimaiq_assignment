import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-chunking-eval"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def store_chunks(chunks: list[dict], namespace: str):
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

