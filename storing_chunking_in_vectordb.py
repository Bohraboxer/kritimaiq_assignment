import os
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from chunking.fixed_chunking import fixed_chunking_
from chunking.semantic_chunking import semantic_chunking
from chunking.hierarchical_chunking import hierarchical_chunking
from chunking.mar import custom_chunking
from utils.pdf_loader import load_pdf_text


# laoding the apikey from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rag-chunking-eval"
client = OpenAI()


pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

index = pc.Index(INDEX_NAME)

# upload function
def store_chunks(chunks, namespace):
    print(f" Uploading {len(chunks)} chunks to namespace '{namespace}'")
    for i, chunk in enumerate(tqdm(chunks, desc=namespace)):
        embedding = client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding
        vector = {
            "id": f"{namespace}-{i}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                **chunk.get("metadata", {})
            }
        }
        index.upsert(vectors=[vector], namespace=namespace)

# main function
if __name__ == "__main__":
    file_name = "/Users/rohitbohra/Desktop/assignment/data/Test.pdf"
    text = load_pdf_text(file_name)

    strategies = {
    "fixed": [{"text": c, "metadata": {}} for c in fixed_chunking_(text)],
    "semantic": [{"text": c, "metadata": {}} for c in semantic_chunking(text)],
    "hierarchical": hierarchical_chunking(text),  
    "custom": custom_chunking(file_name)                 
    }

    for name, chunks in strategies.items():
        store_chunks(chunks, namespace=name)
