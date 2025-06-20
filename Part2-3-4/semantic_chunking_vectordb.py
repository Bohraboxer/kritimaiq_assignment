import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from openai import OpenAI
from chunking.semantic_chunking import semantic_chunking
from utils.pdf_loader import load_pdf_text

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "semantic-chunking-eval"

# Initialize clients
openai.api_key = OPENAI_API_KEY
client = OpenAI()
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# function to upload the chunks to DB
def store_chunks(chunks, namespace):
    print(f"\n📦 Uploading {len(chunks)} chunks to namespace '{namespace}'")
    for i, chunk in enumerate(tqdm(chunks, desc=namespace)):
        embedding = client.embeddings.create(
            input=[chunk["text"]], model="text-embedding-ada-002"
        ).data[0].embedding

        vector = {
            "id": f"{namespace}-{i}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                **chunk.get("metadata", {})
            }
        }
        index.upsert(vectors=[vector], namespace=namespace)

# === Main logic ===
if __name__ == "__main__":
    file_path = "/Users/rohitbohra/Desktop/assignment/data/Test.pdf"
    raw_text = load_pdf_text(file_path)

    # Run semantic chunking
    semantic_chunks = semantic_chunking(raw_text)
    formatted_chunks = [{"text": chunk, "metadata": {}} for chunk in semantic_chunks]

    # Upload only semantic chunks
    store_chunks(formatted_chunks, namespace="semantic")
