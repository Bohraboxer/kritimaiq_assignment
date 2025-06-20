from rag_pipeline.config import INDEX_NAME_PIPELINE
from rag_pipeline.chunking import semantic_chunking
from rag_pipeline.vector_store import store_chunks, load_pdf_text
from rag_pipeline.evaluation import evaluate_from_json

# Step 1: Load PDF and chunk
text = load_pdf_text("data/Test.pdf")
chunks = semantic_chunking(text)
chunk_docs = [{"text": chunk, "metadata": {}} for chunk in chunks]

# Step 2: Store in Pinecone
store_chunks(chunk_docs, namespace="semantic")

# Step 3: Evaluate using query set
evaluate_from_json("/Users/rohitbohra/Desktop/assignment/rag_pipeline/data/queries.json", namespace="semantic")
