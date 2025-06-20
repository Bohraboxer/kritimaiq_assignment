from utils.pdf_loader import load_pdf_text
from chunking.fixed_chunking import fixed_chunking_
from chunking.semantic_chunking import semantic_chunking
from chunking.hierarchical_chunking import hierarchical_chunking
from chunking.custom_chunking import custom_chunking


if __name__ == "__main__":
    text = load_pdf_text("/Users/rohitbohra/Desktop/assignment/data/test.pdf")
    chunks = fixed_chunking_(text, chunk_size=512, overlap=128)

    print(f"Generated {len(chunks)} chunks.")
    print("\n--- Sample Chunk ---\n")
    print(chunks[0])
    print("*"*20)

    # Run semantic chunker
    semantic_chunks = semantic_chunking(text, chunk_size=100, overlap=2)

    print(f"Generated {len(semantic_chunks)} semantic chunks.")
    print("\n--- Sample Semantic Chunk ---\n")
    print(semantic_chunks[0])
    print("*"*20)

    hier_chunks = hierarchical_chunking(text)
    print(f"Generated {len(hier_chunks)} hierarchical chunks.\n")
    print("--- Sample ---")
    print(f"Title: {hier_chunks[1]['title']}")
    print(f"Level: {hier_chunks[1]['level']}")
    print(f"Text: {hier_chunks[1]['text']}")
    print("*"*20)

    chunks = custom_chunking(text, min_tokens=30)

    print(f"Created {len(chunks)} custom chunks with section metadata.\n")
    
    if chunks:
        print("--- Sample Chunk ---")
        print("Metadata:", chunks[0]['metadata'])
        print("Text:\n", chunks[0]['text'])
    else:
        print("No valid chunks were created.")