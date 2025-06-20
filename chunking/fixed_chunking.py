import tiktoken

def fixed_chunking_(text: str, chunk_size: int = 300, overlap: int = 48) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")  # Same tokenizer used in OpenAI models
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append(enc.decode(chunk))
        start += chunk_size - overlap

    return chunks
