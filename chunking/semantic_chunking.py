import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Download the model if it’s missing
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def semantic_chunking(text: str, chunk_size: int = 300, overlap: int = 5) -> list[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_length = len(sentence.split())
        if current_length + token_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            # Start next chunk with some overlap
            overlap_sentences = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_sentences[:]
            current_length = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += token_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
