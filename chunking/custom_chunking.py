import re
from typing import List, Dict


def split_by_sections(text: str) -> Dict[str, str]:
    sections = text.split('## **')
    result = {}
    for section in sections[1:]:
        if '\n' in section:
            title, content = section.split('\n', 1)
            clean_title = title.strip('* ').strip()
            result[clean_title] = content.strip()
    return result


def clean_paragraphs(paragraphs: List[str], min_tokens: int = 30) -> List[str]:
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if re.match(r"^(Fig\.|Reprint|MATHEMA|CIRCLES)", p):
            continue
        if len(p.split()) < min_tokens:
            continue
        cleaned.append(p)
    return cleaned


def split_into_paragraphs(text: str) -> List[str]:
    return re.split(r'\n\s*\n', text)


def split_into_sentences(text: str) -> List[str]:
    # Naive sentence tokenizer — use nltk if needed
    return re.split(r'(?<=[.!?]) +', text)


def group_sentences(sentences: List[str], max_tokens: int = 60, stride: int = 40) -> List[str]:
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = []
        token_count = 0
        j = i
        while j < len(sentences) and token_count + len(sentences[j].split()) <= max_tokens:
            chunk.append(sentences[j])
            token_count += len(sentences[j].split())
            j += 1
        if chunk:
            chunks.append(' '.join(chunk))
        i += stride  # overlap control
    return chunks


def custom_chunking(text: str, min_tokens: int = 30, max_tokens: int = 60, stride: int = 40) -> List[Dict]:
    all_chunks = []
    sections = split_by_sections(text)

    if not sections:
        sections = {"Default": text}

    for section_title, content in sections.items():
        paragraphs = split_into_paragraphs(content)
        paragraphs = clean_paragraphs(paragraphs, min_tokens=min_tokens)

        for i, para in enumerate(paragraphs):
            sentences = split_into_sentences(para)
            subchunks = group_sentences(sentences, max_tokens=max_tokens, stride=stride)

            for j, sub in enumerate(subchunks):
                chunk = {
                    "text": sub.strip(),
                    "metadata": {
                        "section": section_title,
                        "chunk_id": f"{section_title}_{i+1}_{j+1}"
                    }
                }
                all_chunks.append(chunk)

    return all_chunks
