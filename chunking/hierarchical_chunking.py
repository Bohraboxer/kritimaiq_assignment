import re

def hierarchical_chunking(text: str) -> list[dict]:
    """
    Splits text into sections and subsections based on numbered headings.
    Returns a list of chunks with heading metadata.
    """
    # Regex to capture headings like "1. Introduction" or "1.1 Background"
    heading_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+([^\n]+)", re.MULTILINE)

    chunks = []
    last_index = 0
    last_heading = {"title": "Preamble", "level": "0", "start": 0}

    matches = list(heading_pattern.finditer(text))

    for idx, match in enumerate(matches):
        heading_number = match.group(1)
        heading_title = match.group(2)
        start = match.start()

        if idx > 0:
            prev_chunk_text = text[last_index:start].strip()
            if prev_chunk_text:
                chunks.append({
                    "title": last_heading["title"],
                    "level": last_heading["level"],
                    "text": prev_chunk_text
                })

        last_heading = {"title": heading_title, "level": heading_number, "start": start}
        last_index = start

    # Add final chunk
    final_chunk = text[last_index:].strip()
    if final_chunk:
        chunks.append({
            "title": last_heading["title"],
            "level": last_heading["level"],
            "text": final_chunk
        })

    return chunks
