from openai import OpenAI
from rag_pipeline.config import OPENAI_API_KEY

def generate_answer_with_context(question: str, context_docs: list, model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    context_lines = [f"[{i+1}] {doc.page_content.strip()}" for i, doc in enumerate(context_docs)]
    citations = "\n\n".join(context_lines)
    prompt = f"""You are a helpful assistant. Use only the context below to answer the question.

Context:
{citations}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only uses the context provided."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

