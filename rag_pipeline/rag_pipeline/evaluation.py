import json
from typing import List
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rag_pipeline.retrieval import retrieve_chunks
from rag_pipeline.qa import generate_answer_with_context
from rag_pipeline.config import OPENAI_API_KEY

def evaluate_similarity(answer: str, ground_truth: str) -> float:
    client = OpenAI(api_key=OPENAI_API_KEY)
    a_emb = client.embeddings.create(input=[answer], model="text-embedding-ada-002").data[0].embedding
    g_emb = client.embeddings.create(input=[ground_truth], model="text-embedding-ada-002").data[0].embedding
    return float(cosine_similarity([a_emb], [g_emb])[0][0])

def evaluate_from_json(json_path: str, namespace: str) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)

    for item in data:
        question = item["question"]
        ground_truths: List[str] = item["answers"]

        context_docs = retrieve_chunks(query=question, namespace=namespace)
        answer = generate_answer_with_context(question=question, context_docs=context_docs)

        print(f"\nQ: {question}\n🔍 Answer: {answer}")
        for truth in ground_truths:
            score = evaluate_similarity(answer, truth)
            hallucinated = score < 0.75
            print(f"  ✓ GT: {truth} | Cosine Sim: {score:.3f} | Hallucinated: {hallucinated}")
