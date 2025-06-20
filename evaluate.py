import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Dict
from tqdm import tqdm
import json
import numpy as np

# loading .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-chunking-eval"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# test data for evaluation
test_data = [
    {
      "question": "What is the term for a line that has no common points with a circle?",
      "answers": ["non-intersecting line"],
      "difficulty": "easy",
      "section": "10.1 Introduction"
    },
    {
      "question": "How many tangents can be drawn to a circle from a point inside the circle?",
      "answers": ["zero"],
      "difficulty": "easy",
      "section": "10.3 Number of Tangents from a Point on a Circle"
    },
    {
      "question": "What is the relationship between a tangent and a secant as described in the text?",
      "answers": ["A tangent is a special case of a secant when the two endpoints of its corresponding chord coincide"],
      "difficulty": "medium",
      "section": "10.2 Tangent to a Circle"
    },
    {
      "question": "According to Theorem 10.1, what geometric property exists between the tangent at a point on a circle and the radius through that point?",
      "answers": ["The tangent is perpendicular to the radius through the point of contact"],
      "difficulty": "medium",
      "section": "10.2 Tangent to a Circle"
    },
    {
      "question": "In Activity 2, what is observed when drawing lines parallel to a secant of a circle?",
      "answers": ["The length of the chord cut by the lines decreases, eventually becoming zero, resulting in two tangents parallel to the secant"],
      "difficulty": "medium",
      "section": "10.2 Tangent to a Circle"
    },
    {
      "question": "Prove that the lengths of tangents drawn from an external point to a circle are equal, as stated in Theorem 10.2.",
      "answers": [
        "Given a circle with center O and two tangents PQ and PR from an external point P, join OP, OQ, and OR. Angles OQP and ORP are right angles (by Theorem 10.1). In right triangles OQP and ORP, OQ = OR (radii), OP = OP (common), so triangles OQP and ORP are congruent by RHS congruence. Thus, PQ = PR by CPCT."
      ],
      "difficulty": "hard",
      "section": "10.3 Number of Tangents from a Point on a Circle"
    },
    {
      "question": "In Example 1, what is proven about a chord of a larger circle that touches a smaller concentric circle?",
      "answers": ["The chord is bisected at the point of contact"],
      "difficulty": "medium",
      "section": "10.3 Number of Tangents from a Point on a Circle"
    },
    {
      "question": "In Example 3, if PQ is a chord of length 8 cm in a circle of radius 5 cm, and tangents at P and Q intersect at point T, what is the length of TP?",
      "answers": ["20/3 cm"],
      "difficulty": "hard",
      "section": "10.3 Number of Tangents from a Point on a Circle"
    },
    {
      "question": "What is the length of the tangent from a point Q to a circle if the distance from Q to the center is 25 cm and the radius of the circle is 7 cm? (Based on Exercise 10.2, Question 1)",
      "answers": ["24 cm"],
      "difficulty": "medium",
      "section": "EXERCISE 10.2"
    },
    {
      "question": "Prove that in a quadrilateral ABCD circumscribing a circle, the sum of opposite sides is equal (based on Exercise 10.2, Question 8).",
      "answers": [
        "Let the quadrilateral ABCD touch the circle at points P, Q, R, and S, where AB touches at P, BC at Q, CD at R, and DA at S. By Theorem 10.2, the lengths of tangents from a point to a circle are equal. Thus, AP = AS, BP = BQ, CR = CQ, and DR = DS. Therefore, AB + CD = (AP + PB) + (CR + RD) = (AS + BQ) + (CQ + DS) = AD + BC."
      ],
      "difficulty": "hard",
      "section": "EXERCISE 10.2"
    }
]

# function to check cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))

# generating answer by refering to relevant context
def generate_answer(question: str, context: List[str]) -> str:
    joined_context = "\n\n".join(context)
    prompt = (
        f"Use the following context to answer the question concisely.\n"
        f"\nContext:\n{joined_context}\n"
        f"\nQuestion: {question}\nAnswer:"
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful math tutor. Do not use your internal knoweldge to answer the question. Use only the context provided to answer the question"},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    return completion.choices[0].message.content.strip()

# retrieving info chunks from pinecone
def retrieve_chunks(question: str, namespace: str, k: int = 3) -> List[str]:
    embedding = client.embeddings.create(input=[question], model="text-embedding-ada-002").data[0].embedding
    results = index.query(vector=embedding, top_k=k, include_metadata=True, namespace=namespace)
    return [m["metadata"]["text"] for m in results["matches"]]

# cosine similarity score when comparing the generated answer with ground truth
def evaluate_similarity(reference_answers: List[str], generated: str) -> float:
    gen_emb = client.embeddings.create(input=[generated], model="text-embedding-ada-002").data[0].embedding
    scores = []
    for ref in reference_answers:
        ref_emb = client.embeddings.create(input=[ref], model="text-embedding-ada-002").data[0].embedding
        scores.append(cosine_similarity(ref_emb, gen_emb))
    return max(scores) if scores else 0.0

# calculating
def evaluate_with_generation(namespace: str, k: int = 5):
    results = []
    scores = []

    for item in tqdm(test_data, desc=f"Evaluating {namespace}"):
        question = item["question"]
        references = item["answers"]

        retrieved_chunks = retrieve_chunks(question, namespace, k)
        generated_answer = generate_answer(question, retrieved_chunks)
        sim_score = evaluate_similarity(references, generated_answer)

        scores.append(sim_score)
        results.append({
            "question": question,
            "expected": references,
            "generated": generated_answer,
            "similarity": sim_score,
            "chunks": retrieved_chunks
        })

    avg_score = round(sum(scores) / len(scores), 4)
    print(f"{namespace} average similarity score: {avg_score}")

    with open(f"generated_answers_{namespace}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return avg_score

# === Run for all chunking strategies ===
if __name__ == "__main__":
    strategy_scores = {}
    for strategy in ["fixed", "semantic", "hierarchical", "custom"]:
        score = evaluate_with_generation(strategy)
        strategy_scores[strategy] = score

    print("\nSummary of average cosine similarity scores:")
    for strat, score in sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{strat:<15}: {score:.4f}")
