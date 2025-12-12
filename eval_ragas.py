import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

API_URL = "https://mi-chatbot-ynz8.onrender.com/api/ask/"

QUESTIONS = [
    "¿Qué información aparece en los documentos sobre X?",
    "¿Qué procedimiento se describe en los archivos?",
]

def ask_chatbot(question: str):
    response = requests.post(
        API_URL,
        json={
            "question": question,
            "n_results": 15,
            "distance_threshold": 1.2
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()

rows = []

for q in QUESTIONS:
    data = ask_chatbot(q)

    # Solo evaluamos respuestas RAG reales
    if not data.get("fragmentos_usados"):
        continue

    rows.append({
        "question": q,
        "answer": data.get("respuesta", ""),
        "contexts": data.get("fragmentos_usados", [])
    })

dataset = Dataset.from_list(rows)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ]
)

print(results)
print(results.to_pandas())
