import os
import time
import json
from typing import List, Dict, Any, Optional

import requests
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# -------------------------
# CONFIG (ALINEADO CON BACKEND)
# -------------------------
BASE_URL = "https://mi-chatbot-ynz8.onrender.com"
LOGIN_URL = f"{BASE_URL}/login"
ASK_URL = f"{BASE_URL}/api/ask/"

PASSWORD = os.getenv("CHATBOT_PASSWORD", "PON_AQUI_TU_PASSWORD")

QUESTIONS = [
    "Como funciona el sistema de automatizacion",
    "Que es una guia de cooperativistas",
    "puedo hacer lo que quiera en una cooperativa"
    "las automatizaciones sirven para cualquier cosa"
]

# Parámetros del endpoint (mismos por defecto que tu backend)
N_RESULTS = 15
DISTANCE_THRESHOLD = 1.2

REQUEST_TIMEOUT_SECONDS = 180
REQUEST_RETRIES = 3

# Límites alineados con tu backend (ver main.py corregido)
MAX_CONTEXTS = 3                 # MAX_CONTEXT_FRAGMENTS
MAX_CONTEXT_CHARS_TOTAL = 6000   # MAX_CONTEXT_CHARS_BACKEND (total)
MAX_ANSWER_CHARS = 2500          # un poco más generoso, pero seguro para RAGAS

OUT_JSONL = "ragas_rows.jsonl"
OUT_CSV = "ragas_results.csv"

# -------------------------
# UTILIDADES
# -------------------------
def trim(text: str, n: int) -> str:
    text = (text or "").strip()
    return text[:n]

def join_and_trim_contexts(contexts: List[str]) -> List[str]:
    """
    - Limita a MAX_CONTEXTS
    - Aplica recorte global (MAX_CONTEXT_CHARS_TOTAL) al conjunto (como hace el backend)
    """
    contexts = [c.strip() for c in (contexts or []) if c and c.strip()]
    contexts = contexts[:MAX_CONTEXTS]

    # Recorte global del “contexto total”
    total = "\n\n".join(contexts)
    total = trim(total, MAX_CONTEXT_CHARS_TOTAL)

    # Re-separa en lista: a RAGAS le da igual que sean 1 o N, mientras sea lista de strings
    # Mantengo como 1 bloque para minimizar tokens en métricas y evitar truncados.
    return [total] if total else []

def login_session() -> requests.Session:
    if not PASSWORD or PASSWORD == "PON_AQUI_TU_PASSWORD":
        raise RuntimeError(
            "Falta PASSWORD. Ponla en el script o define CHATBOT_PASSWORD."
        )

    s = requests.Session()
    r = s.post(LOGIN_URL, data={"password": PASSWORD}, timeout=60)
    r.raise_for_status()
    data = r.json()

    if not data.get("ok"):
        raise RuntimeError(f"Login fallido: {data}")
    return s

def ask_chatbot(session: requests.Session, question: str) -> Dict[str, Any]:
    payload = {
        "question": question,
        "n_results": N_RESULTS,
        "distance_threshold": DISTANCE_THRESHOLD,
    }

    last_err: Optional[Exception] = None
    for i in range(REQUEST_RETRIES):
        try:
            r = session.post(ASK_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)

            if r.status_code in (401, 403):
                raise RuntimeError(f"Auth error {r.status_code}: {r.text}")

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            time.sleep(2 * (i + 1))

    raise last_err if last_err else RuntimeError("Fallo desconocido en ask_chatbot")

def warmup(session: requests.Session) -> None:
    try:
        ask_chatbot(session, "hola")
    except Exception:
        pass

def build_rows(session: requests.Session, questions: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for q in questions:
        data = ask_chatbot(session, q)

        contexts_raw = data.get("fragmentos_usados") or []
        contexts = join_and_trim_contexts(contexts_raw)

        # Solo evaluamos si hay contexto (RAG real)
        if not contexts:
            print(f"[SKIP] Sin fragmentos usados para: {q}")
            continue

        answer = trim(data.get("respuesta", ""), MAX_ANSWER_CHARS)

        row = {
            "question": q,
            "answer": answer,
            "contexts": contexts,
            # auditoría
            "fuentes": data.get("fuentes", []),
            "distancias": data.get("distancias", []),
        }
        rows.append(row)

    return rows

def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_results_csv(df, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


# -------------------------
# RAGAS: LLM + EMBEDDINGS FORZADOS (evita embed_query / NaNs)
# -------------------------
def build_ragas_llm_and_embeddings():
    """
    Fuerza los componentes que RAGAS usa internamente para:
    - Faithfulness (LLM)
    - Answer relevancy (Embeddings + LLM)
    """
    # RAGAS suele integrarse muy bien vía LangChain
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as e:
        raise RuntimeError(
            "Faltan dependencias para forzar LLM/embeddings.\n"
            "Instala:\n"
            "  pip install -U langchain-openai\n"
            "Y vuelve a ejecutar.\n"
            f"Detalle: {e}"
        )

    # LLM para evaluación (no es el de tu chatbot; es el “juez” de RAGAS)
    # max_tokens alto para evitar truncados en faithfulness
    eval_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=4096,
        request_timeout=120,
    )

    # Embeddings para answer_relevancy
    eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return LangchainLLMWrapper(eval_llm), LangchainEmbeddingsWrapper(eval_embeddings)


# -------------------------
# MAIN
# -------------------------
def main():
    # 1) OpenAI API key (RAGAS la necesita)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta OPENAI_API_KEY.\n"
            "PowerShell:\n"
            "  $env:OPENAI_API_KEY='sk-...'\n"
            "  $env:CHATBOT_PASSWORD='...'\n"
            "  python eval_ragas.py"
        )

    # 2) Login + warmup
    session = login_session()
    warmup(session)

    # 3) Dataset bruto (pregunta/answer/contexts)
    rows = build_rows(session, QUESTIONS)
    if not rows:
        raise RuntimeError(
            "No hay filas evaluables. Revisa QUESTIONS: deben devolver fragmentos_usados."
        )

    save_jsonl(rows, OUT_JSONL)
    print(f"[OK] Dataset guardado: {OUT_JSONL} ({len(rows)} filas)")

    # 4) Dataset RAGAS
    ragas_rows = [{"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]} for r in rows]
    dataset = Dataset.from_list(ragas_rows)

    # 5) LLM/Embeddings forzados
    llm, embeddings = build_ragas_llm_and_embeddings()

    # 6) Evaluación
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n=== MÉTRICAS AGREGADAS ===")
    print(results)

    df = results.to_pandas()
    print("\n=== RESULTADOS POR FILA ===")
    print(df)

    save_results_csv(df, OUT_CSV)
    print(f"\n[OK] Resultados guardados: {OUT_CSV}")


if __name__ == "__main__":
    main()
