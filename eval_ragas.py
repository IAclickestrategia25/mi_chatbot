import os
import time
import json
import csv
from typing import List, Dict, Any, Optional

import requests
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy


# =========================
# CONFIGURACIÓN
# =========================
BASE_URL = "https://mi-chatbot-ynz8.onrender.com"
LOGIN_URL = f"{BASE_URL}/login"
ASK_URL = f"{BASE_URL}/api/ask/"

# Pega aquí tu password de USER o ADMIN (con USER vale para preguntar)
PASSWORD = os.getenv("CHATBOT_PASSWORD", "PON_AQUI_TU_PASSWORD")

# Preguntas de evaluación. Recomendación: que sean preguntas que seguro existen en los documentos.
QUESTIONS = [
    "Como funciona el sistema de automatizacion",
    "Que es una guia de cooperativistas",
    # añade más preguntas reales
]

# Parámetros del endpoint
N_RESULTS = 15
DISTANCE_THRESHOLD = 1.2

# Timeouts y reintentos (Render + OpenAI + Chroma pueden tardar)
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_RETRIES = 3

# Recortes para evitar problemas de longitud (max_tokens) en RAGAS
MAX_CONTEXTS = 3            # número máximo de fragmentos
MAX_CONTEXT_CHARS = 2000    # recorte por fragmento
MAX_ANSWER_CHARS = 1500     # recorte de la respuesta

# Salidas
OUT_JSONL = "ragas_rows.jsonl"
OUT_CSV = "ragas_results.csv"


# =========================
# UTILIDADES
# =========================
def trim(text: str, n: int) -> str:
    text = (text or "").strip()
    return text[:n]


def login_session() -> requests.Session:
    """
    Tu backend requiere cookie (Depends(require_auth)).
    Se hace login y se conserva la cookie en una sesión requests.Session().
    """
    if not PASSWORD or PASSWORD == "PON_AQUI_TU_PASSWORD":
        raise RuntimeError(
            "Falta PASSWORD. Ponla en el script o define la variable de entorno CHATBOT_PASSWORD."
        )

    s = requests.Session()
    # El frontend usa FormData; aquí enviamos form-encoded equivalente.
    r = s.post(LOGIN_URL, data={"password": PASSWORD}, timeout=60)
    r.raise_for_status()
    data = r.json()

    if not data.get("ok"):
        raise RuntimeError(f"Login fallido: {data}")

    return s


def ask_chatbot(session: requests.Session, question: str) -> Dict[str, Any]:
    """
    Llama a /api/ask/ con reintentos y timeout alto.
    """
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
            # backoff simple
            time.sleep(2 * (i + 1))

    raise last_err if last_err else RuntimeError("Fallo desconocido en ask_chatbot")


def warmup(session: requests.Session) -> None:
    """
    “Despierta” Render y calienta conexiones antes de evaluar en serio.
    """
    try:
        ask_chatbot(session, "hola")
    except Exception:
        pass


def build_rows(session: requests.Session, questions: List[str]) -> List[Dict[str, Any]]:
    """
    Ejecuta preguntas contra tu API y construye filas para RAGAS:
    question / answer / contexts.
    Recorta contextos y respuesta para evitar max_tokens.
    """
    rows: List[Dict[str, Any]] = []

    for q in questions:
        data = ask_chatbot(session, q)

        contexts = data.get("fragmentos_usados") or []
        # Solo evaluamos “RAG real”: si no hay fragmentos usados, no hay contexto.
        if not contexts:
            print(f"[SKIP] Sin fragmentos usados para: {q}")
            continue

        contexts = contexts[:MAX_CONTEXTS]
        contexts = [trim(c, MAX_CONTEXT_CHARS) for c in contexts]

        answer = trim(data.get("respuesta", ""), MAX_ANSWER_CHARS)

        row = {
            "question": q,
            "answer": answer,
            "contexts": contexts,
            # extras útiles para auditoría (no los usa RAGAS, pero los guardamos):
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
    """
    Guarda el dataframe de resultados (pandas) a CSV en UTF-8.
    """
    df.to_csv(path, index=False, encoding="utf-8")


# =========================
# MAIN
# =========================
def main():
    # 1) Seguridad: OpenAI API key en local (RAGAS la necesita para evaluar)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta OPENAI_API_KEY en tu entorno local. "
            "En PowerShell: $env:OPENAI_API_KEY='sk-...'; python eval_ragas.py"
        )

    # 2) Login + warmup
    session = login_session()
    warmup(session)

    # 3) Generar dataset (pregunta/answer/contexts)
    rows = build_rows(session, QUESTIONS)
    if not rows:
        raise RuntimeError(
            "No hay filas evaluables. "
            "Revisa QUESTIONS: deben devolver fragmentos_usados (contextos) en tu API."
        )

    # 4) Guardar dataset bruto (para auditoría)
    save_jsonl(rows, OUT_JSONL)
    print(f"[OK] Dataset guardado: {OUT_JSONL} ({len(rows)} filas)")

    # 5) Dataset para RAGAS (solo los campos que usa)
    ragas_rows = [{"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]} for r in rows]
    dataset = Dataset.from_list(ragas_rows)

    # 6) Evaluación
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,        # fidelidad (si lo dicho está soportado por contexts)
            answer_relevancy,    # relevancia de la respuesta respecto a la pregunta
        ],
    )

    # 7) Mostrar resultados agregados
    print("\n=== MÉTRICAS AGREGADAS ===")
    print(results)

    # 8) Tabla por fila
    df = results.to_pandas()
    print("\n=== RESULTADOS POR FILA ===")
    print(df)

    # 9) Guardar CSV
    save_results_csv(df, OUT_CSV)
    print(f"\n[OK] Resultados guardados: {OUT_CSV}")


if __name__ == "__main__":
    main()
