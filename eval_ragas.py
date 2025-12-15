import os
import time
import json
from typing import List, Dict, Any, Optional

import requests
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness

# Compatibilidad entre versiones de RAGAS:
# - algunas tienen answer_relevancy
# - otras answer_relevance
try:
    from ragas.metrics import answer_relevancy as _answer_rel_metric
except Exception:
    try:
        from ragas.metrics import answer_relevance as _answer_rel_metric
    except Exception:
        _answer_rel_metric = None


# =========================
# CONFIGURACIÓN
# =========================
BASE_URL = "https://mi-chatbot-ynz8.onrender.com"
LOGIN_URL = f"{BASE_URL}/login"
ASK_URL = f"{BASE_URL}/api/ask/"

# Password (mejor por variable de entorno)
PASSWORD = os.getenv("CHATBOT_PASSWORD", "PON_AQUI_TU_PASSWORD")

QUESTIONS = [
    "Como funciona el sistema de automatizacion",
    "Que es una guia de cooperativistas",
    # añade más preguntas reales
]

# Parámetros del endpoint
N_RESULTS = 15
DISTANCE_THRESHOLD = 1.2

# Timeouts y reintentos
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_RETRIES = 3

# Recortes (IMPORTANTE):
# - recortamos CONTEXTOS para evitar max_tokens en RAGAS
# - NO recortamos ANSWER para no penalizar por truncado artificial
MAX_CONTEXTS = 2            # recomendado: 2 o 3
MAX_CONTEXT_CHARS = 1500    # recomendado: 1200-2000
MAX_ANSWER_CHARS = None     # None = no recortar respuesta

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
    Ejecuta preguntas contra tu API y construye filas para RAGAS.
    - recorta CONTEXTOS para evitar max_tokens
    - NO recorta ANSWER (salvo que quieras)
    """
    rows: List[Dict[str, Any]] = []

    for q in questions:
        try:
            data = ask_chatbot(session, q)
        except Exception as e:
            print(f"[ERROR] Falló la pregunta '{q}': {e}")
            continue

        contexts = data.get("fragmentos_usados") or []
        if not contexts:
            print(f"[SKIP] Sin fragmentos usados para: {q}")
            continue

        # recortar contextos
        contexts = contexts[:MAX_CONTEXTS]
        contexts = [trim(c, MAX_CONTEXT_CHARS) for c in contexts]

        # NO recortar answer salvo que MAX_ANSWER_CHARS tenga un número
        answer_raw = (data.get("respuesta") or "").strip()
        answer = answer_raw if MAX_ANSWER_CHARS is None else trim(answer_raw, MAX_ANSWER_CHARS)

        row = {
            "question": q,
            "answer": answer,
            "contexts": contexts,
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


# =========================
# MAIN
# =========================
def main():
    # OpenAI key en local (RAGAS la usa para evaluar)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta OPENAI_API_KEY en tu entorno local.\n"
            "En PowerShell:\n"
            "  $env:OPENAI_API_KEY='sk-...'\n"
            "  $env:CHATBOT_PASSWORD='...'\n"
            "  python eval_ragas.py"
        )

    session = login_session()
    warmup(session)

    rows = build_rows(session, QUESTIONS)
    if not rows:
        raise RuntimeError(
            "No hay filas evaluables.\n"
            "Revisa QUESTIONS: deben devolver fragmentos_usados (contextos) en tu API."
        )

    save_jsonl(rows, OUT_JSONL)
    print(f"[OK] Dataset guardado: {OUT_JSONL} ({len(rows)} filas)")

    # Dataset para RAGAS (solo los campos que usa)
    ragas_rows = [{"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]} for r in rows]
    dataset = Dataset.from_list(ragas_rows)

    metrics = [faithfulness]
    if _answer_rel_metric is not None:
        metrics.append(_answer_rel_metric)
    else:
        print("[WARN] No se pudo importar answer_relevancy/answer_relevance. Se evaluará solo faithfulness.")

    results = evaluate(dataset, metrics=metrics)

    print("\n=== MÉTRICAS AGREGADAS ===")
    print(results)

    df = results.to_pandas()
    print("\n=== RESULTADOS POR FILA ===")
    print(df)

    save_results_csv(df, OUT_CSV)
    print(f"\n[OK] Resultados guardados: {OUT_CSV}")


if __name__ == "__main__":
    main()
