import time
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

BASE_URL = "https://mi-chatbot-ynz8.onrender.com"
LOGIN_URL = f"{BASE_URL}/login"
ASK_URL = f"{BASE_URL}/api/ask/"

# Pon aquí la contraseña de USER o ADMIN (con user vale para preguntar)
PASSWORD = "user"

QUESTIONS = [
    "Como funciona el sistema de automatizacion",
    "Que es una guia de cooperativistas",
]

def login_session() -> requests.Session:
    s = requests.Session()
    # login.html manda FormData; aquí lo replicamos como form-encoded
    r = s.post(LOGIN_URL, data={"password": PASSWORD}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Login fallido: {data}")
    return s

def ask_chatbot(session: requests.Session, question: str, timeout=180, retries=3):
    payload = {"question": question, "n_results": 15, "distance_threshold": 1.2}

    last_err = None
    for i in range(retries):
        try:
            r = session.post(ASK_URL, json=payload, timeout=timeout)
            # Si la sesión no vale, mejor verlo claro
            if r.status_code in (401, 403):
                raise RuntimeError(f"Auth error {r.status_code}: {r.text}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            # backoff sencillo
            time.sleep(2 * (i + 1))

    raise last_err

def warmup(session: requests.Session):
    # primera llamada para “despertar” Render y calentar conexiones
    try:
        ask_chatbot(session, "hola", timeout=180, retries=2)
    except Exception:
        # no pasa nada si el warmup falla, seguimos igual
        pass

def main():
    session = login_session()
    warmup(session)

    rows = []
    for q in QUESTIONS:
        data = ask_chatbot(session, q, timeout=180, retries=3)

        # Solo evaluamos cuando realmente hay contexto recuperado
        contexts = data.get("fragmentos_usados") or []
        if not contexts:
            print(f"[SKIP] Sin fragmentos usados para: {q}")
            continue

        rows.append({
            "question": q,
            "answer": data.get("respuesta", ""),
            "contexts": contexts
        })

    if not rows:
        raise RuntimeError("No hay filas evaluables. Revisa QUESTIONS: deben tener respuesta con fragmentos_usados.")

    dataset = Dataset.from_list(rows)
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    print(results)
    print(results.to_pandas())

if __name__ == "__main__":
    main()
