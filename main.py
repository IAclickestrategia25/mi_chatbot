# main.py
import os
import re
import base64
import mimetypes
import hashlib
import csv
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import fitz  # PyMuPDF
import docx  # python-docx
from dotenv import load_dotenv
from openai import OpenAI

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Request,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel

from chroma_connection import get_chroma_collection


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
USER_PASSWORD = os.getenv("USER_PASSWORD")

AUTH_COOKIE_NAME = "client_auth"

# Sesiones válidas: token -> rol ("admin" o "user")
VALID_SESSIONS: Dict[str, str] = {}

# Bloqueo de contraseña tras X intentos por dispositivo
FAILED_ATTEMPTS: Dict[str, Dict[str, Any]] = {}
MAX_ATTEMPTS = 5
BLOCK_TIME_MINUTES = 5

# Límites para calidad/latencia
MAX_RERANK_POOL = 20
MAX_CONTEXT_FRAGMENTS_DEFAULT = 3
MAX_CONTEXT_CHARS_BACKEND = 6000
FALLBACK_TOPK_IF_EMPTY = 10

# Respuesta “sin datos”
NO_DATA_PHRASE = "No consta en los documentos proporcionados."


# -------------------------------------------------
# APP FASTAPI + CORS + STATIC
# -------------------------------------------------
app = FastAPI(title="ChromaDB FastAPI Integration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en privado, restringe dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


# -------------------------------------------------
# UTILIDADES
# -------------------------------------------------
def normalize_ws(s: str) -> str:
    if not s:
        return ""
    lines = [line.rstrip() for line in s.splitlines()]
    return "\n".join(lines).strip()


def suggested_k(question: str) -> int:
    q = (question or "").strip().lower()
    if q.startswith(("qué es", "que es", "define", "definición", "definicion")):
        return 2
    return MAX_CONTEXT_FRAGMENTS_DEFAULT


def clean_llm_answer(text: str) -> str:
    """
    Elimina cualquier etiqueta tipo [F1], [F2]... y normaliza espacios.
    """
    t = (text or "").strip()
    t = re.sub(r"\[F\d+\]", "", t)       # quita [F1], [F2], ...
    t = re.sub(r"\s+", " ", t).strip()  # normaliza espacios
    return t


def is_no_data_answer(text: str) -> bool:
    """
    Considera equivalente con o sin punto final y mayúsculas/minúsculas.
    """
    a = (text or "").strip().rstrip(".").lower()
    b = NO_DATA_PHRASE.strip().rstrip(".").lower()
    return a == b


# -------------------------------------------------
# AUTENTICACIÓN Y ROLES
# -------------------------------------------------
def generate_session_token() -> str:
    return hashlib.sha256(os.urandom(32)).hexdigest()


def get_session_role(request: Request) -> Optional[str]:
    cookie_val = request.cookies.get(AUTH_COOKIE_NAME)
    if not cookie_val:
        return None
    return VALID_SESSIONS.get(cookie_val)


def require_auth(request: Request) -> str:
    role = get_session_role(request)
    if not role:
        raise HTTPException(status_code=401, detail="No autorizado")
    return role


def require_admin(request: Request) -> str:
    role = require_auth(request)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Solo disponible para administradores")
    return role


# -------------------------------------------------
# RUTAS FRONTEND
# -------------------------------------------------
@app.get("/")
async def serve_frontend(request: Request):
    if not get_session_role(request):
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse("frontend/index.html")


@app.get("/login")
async def login_page(request: Request):
    if get_session_role(request):
        return RedirectResponse(url="/", status_code=302)
    return FileResponse("frontend/login.html")


@app.post("/login")
async def do_login(request: Request):
    form = await request.form()
    password = (form.get("password", "") or "").strip()

    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    device_key = f"{client_ip}|{user_agent[:80]}"

    now = datetime.utcnow()

    # Bloqueo por intentos
    block_info = FAILED_ATTEMPTS.get(device_key)
    if block_info and block_info.get("until") and block_info["until"] > now:
        remaining_seconds = int((block_info["until"] - now).total_seconds())
        remaining_minutes = max(1, remaining_seconds // 60)
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    "Demasiados intentos fallidos en este dispositivo. "
                    f"Inténtalo de nuevo en {remaining_minutes} minuto(s)."
                ),
            },
            status_code=429,
        )
    elif block_info and block_info.get("until") and block_info["until"] <= now:
        del FAILED_ATTEMPTS[device_key]

    if not ADMIN_PASSWORD and not USER_PASSWORD:
        return JSONResponse(
            {"ok": False, "error": "No hay contraseñas configuradas en el servidor."},
            status_code=500,
        )

    role: Optional[str] = None
    if ADMIN_PASSWORD and password == ADMIN_PASSWORD:
        role = "admin"
    elif USER_PASSWORD and password == USER_PASSWORD:
        role = "user"

    if role is None:
        data = FAILED_ATTEMPTS.get(device_key, {"count": 0, "until": None})
        data["count"] += 1
        if data["count"] >= MAX_ATTEMPTS:
            data["until"] = now + timedelta(minutes=BLOCK_TIME_MINUTES)
        FAILED_ATTEMPTS[device_key] = data
        return JSONResponse({"ok": False, "error": "Contraseña incorrecta"}, status_code=401)

    if device_key in FAILED_ATTEMPTS:
        del FAILED_ATTEMPTS[device_key]

    token = generate_session_token()
    VALID_SESSIONS[token] = role

    response = JSONResponse({"ok": True, "role": role})

    # Secure automático: https en Render, http en local
    is_https = request.url.scheme == "https"
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        httponly=True,
        secure=is_https,
        samesite="Strict",
        max_age=60 * 60 * 12,
        path="/",
    )
    return response


@app.get("/logout")
async def logout(request: Request):
    cookie_val = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_val in VALID_SESSIONS:
        VALID_SESSIONS.pop(cookie_val, None)

    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return response


# -------------------------------------------------
# MODELOS
# -------------------------------------------------
class AddDocumentsBody(BaseModel):
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]


class AskBody(BaseModel):
    question: str
    n_results: int = 15
    distance_threshold: float = 1.2


# -------------------------------------------------
# TROCEADO / LECTURA
# -------------------------------------------------
def trocear_texto(texto: str, max_chars: int = 800) -> List[str]:
    trozos: List[str] = []
    actual = ""
    for linea in (texto or "").splitlines(keepends=True):
        if len(actual) + len(linea) > max_chars:
            if actual.strip():
                trozos.append(actual.strip())
            actual = linea
        else:
            actual += linea
    if actual.strip():
        trozos.append(actual.strip())
    return trozos


def leer_pdf_bytes(data: bytes) -> str:
    texto = ""
    with fitz.open(stream=data, filetype="pdf") as pdf:
        for pagina in pdf:
            texto += pagina.get_text()
    return texto


def leer_docx_bytes(data: bytes) -> str:
    archivo = BytesIO(data)
    d = docx.Document(archivo)
    return "\n".join(p.text for p in d.paragraphs)


def leer_csv_bytes(data: bytes, encoding: str = "utf-8") -> str:
    texto = data.decode(encoding, errors="ignore")
    f = StringIO(texto)
    reader = csv.reader(f)
    filas = []
    for row in reader:
        linea = " ; ".join((col or "").strip() for col in row if col is not None)
        if linea.strip():
            filas.append(linea)
    return "\n".join(filas)


def describir_imagen_bytes(data: bytes, filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "image/png"

    b64 = base64.b64encode(data).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    prompt = (
        "Describe con detalle el contenido de esta imagen en español. "
        "Incluye cualquier texto legible, tablas, diagramas o datos importantes. "
        "No inventes nada que no se vea claramente."
    )

    resp = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        max_tokens=450,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# -------------------------------------------------
# RE-RANKEO (elige índices)
# -------------------------------------------------
def seleccionar_fragmentos_relevantes(
    pregunta: str,
    candidatos: List[Dict[str, Any]],
    max_frag: int,
) -> List[int]:
    if not candidatos:
        return []

    partes = []
    for i, c in enumerate(candidatos):
        meta = c.get("meta") or {}
        dist = float(c.get("dist", 0.0) or 0.0)
        filename = meta.get("filename", "desconocido")
        doc = normalize_ws(c.get("doc", "") or "")
        partes.append(f"[{i}] (archivo: {filename}, distancia: {dist:.3f})\n{doc}\n")

    texto_fragmentos = "\n\n".join(partes)

    system_msg = (
        "Actúas como motor de re-ranqueo.\n"
        "Devuelve SOLO índices separados por comas (ej: 0,2,5). Sin texto extra.\n"
        f"Devuelve como máximo {max_frag} índices.\n"
    )

    user_msg = (
        f"Pregunta:\n{pregunta}\n\n"
        "Fragmentos candidatos:\n\n"
        f"{texto_fragmentos}\n\n"
        "Devuelve los índices."
    )

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            max_tokens=30,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        texto_indices = (completion.choices[0].message.content or "").strip()
    except Exception:
        return list(range(min(max_frag, len(candidatos))))

    indices: List[int] = []
    for trozo in texto_indices.replace("\n", ",").split(","):
        t = trozo.strip()
        if t.isdigit():
            idx = int(t)
            if 0 <= idx < len(candidatos):
                indices.append(idx)

    out: List[int] = []
    for i in indices:
        if i not in out:
            out.append(i)
    return out[:max_frag]


# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------
@app.get("/api/active-sessions/")
async def active_sessions():
    return {"count": len(VALID_SESSIONS)}


@app.post("/api/documents/")
async def add_documents(
    body: AddDocumentsBody,
    role: str = Depends(require_admin),
    col=Depends(get_chroma_collection),
):
    try:
        col.add(ids=body.ids, documents=body.documents, metadatas=body.metadatas)
        return {"message": "Documents added successfully", "ids": body.ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-files/")
async def upload_files(
    files: List[UploadFile] = File(...),
    role: str = Depends(require_admin),
    col=Depends(get_chroma_collection),
):
    if not files:
        raise HTTPException(status_code=400, detail="No se ha enviado ningún archivo.")

    BATCH_SIZE = 200
    total_archivos = 0
    total_fragmentos = 0

    for file in files:
        filename = file.filename or "desconocido"
        ext = filename.lower().split(".")[-1]
        data = await file.read()

        try:
            if ext == "pdf":
                texto = leer_pdf_bytes(data)
                tipo = "pdf"
            elif ext == "docx":
                texto = leer_docx_bytes(data)
                tipo = "docx"
            elif ext in ("png", "jpg", "jpeg"):
                texto = describir_imagen_bytes(data, filename)
                tipo = "imagen"
            elif ext == "csv":
                texto = leer_csv_bytes(data)
                tipo = "csv"
            else:
                print(f"Saltando archivo no soportado: {filename}")
                continue
        except Exception as e:
            print(f"Error procesando {filename}: {e}")
            continue

        texto = normalize_ws(texto)
        if not texto:
            print(f"Sin texto/descripcion útil en {filename}, se omite.")
            continue

        fragmentos = trocear_texto(texto, max_chars=800)

        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_metas: List[Dict[str, Any]] = []

        for i, frag in enumerate(fragmentos):
            doc_id = f"{filename}_{i}"
            batch_ids.append(doc_id)
            batch_docs.append(frag)
            batch_metas.append({"filename": filename, "chunk": i, "tipo": tipo})

            if len(batch_ids) >= BATCH_SIZE:
                try:
                    col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error guardando en Chroma (lote) para {filename}: {e}",
                    )
                total_fragmentos += len(batch_ids)
                batch_ids, batch_docs, batch_metas = [], [], []

        if batch_ids:
            try:
                col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error guardando en Chroma (último lote) para {filename}: {e}",
                )
            total_fragmentos += len(batch_ids)

        total_archivos += 1

    if total_archivos == 0:
        return {
            "message": (
                "Los archivos se han recibido, pero no se ha podido extraer "
                "texto útil para indexarlos. Es posible que sean PDFs escaneados "
                "o imágenes sin texto."
            ),
            "total_archivos": 0,
            "total_fragmentos": 0,
        }

    return {
        "message": "Archivos procesados e indexados correctamente.",
        "total_archivos": total_archivos,
        "total_fragmentos": total_fragmentos,
    }


@app.post("/api/ask/")
async def ask_documents(
    body: AskBody,
    role: str = Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    pregunta_original = (body.question or "").strip()
    if not pregunta_original:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    # 0) Reescritura de consulta (solo para búsqueda)
    rewrite_system = (
        "Eres un asistente especializado en generar consultas de búsqueda.\n"
        "Convierte la pregunta en una consulta corta y efectiva.\n"
        "Devuelve SOLO la consulta.\n"
    )
    try:
        rew = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            max_tokens=60,
            messages=[
                {"role": "system", "content": rewrite_system},
                {"role": "user", "content": f"Pregunta: {pregunta_original}"},
            ],
        )
        consulta_chroma = (rew.choices[0].message.content or "").strip() or pregunta_original
    except Exception:
        consulta_chroma = pregunta_original

    # 1) Query en Chroma
    n_candidatos = max(int(body.n_results), 20)
    try:
        res = col.query(
            query_texts=[consulta_chroma],
            n_results=n_candidatos,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando Chroma: {e}")

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    candidatos: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        candidatos.append({"doc": doc, "meta": meta, "dist": float(dist)})

    if not candidatos:
        return {"respuesta": NO_DATA_PHRASE, "fuentes": [], "fragmentos_usados": [], "distancias": []}

    # 1.1) Filtrar por threshold con fallback
    thr = float(body.distance_threshold)
    candidatos_sorted = sorted(candidatos, key=lambda x: x["dist"])
    filtrados_thr = [c for c in candidatos_sorted if c["dist"] <= thr]
    if filtrados_thr:
        candidatos_use = filtrados_thr
    else:
        candidatos_use = candidatos_sorted[:FALLBACK_TOPK_IF_EMPTY]

    if not candidatos_use:
        return {"respuesta": NO_DATA_PHRASE, "fuentes": [], "fragmentos_usados": [], "distancias": []}

    # 1.2) Pool para rerank
    rerank_pool = candidatos_use[:MAX_RERANK_POOL]

    # 2) Re-ranqueo + selección final
    k = suggested_k(pregunta_original)
    idxs = seleccionar_fragmentos_relevantes(pregunta_original, rerank_pool, max_frag=k)
    if not idxs:
        buenos = rerank_pool[:k]
    else:
        buenos = [rerank_pool[i] for i in idxs][:k]

    # 3) Construir contexto SIN etiquetas [F1] (para evitar que se copien)
    contexto_partes: List[str] = []
    filtrados: List[Dict[str, Any]] = []

    for c in buenos:
        doc_txt = normalize_ws(c.get("doc", "") or "")
        meta = c.get("meta") or {}
        dist = float(c.get("dist", 0.0) or 0.0)
        filename = meta.get("filename", "desconocido")

        filtrados.append({"doc": doc_txt, "meta": meta, "dist": dist})

        contexto_partes.append(
            f"[archivo: {filename} | distancia: {dist:.3f}]\n{doc_txt}\n"
        )

    contexto = "\n\n".join(contexto_partes).strip()
    if len(contexto) > MAX_CONTEXT_CHARS_BACKEND:
        contexto = contexto[:MAX_CONTEXT_CHARS_BACKEND]

    system_msg = (
        "Eres un asistente RAG.\n"
        "Responde SOLO con información explícita en los fragmentos.\n\n"
        "Reglas:\n"
        f"- Si la respuesta no está en los fragmentos, di exactamente: \"{NO_DATA_PHRASE}\" y no añadas nada más.\n"
        "- NO inventes ni completes con suposiciones.\n"
        "- PROHIBIDO incluir etiquetas de cita en la respuesta (por ejemplo: [F1], [F2], etc.).\n"
        "- No uses conocimiento externo.\n"
        "- Español neutro.\n"
    )

    user_msg = (
        f"Pregunta:\n{pregunta_original}\n\n"
        "Fragmentos:\n\n"
        f"{contexto}"
    )

    # 4) Llamada al LLM
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            max_tokens=450,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        respuesta_raw = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error llamando a OpenAI: {e}")

    # 5) Limpieza de salida + política “sin fuentes si no hay respuesta”
    respuesta = clean_llm_answer(respuesta_raw)
    if is_no_data_answer(respuesta):
        return {"respuesta": NO_DATA_PHRASE, "fuentes": [], "fragmentos_usados": [], "distancias": []}

    # 6) Fuentes deterministas (no dependen del texto del modelo)
    fuentes = list({(f["meta"] or {}).get("filename", "desconocido") for f in filtrados})
    fragmentos_usados = [f["doc"] for f in filtrados]
    distancias = [float(f["dist"]) for f in filtrados]

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "fragmentos_usados": fragmentos_usados,
        "distancias": distancias,
    }


@app.get("/api/index-summary/")
async def index_summary(
    role: str = Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    try:
        res = col.get(include=["metadatas"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo de Chroma: {e}")

    metadatas = res.get("metadatas", [])
    contador: Dict[str, int] = {}

    for meta in metadatas:
        filename = (meta or {}).get("filename", "desconocido")
        contador[filename] = contador.get(filename, 0) + 1

    archivos = [{"filename": name, "total_fragmentos": count} for name, count in sorted(contador.items())]
    return {"archivos": archivos}


@app.get("/api/file-fragments/")
async def file_fragments(
    filename: str,
    limit: int = 3,
    role: str = Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    if not filename:
        raise HTTPException(status_code=400, detail="Debe indicarse un nombre de archivo.")

    try:
        res = col.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo de Chroma: {e}")

    docs = res.get("documents", [])
    if docs and isinstance(docs[0], list):
        docs = docs[0]

    return {"filename": filename, "documentos": docs or []}


@app.delete("/api/delete-by-filename/")
async def delete_by_filename(
    filename: str,
    role: str = Depends(require_admin),
    col=Depends(get_chroma_collection),
):
    if not filename:
        raise HTTPException(status_code=400, detail="Debe indicarse un nombre de archivo.")

    try:
        col.delete(where={"filename": filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando en Chroma: {e}")

    return {"message": f"Se han eliminado todos los fragmentos asociados a '{filename}'."}
