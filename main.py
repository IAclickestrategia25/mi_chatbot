# main.py
import os
import base64
import mimetypes
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional
import hashlib
import json

import csv
import fitz          # PyMuPDF para PDF
import docx          # python-docx para DOCX
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

from datetime import datetime, timedelta
from pydantic import BaseModel
from chroma_connection import get_chroma_collection


# -------------------------------------------------
# CARGA VARIABLES ENTORNO Y CLIENTE OPENAI
# -------------------------------------------------
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Contraseñas de acceso
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
USER_PASSWORD = os.getenv("USER_PASSWORD")

AUTH_COOKIE_NAME = "client_auth"

# Sesiones válidas: token -> rol ("admin" o "user")
VALID_SESSIONS: Dict[str, str] = {}

# Bloqueo de contraseña tras X intentos por dispositivo
FAILED_ATTEMPTS: Dict[str, Dict[str, Any]] = {}  # { "device_key": {"count": X, "until": datetime } }
MAX_ATTEMPTS = 5
BLOCK_TIME_MINUTES = 5


# -------------------------------------------------
# APP FASTAPI + CORS + STATIC
# -------------------------------------------------
app = FastAPI(title="ChromaDB FastAPI Integration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


# -------------------------------------------------
# AUTENTICACIÓN Y ROLES
# -------------------------------------------------
def generate_session_token() -> str:
    """
    Genera un token de sesión irrepetible.
    No depende de la contraseña para evitar patrones.
    """
    return hashlib.sha256(os.urandom(32)).hexdigest()


def get_session_role(request: Request) -> Optional[str]:
    """
    Devuelve el rol asociado al token de la cookie, o None si no es válido.
    """
    cookie_val = request.cookies.get(AUTH_COOKIE_NAME)
    if not cookie_val:
        return None
    return VALID_SESSIONS.get(cookie_val)


def require_auth(request: Request) -> str:
    """
    Dependencia general: requiere estar logueado.
    Devuelve el rol ("admin" o "user").
    """
    role = get_session_role(request)
    if not role:
        raise HTTPException(status_code=401, detail="No autorizado")
    return role


def require_admin(request: Request) -> str:
    """
    Dependencia para endpoints solo de administrador.
    """
    role = require_auth(request)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Solo disponible para administradores")
    return role


# -------------------------------------------------
# RUTAS FRONTEND (LOGIN + APP)
# -------------------------------------------------
@app.get("/")
async def serve_frontend(request: Request):
    # Si no está autenticado, lo mandamos al login
    if not get_session_role(request):
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse("frontend/index.html")


@app.get("/login")
async def login_page(request: Request):
    # Si ya está autenticado, lo mandamos directamente al asistente
    if get_session_role(request):
        return RedirectResponse(url="/", status_code=302)
    return FileResponse("frontend/login.html")


@app.post("/login")
async def do_login(request: Request):
    """
    Recibe el formulario de login (campo 'password') y, si es correcto,
    crea una cookie de sesión y responde en JSON.
    El frontend se encarga de redirigir o mostrar el error.
    """
    form = await request.form()
    password = form.get("password", "")

    # Identificador de dispositivo: IP + user-agent
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    device_key = f"{client_ip}|{user_agent[:80]}"

    now = datetime.utcnow()

    # 1) Comprobar si el dispositivo está bloqueado
    block_info = FAILED_ATTEMPTS.get(device_key)
    if block_info and block_info.get("until") and block_info["until"] > now:
        remaining_seconds = int((block_info["until"] - now).total_seconds())
        remaining_minutes = max(1, remaining_seconds // 60)
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    f"Demasiados intentos fallidos en este dispositivo. "
                    f"Inténtalo de nuevo en {remaining_minutes} minuto(s)."
                ),
            },
            status_code=429,
        )
    elif block_info and block_info.get("until") and block_info["until"] <= now:
        # Bloqueo expirado -> limpiamos
        del FAILED_ATTEMPTS[device_key]

    if not ADMIN_PASSWORD and not USER_PASSWORD:
        return JSONResponse(
            {"ok": False, "error": "No hay contraseñas configuradas en el servidor."},
            status_code=500,
        )

    # 2) Determinar rol según la contraseña introducida
    role: Optional[str] = None
    if ADMIN_PASSWORD and password == ADMIN_PASSWORD:
        role = "admin"
    elif USER_PASSWORD and password == USER_PASSWORD:
        role = "user"

    # 3) Contraseña incorrecta -> registrar intento
    if role is None:
        data = FAILED_ATTEMPTS.get(device_key, {"count": 0, "until": None})
        data["count"] += 1

        # Si se supera el límite, se bloquea durante X minutos
        if data["count"] >= MAX_ATTEMPTS:
            data["until"] = now + timedelta(minutes=BLOCK_TIME_MINUTES)
        FAILED_ATTEMPTS[device_key] = data

        return JSONResponse(
            {"ok": False, "error": "Contraseña incorrecta"},
            status_code=401,
        )

    # 4) Login correcto -> limpiar intentos fallidos de este dispositivo
    if device_key in FAILED_ATTEMPTS:
        del FAILED_ATTEMPTS[device_key]

    # 5) Generar token de sesión y guardarlo con el rol
    token = generate_session_token()
    VALID_SESSIONS[token] = role

    response = JSONResponse({"ok": True, "role": role})
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        httponly=True,          # JS no puede leerla
        secure=True,            # en Render va sobre HTTPS
        samesite="Strict",      # previene CSRF
        max_age=60 * 60 * 12,   # 12h de sesión
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
# MODELOS Pydantic
# -------------------------------------------------
class AddDocumentsBody(BaseModel):
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]


class AskBody(BaseModel):
    question: str
    n_results: int = 15
    distance_threshold: float = 1.2


class FileIndexItem(BaseModel):
    filename: str
    total_fragmentos: int


# -------------------------------------------------
# SELECCIÓN DE FRAGMENTOS RELEVANTES
# -------------------------------------------------
def seleccionar_fragmentos_relevantes(
    pregunta: str,
    candidatos: List[Dict[str, Any]],
    max_frag: int = 6
) -> List[int]:
    if not candidatos:
        return []

    partes = []
    for i, c in enumerate(candidatos):
        meta = c.get("meta", {})
        dist = c.get("dist", 0.0)
        filename = meta.get("filename", "desconocido")
        partes.append(
            f"[{i}] (archivo: {filename}, distancia: {dist:.3f})\n{c.get('doc','')}\n"
        )

    texto_fragmentos = "\n\n".join(partes)

    system_msg = (
        "Eres un asistente que actúa como motor de re-ranqueo de documentos.\n"
        "Tu tarea es elegir qué fragmentos de texto son MÁS ÚTILES para "
        "responder a la pregunta del usuario.\n\n"
        "Instrucciones:\n"
        " - Devuelve SOLO una lista de índices separados por comas (por ejemplo: 0,2,5).\n"
        " - No expliques nada, no añadas texto adicional.\n"
        " - Elige como máximo unos pocos fragmentos muy relevantes.\n"
    )

    user_msg = (
        f"Pregunta del usuario:\n{pregunta}\n\n"
        "Estos son los fragmentos candidatos:\n\n"
        f"{texto_fragmentos}\n\n"
        f"Indica los índices de los fragmentos más relevantes (máx. {max_frag}) "
        "para responder a la pregunta."
    )

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        texto_indices = completion.choices[0].message.content.strip()
    except Exception:
        return list(range(min(max_frag, len(candidatos))))

    indices: List[int] = []
    for trozo in texto_indices.replace("\n", ",").split(","):
        trozo = trozo.strip()
        if trozo.isdigit():
            i = int(trozo)
            if 0 <= i < len(candidatos):
                indices.append(i)

    indices_unicos: List[int] = []
    for i in indices:
        if i not in indices_unicos:
            indices_unicos.append(i)
    return indices_unicos[:max_frag]


# -------------------------------------------------
# UTILIDADES PARA TROCEAR TEXTO
# -------------------------------------------------
def trocear_texto(texto: str, max_chars: int = 800) -> List[str]:
    trozos: List[str] = []
    actual = ""

    for linea in texto.splitlines(keepends=True):
        if len(actual) + len(linea) > max_chars:
            if actual.strip():
                trozos.append(actual.strip())
            actual = linea
        else:
            actual += linea

    if actual.strip():
        trozos.append(actual.strip())

    return trozos


# -------------------------------------------------
# LECTURA DE PDF, DOCX, CSV
# -------------------------------------------------
def leer_pdf_bytes(data: bytes) -> str:
    texto = ""
    with fitz.open(stream=data, filetype="pdf") as pdf:
        for pagina in pdf:
            texto += pagina.get_text()
    return texto


def leer_docx_bytes(data: bytes) -> str:
    archivo = BytesIO(data)
    doc = docx.Document(archivo)
    return "\n".join(p.text for p in doc.paragraphs)


def leer_csv_bytes(data: bytes, encoding: str = "utf-8") -> str:
    texto = data.decode(encoding, errors="ignore")
    f = StringIO(texto)
    reader = csv.reader(f)

    filas = []
    for row in reader:
        linea = " ; ".join(col.strip() for col in row if col is not None)
        if linea.strip():
            filas.append(linea)

    return "\n".join(filas)


# -------------------------------------------------
# DESCRIPCIÓN DE IMÁGENES CON OPENAI (VISIÓN)
# -------------------------------------------------
def describir_imagen_bytes(data: bytes, filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime is None:
        mime = "image/png"

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

    return resp.choices[0].message.content.strip()




# -------------------------------------------------
# ENDPOINT: Nº DE SESIONES ACTIVAS (APROX.)
# -------------------------------------------------
@app.get("/api/active-sessions/")
async def active_sessions():

    return {
        "count": len(VALID_SESSIONS)
    }



# -------------------------------------------------
# ENDPOINT: SUBIR DOCUMENTOS (JSON) A CHROMA (SOLO ADMIN)
# -------------------------------------------------
@app.post("/api/documents/")
async def add_documents(
    body: AddDocumentsBody,
    role: str = Depends(require_admin),
    col=Depends(get_chroma_collection),
):
    try:
        col.add(
            ids=body.ids,
            documents=body.documents,
            metadatas=body.metadatas,
        )
        return {"message": "Documents added successfully", "ids": body.ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# ENDPOINT: SUBIR PDFs, DOCX, PNG, JPG, CSV (SOLO ADMIN)
# -------------------------------------------------
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
        filename = file.filename
        ext = (filename or "").lower().split(".")[-1]

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

        if not texto.strip():
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
            batch_metas.append(
                {
                    "filename": filename,
                    "chunk": i,
                    "tipo": tipo,
                }
            )

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


    
    
# -------------------------------------------------
# ENDPOINT: HACER PREGUNTAS A TUS DOCUMENTOS (ADMIN Y USUARIO)
# -------------------------------------------------
@app.post("/api/ask/")
async def ask_documents(
    body: AskBody,
    role: str = Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    pregunta_original = body.question.strip()

    # ---------- 0. Reescritura de la pregunta para búsqueda ----------
    rewrite_system = (
        "Eres un asistente especializado en generación de consultas de búsqueda.\n"
        "Transforma la pregunta del usuario en una consulta MUY corta y útil para "
        "buscar en una base de conocimiento. Devuelve SOLO la consulta, sin comillas.\n"
    )
    rewrite_user = f"Pregunta del usuario: {pregunta_original}"

    try:
        rew = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": rewrite_system},
                {"role": "user", "content": rewrite_user},
            ],
        )
        consulta_chroma = rew.choices[0].message.content.strip()
    except Exception:
        consulta_chroma = pregunta_original

    # ---------- 1. Consultar Chroma ----------
    n_candidatos = max(body.n_results, 20)
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
        # Sin nada relevante en Chroma
        return {
            "respuesta": "No he encontrado información relevante en los documentos para responder a esta pregunta.",
            "fuentes": [],
            "fragmentos_usados": [],
            "distancias": [],
        }

    # ---------- 2. Re-ranqueo con GPT ----------
    indices_buenos = seleccionar_fragmentos_relevantes(
        pregunta=pregunta_original,
        candidatos=candidatos,
        max_frag=6,
    )

    if not indices_buenos:
        k = 5
        candidatos_ordenados = sorted(candidatos, key=lambda x: x["dist"])
        buenos = candidatos_ordenados[:k]
    else:
        buenos = [candidatos[i] for i in indices_buenos]

    # ---------- 3. Construir contexto ----------
    contexto_partes = []
    filtrados = []
    for i, c in enumerate(buenos, start=1):
        doc = c["doc"]
        meta = c["meta"]
        dist = c["dist"]
        filtrados.append((doc, meta, dist))
        filename = meta.get("filename", "desconocido")
        contexto_partes.append(
            f"[FRAGMENTO {i} | archivo: {filename} | distancia: {dist:.3f}]\n{doc}\n"
        )

    contexto = "\n\n".join(contexto_partes)

    # Marcador especial para saber si NO se han usado datos de los documentos
    MARKER_SIN_DATOS = "[[SIN_DATOS_DOCUMENTOS]]"

    system_msg = (
        "Eres un asistente que responde ÚNICAMENTE usando la información "
        "que aparece en los fragmentos de texto proporcionados.\n\n"
        "Instrucciones importantes:\n"
        "1) Usa siempre los fragmentos como única fuente de verdad cuando exista información relevante.\n"
        "2) Si la pregunta es genérica, responde resumiendo la información relevante de los fragmentos.\n"
        "3) Si los fragmentos NO contienen información útil para responder a la pregunta "
        "(por ejemplo, saludos como 'hola', preguntas sobre el tiempo actual, etc.), "
        "puedes responder con un mensaje general indicando que no hay datos en los documentos.\n"
        f"4) Cuando NO utilices información de los documentos para elaborar tu respuesta, "
        f"AÑADE al final de la respuesta exactamente este marcador: {MARKER_SIN_DATOS}\n"
        "   - No añadas texto después del marcador.\n"
        "   - El marcador sirve para que el sistema sepa que no se han usado fragmentos.\n"
        "5) Responde siempre en español neutro."
    )

    user_msg = (
        f"Pregunta original del usuario:\n{pregunta_original}\n\n"
        f"Consulta usada para buscar en Chroma:\n{consulta_chroma}\n\n"
        "A continuación tienes fragmentos de documentos (contexto). "
        "Utilízalos para responder, sin añadir información externa:\n\n"
        f"{contexto}"
    )

    # ---------- 4. Llamar a OpenAI ----------
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw_answer = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error llamando a OpenAI: {e}")

    # ---------- 5. Detectar si la respuesta NO usa documentos ----------
    sin_datos_documentos = MARKER_SIN_DATOS in raw_answer
    respuesta = raw_answer.replace(MARKER_SIN_DATOS, "").strip()

    if sin_datos_documentos:
        # No mostramos fuentes ni fragmentos
        return {
            "respuesta": respuesta,
            "fuentes": [],
            "fragmentos_usados": [],
            "distancias": [],
        }

    # ---------- 6. Caso normal: sí ha usado documentos ----------
    return {
        "respuesta": respuesta,
        "fuentes": list({meta.get("filename", "desconocido") for _, meta, _ in filtrados}),
        "fragmentos_usados": [doc for doc, _, _ in filtrados],
        "distancias": [float(dist) for _, _, dist in filtrados],
    }



# -------------------------------------------------
# ENDPOINT: RESUMEN DEL ÍNDICE (ARCHIVOS + Nº FRAGMENTOS)
# -------------------------------------------------
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
        filename = meta.get("filename", "desconocido")
        contador[filename] = contador.get(filename, 0) + 1

    archivos = [
        {"filename": name, "total_fragmentos": count}
        for name, count in sorted(contador.items())
    ]

    return {"archivos": archivos}


# -------------------------------------------------
# ENDPOINT: VER ALGUNOS FRAGMENTOS DE UN ARCHIVO
# -------------------------------------------------
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

    return {
        "filename": filename,
        "documentos": docs or [],
    }


# -------------------------------------------------
# ENDPOINT: ELIMINAR TODOS LOS FRAGMENTOS DE UN ARCHIVO (SOLO ADMIN)
# -------------------------------------------------
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
