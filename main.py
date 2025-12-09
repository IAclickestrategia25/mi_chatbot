# main.py
import os
import base64
import mimetypes
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional
import hashlib

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
    Header,
    Request,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse, JSONResponse

from datetime import datetime, timedelta
from pydantic import BaseModel
from chroma_connection import get_chroma_collection

# -------------------------------------------------
# CARGA .env Y CLIENTE OPENAI
# -------------------------------------------------
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Token de administrador para subir/borrar archivos
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

# Contraseña de acceso del cliente (login)
CLIENT_PASSWORD = os.getenv("CLIENT_PASSWORD")
AUTH_COOKIE_NAME = "client_auth"

VALID_SESSIONS: set[str] = set()
#Bloqueo de contraseña tras 5 intentos
FAILED_ATTEMPTS = {}  # { "IP": {"count": X, "until": datetime } }
MAX_ATTEMPTS = 5
BLOCK_TIME_MINUTES = 5
# -------------------------------------------------
# APP FASTAPI + CORS + STATIC
# -------------------------------------------------
app = FastAPI(title="ChromaDB FastAPI Integration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # en local y en Render está bien
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir estáticos si algún día tienes assets separados
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# -------------------------------------------------
# AUTENTICACIÓN SENCILLA POR COOKIE
# -------------------------------------------------
def is_authenticated(request: Request) -> bool:
    """
    Comprueba si el token de la cookie pertenece a una sesión válida.
    """
    cookie_val = request.cookies.get(AUTH_COOKIE_NAME)
    if not cookie_val:
        return False
    return cookie_val in VALID_SESSIONS


def require_auth(request: Request):
    """
    Dependencia para proteger endpoints de la API.
    """
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="No autorizado")


def generate_session_token() -> str:
    """
    Genera un token de sesión irrepetible a partir de la contraseña
    y un valor aleatorio. No se reutiliza entre sesiones.
    """
    if not CLIENT_PASSWORD:
        # Por seguridad, nunca debería pasar en producción
        raise RuntimeError("CLIENT_PASSWORD no está configurada")

    random_bytes = os.urandom(16)
    raw = CLIENT_PASSWORD.encode("utf-8") + random_bytes
    return hashlib.sha256(raw).hexdigest()

# -------------------------------------------------
# RUTAS DE FRONT (LOGIN + APP)
# -------------------------------------------------
# Servir el frontend en la raíz "/"
@app.get("/")
async def serve_frontend(request: Request):
    # Si no está autenticado, lo mandamos al login
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse("frontend/index.html")


@app.get("/login")
async def login_page(request: Request):
    # Si ya está autenticado, lo mandamos directamente al asistente
    if is_authenticated(request):
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

    # IP del cliente
    client_ip = request.client.host if request.client else "unknown"
    # User-Agent del navegador/dispositivo
    user_agent = request.headers.get("user-agent", "unknown")

    # Clave de dispositivo: IP + User-Agent (recortado para no hacerlo eterno)
    device_key = f"{client_ip}|{user_agent[:80]}"

    now = datetime.utcnow()

    # 1) Comprobar si este "dispositivo" está bloqueado temporalmente
    block_info = FAILED_ATTEMPTS.get(device_key)
    if block_info and block_info.get("until") and block_info["until"] > now:
        remaining_seconds = int((block_info["until"] - now).total_seconds())
        remaining_minutes = max(1, remaining_seconds // 60)
        return JSONResponse(
            {
                "ok": False,
                "error": f"Demasiados intentos fallidos en este dispositivo. "
                         f"Inténtalo de nuevo en {remaining_minutes} minuto(s).",
            },
            status_code=429,
        )
    elif block_info and block_info.get("until") and block_info["until"] <= now:
        # Bloqueo expirado -> limpiamos
        del FAILED_ATTEMPTS[device_key]

    if not CLIENT_PASSWORD:
        return JSONResponse(
            {"ok": False, "error": "No hay contraseña de cliente configurada en el servidor."},
            status_code=500,
        )

    # 2) Contraseña incorrecta -> registrar intento para este dispositivo
    if password != CLIENT_PASSWORD:
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

    # 3) Login correcto -> limpiar intentos fallidos de este dispositivo
    if device_key in FAILED_ATTEMPTS:
        del FAILED_ATTEMPTS[device_key]

    # 4) Generar token de sesión nuevo y guardarlo en memoria
    token = generate_session_token()
    VALID_SESSIONS.add(token)

    response = JSONResponse({"ok": True})
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        httponly=True,          # el JS no puede leerla
        secure=True,            # SOLO funciona con HTTPS (en local podrías poner False si da problemas)
        samesite="Strict",      # previene CSRF
        max_age=60 * 60 * 12,   # 12h de sesión (ajustable)
        path="/",
    )

    return response


# (Opcional) logout sencillo
@app.get("/logout")
async def logout(request: Request):
    cookie_val = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_val in VALID_SESSIONS:
        VALID_SESSIONS.discard(cookie_val)

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


def seleccionar_fragmentos_relevantes(
    pregunta: str,
    candidatos: List[Dict[str, Any]],
    max_frag: int = 6
) -> List[int]:
    """
    Usa GPT para seleccionar los fragmentos realmente útiles
    entre los candidatos devueltos por Chroma.

    Devuelve una lista de índices (posiciones en la lista candidatos).
    """

    if not candidatos:
        return []

    # Construimos un texto con todos los fragmentos candidatos numerados
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
        # Si algo falla, devolvemos simplemente los primeros max_frag
        return list(range(min(max_frag, len(candidatos))))

    # Parsear números tipo "0, 2, 5"
    indices: List[int] = []
    for trozo in texto_indices.replace("\n", ",").split(","):
        trozo = trozo.strip()
        if trozo.isdigit():
            i = int(trozo)
            if 0 <= i < len(candidatos):
                indices.append(i)

    # Evitar duplicados y limitar el tamaño
    indices_unicos = []
    for i in indices:
        if i not in indices_unicos:
            indices_unicos.append(i)
    return indices_unicos[:max_frag]


# -------------------------------------------------
# UTILIDADES PARA TROCEAR TEXTO
# -------------------------------------------------
def trocear_texto(texto: str, max_chars: int = 800) -> List[str]:
    """
    Divide el texto en trozos de tamaño máximo max_chars,
    intentando cortar por líneas.
    """
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
    """
    Convierte un CSV (en bytes) a un texto plano legible,
    línea a línea, para indexarlo en Chroma.
    """
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
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    )

    return resp.choices[0].message.content.strip()


# -------------------------------------------------
# ENDPOINT: SUBIR DOCUMENTOS (JSON) A CHROMA
# -------------------------------------------------
@app.post("/api/documents/")
async def add_documents(
    body: AddDocumentsBody,
    auth=Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    """
    Recibe listas de ids, documents y metadatas y los guarda en la colección de Chroma.
    Útil si indexas desde scripts Python en vez de desde el navegador.
    """
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
# ENDPOINT: SUBIR PDFs, DOCX, PNG, JPG, CSV DESDE EL NAVEGADOR (PROTEGIDO)
# -------------------------------------------------
@app.post("/api/upload-files/")
async def upload_files(
    files: List[UploadFile] = File(...),
    admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    auth=Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    """
    Recibe ficheros (PDF, DOCX, PNG, JPG/JPEG, CSV) desde el navegador,
    extrae el texto o la descripción y los indexa directamente en Chroma.
    Solo permite la subida si el token de admin es correcto.
    """
    # Protección por token
    if ADMIN_TOKEN and admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Token de administrador incorrecto para subir archivos.",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No se ha enviado ningún archivo.")

    # Tamaño máximo de lote al enviar a Chroma (número de fragmentos)
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

        # Troceamos el texto del archivo en fragmentos
        fragmentos = trocear_texto(texto, max_chars=800)

        # Enviar a Chroma en lotes pequeños
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

            # Cuando el lote alcanza el tamaño máximo, lo enviamos a Chroma
            if len(batch_ids) >= BATCH_SIZE:
                try:
                    col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error guardando en Chroma (lote) para {filename}: {e}",
                    )
                total_fragmentos += len(batch_ids)
                # Reseteamos el lote
                batch_ids, batch_docs, batch_metas = [], [], []

        # Enviar el último lote (si queda algo pendiente)
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
        # En lugar de devolver 400, devolvemos éxito pero indicando que no se ha indexado nada
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
# ENDPOINT: HACER PREGUNTAS A TUS DOCUMENTOS
# -------------------------------------------------
@app.post("/api/ask/")
async def ask_documents(
    body: AskBody,
    auth=Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    """
    1) Reescribe la pregunta a una consulta corta para Chroma.
    2) Recupera muchos candidatos de Chroma.
    3) Usa GPT para re-ranquear y elegir los fragmentos realmente relevantes.
    4) Genera la respuesta usando SOLO esos fragmentos.
    """

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

    # ---------- 1. Consultar Chroma (recuperar bastantes candidatos) ----------
    n_candidatos = max(body.n_results, 20)  # apunta alto para que haya material
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
        return {
            "respuesta": "No he encontrado información relevante sobre esa pregunta en los documentos.",
            "fragmentos_usados": [],
            "distancias": [],
        }

    # ---------- 2. Re-ranqueo con GPT para elegir los fragmentos buenos ----------
    indices_buenos = seleccionar_fragmentos_relevantes(
        pregunta=pregunta_original,
        candidatos=candidatos,
        max_frag=6,
    )

    if not indices_buenos:
        # Fallback: usar los k mejores por distancia
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

    system_msg = (
        "Eres un asistente que responde ÚNICAMENTE usando la información "
        "que aparece en los fragmentos de texto proporcionados.\n\n"
        "Instrucciones importantes:\n"
        "1) Usa siempre los fragmentos como única fuente de verdad.\n"
        "2) Si la pregunta es genérica (por ejemplo, 'precio productos', 'stock', "
        "'categoría', etc.), interpreta la intención y responde resumiendo la "
        "información relevante de los fragmentos (tablas, listas...).\n"
        "3) Si falta un dato concreto, dilo claramente, indicando qué sí aparece en "
        "los fragmentos y qué no se menciona.\n"
        "4) No añadas conocimientos externos ni inventes nada.\n"
        "Responde siempre en español neutro."
    )

    user_msg = (
        f"Pregunta original del usuario:\n{pregunta_original}\n\n"
        f"Consulta usada para buscar en Chroma:\n{consulta_chroma}\n\n"
        "A continuación tienes fragmentos de documentos (contexto). "
        "Utilízalos para responder, sin añadir información externa:\n\n"
        f"{contexto}"
    )

    # ---------- 4. Llamar a OpenAI para generar la respuesta ----------
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        respuesta = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error llamando a OpenAI: {e}")

    return {
        "respuesta": respuesta,
        "fuentes": list({meta.get("filename", "desconocido") for _, meta, _ in filtrados}),
        "fragmentos_usados": [doc for doc, _, _ in filtrados],
        "distancias": [float(dist) for _, _, dist in filtrados],
    }


# -------------------------------------------------
# NUEVO ENDPOINT: RESUMEN DEL ÍNDICE (archivos + nº fragmentos)
# -------------------------------------------------
@app.get("/api/index-summary/")
async def index_summary(
    auth=Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    """
    Devuelve el listado de archivos presentes en la colección,
    junto con el número de fragmentos indexados por cada uno.
    """
    try:
        res = col.get(include=["metadatas"])  # recupera todos los metadatos
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
# NUEVO ENDPOINT: VER ALGUNOS FRAGMENTOS DE UN ARCHIVO
# -------------------------------------------------
@app.get("/api/file-fragments/")
async def file_fragments(
    filename: str,
    limit: int = 3,
    auth=Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    """
    Devuelve algunos fragmentos de ejemplo de un archivo concreto.
    Se busca por metadatas.filename == filename.
    """
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
    # docs es una lista de listas en algunas versiones, normalizamos:
    if docs and isinstance(docs[0], list):
        docs = docs[0]

    return {
        "filename": filename,
        "documentos": docs or [],
    }


# -------------------------------------------------
# NUEVO ENDPOINT: ELIMINAR TODOS LOS FRAGMENTOS DE UN ARCHIVO (PROTEGIDO)
# -------------------------------------------------
@app.delete("/api/delete-by-filename/")
async def delete_by_filename(
    filename: str,
    admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    auth=Depends(require_auth),
    col=Depends(get_chroma_collection),
):
    """
    Elimina de la colección todos los fragmentos cuyo metadato
    'filename' coincida con el valor indicado.
    Solo permite eliminar si el token de admin es correcto.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Debe indicarse un nombre de archivo.")

    # Protección por token
    if ADMIN_TOKEN and admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Token de administrador incorrecto para eliminar archivos.",
        )

    try:
        col.delete(where={"filename": filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando en Chroma: {e}")

    return {"message": f"Se han eliminado todos los fragmentos asociados a '{filename}'."}
