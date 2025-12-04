# main.py
import os
import base64
import mimetypes
from io import BytesIO, StringIO
from typing import List, Dict, Any

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
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from chroma_connection import get_chroma_collection




# -------------------------------------------------
# CARGA .env Y CLIENTE OPENAI
# -------------------------------------------------
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------------------------
# APP FASTAPI + CORS + STATIC
# -------------------------------------------------
app = FastAPI(title="ChromaDB FastAPI Integration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # en local está bien
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Sirve el frontend en la raíz "/"
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")



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
    # 1) Pasar de bytes -> str
    texto = data.decode(encoding, errors="ignore")

    # 2) Usar StringIO (modo texto) con csv.reader
    f = StringIO(texto)
    reader = csv.reader(f)

    filas = []
    for row in reader:
        # row es una lista de strings: ["Juan", "Pérez", "35", "Bilbao", ...]
        linea = " ; ".join(col.strip() for col in row if col is not None)
        if linea.strip():
            filas.append(linea)

    # Devolvemos un único texto con saltos de línea
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
# ENDPOINT: SUBIR PDFs, DOCX, PNG, JPG, CSV DESDE EL NAVEGADOR
# -------------------------------------------------
@app.post("/api/upload-files/")
async def upload_files(
    files: List[UploadFile] = File(...),
    col=Depends(get_chroma_collection),
):
    """
    Recibe ficheros (PDF, DOCX, PNG, JPG/JPEG, CSV) desde el navegador,
    extrae el texto o la descripción y los indexa directamente en Chroma.
    """
    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    if not files:
        raise HTTPException(status_code=400, detail="No se ha enviado ningún archivo.")

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

        for i, frag in enumerate(fragmentos):
            doc_id = f"{filename}_{i}"
            ids.append(doc_id)
            docs.append(frag)
            metadatas.append(
                {
                    "filename": filename,
                    "chunk": i,
                    "tipo": tipo,
                }
            )

    if not ids:
        raise HTTPException(
            status_code=400,
            detail="No se ha podido extraer información útil de los archivos subidos.",
        )

    try:
        col.add(ids=ids, documents=docs, metadatas=metadatas)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error guardando en Chroma: {e}",
        )

    return {
        "message": "Archivos procesados e indexados correctamente.",
        "total_archivos": len(files),
        "total_fragmentos": len(ids),
    }


# -------------------------------------------------
# ENDPOINT: HACER PREGUNTAS A TUS DOCUMENTOS
@app.post("/api/ask/")
async def ask_documents(body: AskBody, col=Depends(get_chroma_collection)):
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
        "fragmentos_usados": [doc for doc, _, _ in filtrados],
        "distancias": [float(dist) for _, _, dist in filtrados],
    }



# -------------------------------------------------
# NUEVO ENDPOINT: RESUMEN DEL ÍNDICE (archivos + nº fragmentos)
# -------------------------------------------------
@app.get("/api/index-summary/")
async def index_summary(col=Depends(get_chroma_collection)):
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
# NUEVO ENDPOINT: ELIMINAR TODOS LOS FRAGMENTOS DE UN ARCHIVO
# -------------------------------------------------
@app.delete("/api/delete-by-filename/")
async def delete_by_filename(
    filename: str,
    col=Depends(get_chroma_collection),
):
    """
    Elimina de la colección todos los fragmentos cuyo metadato
    'filename' coincida con el valor indicado.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Debe indicarse un nombre de archivo.")

    try:
        # Borrado por filtro de metadatos
        col.delete(where={"filename": filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando en Chroma: {e}")

    return {"message": f"Se han eliminado todos los fragmentos asociados a '{filename}'."}
