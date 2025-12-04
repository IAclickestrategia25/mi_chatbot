import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import docx  # de python-docx

# 1. Configurar cliente de Chroma (base de datos en disco)
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(allow_reset=True)
)

# 2. Definir función de embeddings local (sin API)
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # Modelo ligero y bastante bueno
)

# 3. Crear (o recuperar) la colección donde guardaremos los textos
collection = client.get_or_create_collection(
    name="mis_documentos",
    embedding_function=embedder
)

# OPCIONAL: borrar lo anterior y reconstruir desde cero
existing = collection.get()
if existing["ids"]:
    collection.delete(ids=existing["ids"])

# 4. Carpeta donde están tus documentos
DOCS_DIR = "./docs"


def cargar_texto_desde_archivo(ruta: str) -> str:
    """Devuelve el texto de un .txt o .docx."""
    ext = os.path.splitext(ruta)[1].lower()

    if ext == ".txt":
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".docx":
        doc = docx.Document(ruta)
        # Unimos todos los párrafos con saltos de línea
        return "\n".join(p.text for p in doc.paragraphs)

    # Si no es un tipo soportado, devolvemos cadena vacía
    return ""


def trocear_texto(texto: str, max_chars: int = 800):
    """
    Divide el texto en trozos de tamaño máximo max_chars,
    intentando cortar en saltos de línea.
    """
    trozos = []
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


ids = []
docs = []
metadatas = []

# 5. Recorrer todos los archivos de la carpeta docs (.txt y .docx)
for filename in os.listdir(DOCS_DIR):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".txt", ".docx"]:
        continue

    ruta = os.path.join(DOCS_DIR, filename)
    contenido = cargar_texto_desde_archivo(ruta)

    if not contenido.strip():
        continue

    # trocear el documento en fragmentos manejables
    fragmentos = trocear_texto(contenido, max_chars=800)

    for i, frag in enumerate(fragmentos):
        doc_id = f"{filename}_{i}"
        ids.append(doc_id)
        docs.append(frag)
        metadatas.append({
            "filename": filename,
            "chunk": i
        })

# 6. Guardar en Chroma
if ids:
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas
    )
    print(f"Indexados {len(ids)} fragmentos procedentes de {len(set(m['filename'] for m in metadatas))} archivos.")
else:
    print("No se encontraron documentos para indexar.")
