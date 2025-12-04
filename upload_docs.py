# upload_docs.py
import os
import fitz              # PyMuPDF, para PDF
import docx              # python-docx, para DOCX
import requests

API_URL = "http://127.0.0.1:8000/api/documents/"  # Tu endpoint FastAPI
DOCS_DIR = "./docs"                               # Carpeta donde tienes los archivos


def leer_pdf(ruta: str) -> str:
    texto = ""
    with fitz.open(ruta) as pdf:
        for pagina in pdf:
            texto += pagina.get_text()
    return texto


def leer_docx(ruta: str) -> str:
    texto = ""
    doc = docx.Document(ruta)
    for p in doc.paragraphs:
        texto += p.text + "\n"
    return texto


def trocear(texto: str, max_chars: int = 800):
    """Divide el texto en trozos de tamaño máximo max_chars."""
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


def main():
    ids = []
    documents = []
    metadatas = []

    # Recorremos todos los archivos de ./docs
    for filename in os.listdir(DOCS_DIR):
        ruta = os.path.join(DOCS_DIR, filename)

        if filename.lower().endswith(".pdf"):
            contenido = leer_pdf(ruta)
        elif filename.lower().endswith(".docx"):
            contenido = leer_docx(ruta)
        else:
            # Si en el futuro añades .txt, etc., puedes ampliarlo aquí
            print(f"Saltando archivo no soportado: {filename}")
            continue

        fragmentos = trocear(contenido, max_chars=800)

        for i, frag in enumerate(fragmentos):
            doc_id = f"{filename}_{i}"
            ids.append(doc_id)
            documents.append(frag)
            metadatas.append({
                "filename": filename,
                "chunk": i
            })

    if not ids:
        print("No se encontraron documentos válidos en ./docs")
        return

    payload = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas
    }

    print(f"Enviando {len(ids)} fragmentos al endpoint {API_URL}...")
    resp = requests.post(API_URL, json=payload)

    print("Código de respuesta:", resp.status_code)
    print("Respuesta Chroma:", resp.text)


if __name__ == "__main__":
    main()
