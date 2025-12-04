# upload_images.py
import os
import base64
import mimetypes
import requests

from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno (.env)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

IMAGES_DIR = "./images"  # Carpeta donde pondrás tus PNG/JPG
API_URL = "http://127.0.0.1:8000/api/documents/"  # Tu endpoint FastAPI


def describir_imagen(ruta: str) -> str:
    """
    Envía la imagen al modelo de OpenAI (con visión) y devuelve
    una descripción detallada en texto.
    """
    mime, _ = mimetypes.guess_type(ruta)
    if mime is None:
        mime = "image/png"

    with open(ruta, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:{mime};base64,{b64}"

    prompt = (
        "Describe con detalle el contenido de esta imagen en español. "
        "Incluye cualquier texto legible, tablas, diagramas o datos importantes. "
        "No inventes nada que no se vea claramente."
    )

    # CORRECCIÓN: cambiar input_image → image_url
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",  # Modelo con visión
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
        temperature=0.0,
    )

    return resp.choices[0].message.content.strip()


def main():
    ids = []
    documents = []
    metadatas = []

    if not os.path.isdir(IMAGES_DIR):
        print(f"La carpeta {IMAGES_DIR} no existe.")
        return

    for filename in os.listdir(IMAGES_DIR):
        ruta = os.path.join(IMAGES_DIR, filename)

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Saltando archivo no soportado: {filename}")
            continue

        print(f"Procesando imagen: {filename}...")
        try:
            descripcion = describir_imagen(ruta)
        except Exception as e:
            print(f"Error describiendo {filename}: {e}")
            continue

        if not descripcion.strip():
            print(f"Descripción vacía para {filename}, se omite.")
            continue

        # Guardamos en Chroma como si fuera un documento más
        doc_id = f"img_{filename}"
        ids.append(doc_id)
        documents.append(descripcion)
        metadatas.append(
            {
                "filename": filename,
                "tipo": "imagen",
            }
        )

    if not ids:
        print("No se han generado documentos a partir de imágenes.")
        return

    payload = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }

    print(f"Enviando {len(ids)} descripciones de imágenes a {API_URL}...")
    resp = requests.post(API_URL, json=payload)
    print("Código de respuesta:", resp.status_code)
    print("Respuesta:", resp.text)


if __name__ == "__main__":
    main()
