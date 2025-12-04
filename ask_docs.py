# ask_docs.py
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# 1. Conectar con la base de datos persistente (solo si usas Chroma local)
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(allow_reset=True)
)

# 2. Misma función de embeddings que en el backend
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 3. Misma colección
collection = client.get_or_create_collection(
    name="mis_documentos",
    embedding_function=embedder
)


def preguntar():
    while True:
        pregunta = input("\nEscribe tu pregunta (o 'salir'): ").strip()
        if not pregunta or pregunta.lower() == "salir":
            break

        results = collection.query(
            query_texts=[pregunta],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        documentos = results["documents"][0]
        metadatas = results["metadatas"][0]
        distancias = results["distances"][0]

        print("\n--- FRAGMENTOS MÁS RELEVANTES ---")
        for doc, meta, dist in zip(documentos, metadatas, distancias):
            print("\n------------------------------------")
            print(f"Archivo : {meta.get('filename')}")
            print(f"Fragmento: {meta.get('chunk')}")
            print(f"Distancia: {dist:.4f}")
            print("------------------------------------")
            print(doc)

        print("\n(Usa estos fragmentos para responder, sin inventar nada.)")


if __name__ == "__main__":
    preguntar()
