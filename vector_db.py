import chromadb

chroma_client = chromadb.Client()

collection_name = "test_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

documents = [
    {"id": "doc1", "text": "Hello world! This is the content of document 1."},
    {"id": "doc2", "text": "How are you doing today?"},
    {"id": "doc3", "text": "Goodbye! See you later."},
]

for doc in documents:
    collection.upsert(
        documents=[doc["text"]],
        metadatas=[{"source": doc["id"]}],
        ids=[doc["id"]],
    )

query = "Hello world!"
result = collection.query(query_texts=[query], n_results=3)

print("Query Results:", result)