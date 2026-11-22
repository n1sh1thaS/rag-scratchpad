from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Documents with both semantic content AND specific identifiers
documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports "
        "gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the "
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
        "rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. "
        "Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel "
        "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
        "alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]

print(f"Loaded {len(documents)} documents")


vectorstore = Chroma.from_documents(
    documents, embeddings, collection_name="hybrid_test"
)


vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Return top 3

bm25_retriever = BM25Retriever.from_documents(documents, k=3)  # Return top 3

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])


def test_query(query: str, name, retriever):
    print(f"\n{name} - Query: {query}")
    results = retriever.invoke(query)
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f" {i + 1}. {preview}")
    return results


if __name__ == "__main__":
    test_queries = [
        "SKU-7742X specifications",
        "E_CONN_REFUSED error",
        "How do I authenticate?", 
        "WCAG compliance", 
        "router configuration",
    ]

    for query in test_queries:
        print("=" * 60)

        # Vector only
        vector_results = test_query(query, "VECTOR", vector_retriever)

        # BM25 only
        bm25_results = test_query(query, "BM25", bm25_retriever)

        # Hybrid
        hybrid_results = test_query(query, "HYBRID", ensemble_retriever)

