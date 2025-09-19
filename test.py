import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load env
load_dotenv()
pinecone_api = os.getenv("PINECONE_API_KEY")

# Connect with new Pinecone client
pc = Pinecone(api_key=pinecone_api)

# Index name (must match the one you created during ingestion)
index_name = "rental-agreements"
index = pc.Index(index_name)

# Load MiniLM embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Helper Functions
# -------------------------
def get_embedding(query: str):
    """Convert query into MiniLM embedding."""
    return embedder.encode(query).tolist()

def query_rentals(user_query: str, top_k: int = 5):
    """Query Pinecone with user query and return top matches."""
    embedding = get_embedding(user_query)

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results


# -------------------------
# Main Test
# -------------------------
if __name__ == "__main__":
    test_queries = [
        "yearly rentals of all branches in Lahore",
        "branch with rental expiry coming soon",
        "highest rent branch in Islamabad"
    ]

    for q in test_queries:
        print(f"\n🔎 Query: {q}")
        results = query_rentals(q, top_k=3)

        for match in results["matches"]:
            meta = match["metadata"]
            print(f"  ➡ {meta['branch_id']} ({meta['city']}) | Rent: {meta['yearly_rent']} | Expiry: {meta['expiry_date']}")
