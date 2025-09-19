# retrieval_agent.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from agno.agent import Agent
from agno.models.groq import Groq  # can be swapped for Groq later

# -------------------------
# Setup
# -------------------------
load_dotenv()

pinecone_api = os.getenv("PINECONE_API_KEY")
index_name = "rental-agreements"

# Pinecone client
pc = Pinecone(api_key=pinecone_api)
index = pc.Index(index_name)

# Embedding model for encoding queries
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -------------------------
# Retrieval Function
# -------------------------
def query_pinecone(query, top_k=5):
    """
    Embed a query and retrieve top-k similar rental agreements from Pinecone.

    Steps:
        1. Convert the input query into a vector using MiniLM embeddings.
        2. Query Pinecone index with this embedding.
        3. Return top-k results with metadata.

    Args:
        query (str): Natural language user query.
        top_k (int): Number of results to retrieve.

    Returns:
        dict: Pinecone query results, including metadata and similarity scores.
    """
    query_embedding = embedder.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results


# -------------------------
# Retrieval Agent (Agno)
# -------------------------
retrieval_agent = Agent(
    name="RetrievalAgent",
    model=Groq(id="llama-3.1-8b-instant"),  # currently using Groq
    description=(
        "This agent retrieves rental agreement documents from Pinecone "
        "based on semantic similarity with the input query."
    ),
    instructions=[
        "Take a user query.",
        "Embed it with MiniLM.",
        "Query Pinecone for top-k relevant documents.",
        "Return metadata + similarity scores in a structured format."
    ],
)


def retrieve_agreements(query: str, top_k: int = 5):
    """
    Retrieve rental agreements from Pinecone given a query.

    Args:
        query (str): Natural language user query.
        top_k (int): Number of documents to return.

    Returns:
        list[dict]: List of rental agreements with fields:
            - branch_name
            - city
            - yearly_rent
            - expiry_date
            - score (similarity to query)
    """
    results = query_pinecone(query, top_k)
    return [
        {
            "branch_name": match["metadata"].get("branch_name"),
            "city": match["metadata"].get("city"),
            "yearly_rent": match["metadata"].get("yearly_rent"),
            "expiry_date": match["metadata"].get("expiry_date"),
            "score": match["score"],
        }
        for match in results["matches"]
    ]


# Attach tool to agent
retrieval_agent.tools = [retrieve_agreements]


# -------------------------
# Test Script
# -------------------------
if __name__ == "__main__":
    """
    Run a test retrieval query to ensure Pinecone + embeddings work.
    Prints top-3 results for a sample query.
    """
    query = "Which branches in Karachi have rent above 1,000,000?"
    docs = retrieve_agreements(query, top_k=3)
    print("🔍 Retrieved agreements:")
    for d in docs:
        print(d)
