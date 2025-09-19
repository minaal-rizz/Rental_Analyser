"""
data_ingestion.py

This module handles the ingestion of rental agreement PDFs.
It extracts metadata (branch, city, rent, expiry date),
embeds the content using MiniLM, and indexes it into Pinecone.
It also defines the `ingestion_agent` for use in the Agno multi-agent system.
"""

import os
import pdfplumber
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
import uuid
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from agno.agent import Agent
from agno.models.groq import Groq

# -------------------------
# Setup & Clients
# -------------------------
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pinecone_api = os.getenv("PINECONE_API_KEY")

# Load MiniLM embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to Pinecone
pc = Pinecone(api_key=pinecone_api)
index_name = "rental-agreements"

# Check if index exists, create if not (MiniLM has 384-dim vectors)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# -------------------------
# Agent: Data Loader Agent
# -------------------------
def extract_metadata_from_pdf(pdf_path):
    """
    Extract metadata from a given rental agreement PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        tuple:
            - text (str): Full extracted text from the PDF.
            - metadata (dict): Extracted metadata containing:
                * branch_name (str or None)
                * city (str or None)
                * yearly_rent (int or None)
                * expiry_date (str or None)
    """
    metadata = {}
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

        # Regex-based extraction (since we know PDF format)
        branch_match = re.search(r"Rental Agreement - (.+)", text)
        city_match = re.search(r"City: (.+)", text)
        rent_match = re.search(r"Yearly Rent: PKR ([\d,]+)", text)
        expiry_match = re.search(r"Expiry Date: ([\d-]+)", text)

        metadata["branch_name"] = branch_match.group(1) if branch_match else None
        metadata["city"] = city_match.group(1) if city_match else None
        metadata["yearly_rent"] = int(rent_match.group(1).replace(",", "")) if rent_match else None
        metadata["expiry_date"] = expiry_match.group(1) if expiry_match else None

    return text, metadata


# -------------------------
# Agent: Vector Store Agent
# -------------------------
def embed_and_store(pdf_path):
    """
    Process a rental agreement PDF:
    - Extract metadata and text
    - Generate embeddings using MiniLM
    - Store the document in Pinecone with metadata

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: Metadata dictionary extracted from the PDF.
    """
    text, metadata = extract_metadata_from_pdf(pdf_path)

    # Generate MiniLM embeddings
    embedding = embedder.encode(text).tolist()  # 384-dim vector

    # Unique ID for Pinecone
    doc_id = str(uuid.uuid4())

    # Store in Pinecone
    index.upsert([(doc_id, embedding, metadata)])

    return metadata


# -------------------------
# Main Script
# -------------------------
def main():
    """
    Main pipeline for ingestion:
    - Finds all PDFs in `data/rentals/`
    - Extracts, embeds, and indexes them into Pinecone
    """
    rental_dir = "data/rentals"
    files = [os.path.join(rental_dir, f) for f in os.listdir(rental_dir) if f.endswith(".pdf")]

    print(f"📂 Found {len(files)} rental agreements.")

    for i, pdf_path in enumerate(files, start=1):
        metadata = embed_and_store(pdf_path)
        print(f"[{i}/{len(files)}] Indexed {metadata['branch_name']} ({metadata['city']})")

    print("✅ All agreements indexed into Pinecone.")


# -------------------------
# Agent Definition
# -------------------------
ingestion_agent = Agent(
    name="Ingestion Agent",
    role="Extract & preprocess rental agreement data from PDFs.",
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=(
        "Read PDF text and extract structured fields: "
        "Branch Name, City, Yearly Rent, Expiry Date, and other metadata."
    ),
)

if __name__ == "__main__":
    main()
