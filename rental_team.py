# rental_team.py

from agno.agent import Agent
from agno.team import Team
from agno.models.groq import Groq
from agents.data_ingestion import ingestion_agent


# Optional: Tools for ingestion
import pdfplumber
import fitz  # PyMuPDF

from PIL import Image
import io

# =====================================
# PDF Ingestion Utility (stub)
# =====================================

def extract_text_from_pdf(file_path: str, use_ocr: bool = False) -> str:
    """
    Extracts text from a PDF using pdfplumber/PyMuPDF.
    
    """
    text_content = ""

    # Try pdfplumber first
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() or ""
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    # If text is empty, try PyMuPDF
    if not text_content.strip():
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text_content += page.get_text()
        except Exception as e:
            print(f"PyMuPDF failed: {e}")

    return text_content

# =====================================
# Agents
# =====================================

ingestion_agent = Agent(
    name="Ingestion Agent",
    role="Extract & preprocess rental agreement data from PDFs.",
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=(
        "Read PDF text and extract structured fields: "
        "Branch ID, City, Yearly Rent, Expiry Date, and other metadata."
    ),
)

indexing_agent = Agent(
    name="Indexing Agent",
    role="Insert embeddings + metadata into vector DB.",
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=(
        "Take structured data, generate embeddings with MiniLM (all-MiniLM-L6-v2), "
        "and insert into Pinecone vector DB with metadata filters."
    ),
)

query_interpreter_agent = Agent(
    name="Query Interpreter Agent",
    role="Understand user query intent.",
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=(
        "Parse user query and classify into categories: "
        "rent retrieval, expiry check, or branch comparison."
    ),
)

retrieval_agent = Agent(
    name="Retrieval Agent",
    role="Query vector DB for relevant docs.",
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=(
        "Search Pinecone DB using embeddings and metadata filters "
        "(city, branch, expiry date)."
    ),
)

reasoning_agent = Agent(
    name="Reasoning Agent",
    role="Aggregate and compute structured answers.",
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=(
        "Take retrieved data, calculate results (rents, expiry summaries, comparisons), "
        "and return structured output (tables, JSON, or text)."
    ),
)

streaming_response_agent = Agent(
    name="Streaming Response Agent",
    role="Stream answer back to chat UI.",
    model=Groq(id="llama-3.1-8b-instant"),
    instructions=(
        "Stream the final structured output progressively to the user in chat."
    ),
)

# =====================================
# Team Setup
# =====================================

rental_team = Team(
    name="Rental Agreement Multi-Agent Team",
    model=Groq(id="llama-3.1-8b-instant"),
    members=[
        ingestion_agent,
        indexing_agent,
        query_interpreter_agent,
        retrieval_agent,
        reasoning_agent,
        streaming_response_agent,
    ],
    instructions="Work together to answer user queries about rental agreements."
)


# =====================================
# Example Run
# =====================================

if __name__ == "__main__":
    # Example: load PDF, pass to ingestion agent
    pdf_path = "data/rentals/rental_agreement_1.pdf"  # Use an actual PDF file
    text = extract_text_from_pdf(pdf_path, use_ocr=False)
    print("Extracted PDF Text (preview):")
    print(text[:500])  # Preview first 500 chars

    # Run team pipeline
    user_query = "What is the yearly rent for the Karachi branch?"
    response = rental_team.run(user_query)

    print("\nTeam Response:\n")
    print(response)
