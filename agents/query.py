# query_interpreter_agent.py
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq

# -------------------------
# Setup
# -------------------------
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# -------------------------
# Query Interpreter Agent
# -------------------------
query_interpreter_agent = Agent(
    name="QueryInterpreterAgent",
    model=Groq(id="llama-3.1-8b-instant", api_key=groq_api),
    description="Parses user queries and determines intent (rent retrieval, expiry check, comparison, general search).",
    instructions=[
        "You are a query interpreter for rental agreements.",
        "Take a natural language query from the user.",
        "Identify the intent: rent_retrieval, expiry_check, comparison, or general_search.",
        "Extract important filters: city, branch_name, expiry_date, rent range.",
        "Output a JSON object with 'intent' and 'filters'.",
        "Do NOT answer the query yourself — only classify and extract filters.",
    ],
)


# -------------------------
# Tool: Interpret Query
# -------------------------
def interpret_query(query: str) -> dict:
    """
    Runs the query through the Query Interpreter Agent to extract intent and filters.

    Workflow:
        - Accepts a natural language query.
        - Passes it to the interpreter agent.
        - Ensures structured output (intent + filters).

    Args:
        query (str): The raw user query in natural language.

    Returns:
        dict: A dictionary with the parsed result.
              Example:
              {
                  'intent': 'rent_retrieval',
                  'filters': {'city': 'Karachi'}
              }
    """
    response = query_interpreter_agent.run(
        f"User query: {query}\n\n"
        "Return ONLY a valid JSON with keys 'intent' and 'filters'."
    )
    return {"parsed": response}


# -------------------------
# Test Script
# -------------------------
if __name__ == "__main__":
    """
    Run sample queries to test query interpretation.
    Prints intent classification and extracted filters.
    """
    queries = [
        "Show me all agreements expiring in 2025",
        "Which branch in Karachi has the highest rent?",
    ]

    for q in queries:
        result = interpret_query(q)
        print(f"❓ Query: {q}")
        print(f"🔎 Parsed: {result['parsed']}\n")
