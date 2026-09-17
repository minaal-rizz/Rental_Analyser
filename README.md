🏢 Rental Agreement Assistant
An AI-powered assistant for querying and analyzing rental agreements.
This system uses LLMs (Groq LLaMA models), Pinecone vector database, and Streamlit to let users ask natural language questions about rental agreements, retrieve relevant contracts, and receive structured reasoning-based answers.
✨ Features
📄 Synthetic Rental Agreements Generator
Creates 1,000+ fake rental agreements in PDF format with metadata (CSV + JSON).
🧭 Query Interpreter Agent
Classifies queries into intents:
rent_retrieval
expiry_check
comparison
general_search
Extracts filters like city, branch, expiry, rent range.
🔍 Retrieval Agent
Embeds queries with Sentence-Transformers (MiniLM).
Fetches top-k relevant documents from Pinecone.
🧠 Reasoning Agent
Aggregates and reasons over retrieved agreements.
Generates structured, human-readable answers with reasoning steps.
⚡ Streaming Agent
Streams AI-generated answers token-by-token for real-time feedback.
🎨 Streamlit Frontend
Simple UI for typing questions and getting answers.
Displays parsed intents, retrieved agreements, and live-streamed answers.
📂 Project Structure
```rental_analyser/
│── app.py                  # Streamlit app entrypoint
│── app_ui.py               # UI logic with chat pipeline
│── chat_pipeline.py        # Main pipeline (interpret → retrieve → reason)
│
├── agents/
|   |__data_ingestion.py
│   ├── query.py  # Interprets queries
│   ├── retrieval.py          # Retrieves docs from Pinecone
│   ├── reasoning.py                # Aggregates & reasons over data
│   ├── streaming.py          # Streams answers
|   |__orchestrator.py
│
├── data/
│   ├── rentals/                    # Generated PDF rental agreements
│   ├── rental_metadata.csv         # Metadata (CSV)
│   └── rental_metadata.json        # Metadata (JSON)
│
├── rental_doc.py             # Script to generate fake agreements
|__rental_team.py
├── requirements.txt         # Python dependencies
└── README.md                # 
```
Project documentation
⚙️ Installation
1️⃣ Clone the repo
git clone https://github.com/minaal-rizz/Rental-Analyser.git
cd rental-analyser
2️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Set environment variables
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
📊 Generate Rental Agreements
Run the generator script to create 1,100 fake agreements:
python rental_docs.py
Agreements are saved in: data/rentals/
Metadata is stored in:
data/rental_metadata.csv
data/rental_metadata.json
🚀 Run the App
Launch the Streamlit app:
streamlit run app.py
Example Queries:
"Show me all agreements expiring in 2025"
"Which branch in Karachi has the highest rent?"
"List Lahore branches with yearly rent above 2M PKR."
🧩 Tech Stack
🤖 LLM: Groq LLaMA 3.1
📦 Vector DB: Pinecone
🔑 Embeddings: Sentence-Transformers (MiniLM)
🎛 Frontend: Streamlit
🏗 Synthetic Data: Faker + ReportLab
✅ Example Workflow
User enters:
Which branch in Karachi has the highest rent?
Query Interpreter:
{"intent": "comparison", "filters": {"city": "Karachi"}}
Retrieval Agent:
Finds top rental agreements in Karachi.
Reasoning Agent:
Returns:
### Answer  
The highest rent in Karachi is **Branch-87** with **PKR 4,500,000**.  

### Reasoning  
- Retrieved 5 agreements for Karachi.  
- Compared yearly rents.  
- Found Branch-87 has the maximum rent.  
