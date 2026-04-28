# Document Intelligence API (RAG)

A FastAPI-based Q&A bot that answers questions **only from local documents**.
It will never use outside knowledge — only what is inside the `data/` folder.

---

## Project Structure

```
rag_project/
├── app/
│   ├── __init__.py       # marks app as a Python package
│   ├── config.py         # loads OPENAI_API_KEY from .env
│   ├── schemas.py        # Pydantic models for request and response
│   ├── indexer.py        # loads data/, splits into chunks, builds FAISS index
│   └── rag.py            # retrieves relevant chunks and calls the LLM
├── data/
│   ├── company_handbook.txt
│   ├── medical_guidelines.txt
│   ├── movie_plots.txt
│   ├── space_facts.txt
│   └── python_basics.txt
├── main.py               # FastAPI app, lifespan handler, /v1/query endpoint
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## How It Works

1. On startup, the server reads every `.txt` and `.pdf` file from `data/`.
2. Files are split into small overlapping chunks.
3. Chunks are embedded using OpenAI and stored in a **FAISS in-memory vector store**.
4. On each query, the 4 most relevant chunks are retrieved.
5. Those chunks are passed to **GPT-4o-mini** with a strict system prompt.
6. The model answers **only from those chunks**. If the answer is not there, it returns the fallback message.

---

## Setup Instructions

### Step 1 — Check Python version

You need Python 3.11 or newer.

```bash
python --version
```

### Step 2 — Create a virtual environment

```bash
# Create
python -m venv venv

# Activate on Mac / Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real OpenAI API key:

```
OPENAI_API_KEY=sk-your-real-key-here
```

> ⚠️ Do not share or commit your `.env` file. It is already in `.gitignore`.

---

## How to Run the Server

```bash
uvicorn main:app --reload
```

You will see startup logs like this:

```
Server starting up... indexing documents from data/ folder.
  [indexer] Loaded: company_handbook.txt
  [indexer] Loaded: medical_guidelines.txt
  [indexer] Loaded: movie_plots.txt
  [indexer] Loaded: python_basics.txt
  [indexer] Loaded: space_facts.txt
  [indexer] Created 58 chunks from 5 document pages.
  [indexer] FAISS vector store is ready.
Startup complete. Server is ready to accept requests.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## Using the API

### POST `/v1/query`

Send a question in the request body:

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What is the annual leave policy?"}'
```

**Response:**

```json
{
  "answer": "Full-time employees are entitled to 20 days of paid annual leave per calendar year. Leave must be requested at least 5 business days in advance via the HR portal.",
  "sources": [
    {
      "source": "data/company_handbook.txt",
      "excerpt": "Full-time employees are entitled to 20 days of paid annual leave..."
    }
  ]
}
```

**When the answer is not in the documents:**

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What is the best pizza topping?"}'
```

```json
{
  "answer": "I'm sorry, but the provided documents do not contain information to answer this query.",
  "sources": [...]
}
```

### GET `/health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### Interactive Swagger UI

Open in your browser: **http://localhost:8000/docs**

---

## Sample Questions to Try

| Question | Source Document |
|---|---|
| What is the annual leave policy? | company_handbook.txt |
| How many sick days do employees get? | company_handbook.txt |
| What is the first-line treatment for Type 2 diabetes? | medical_guidelines.txt |
| What is the HbA1c target for elderly patients? | medical_guidelines.txt |
| What is the plot of Inception? | movie_plots.txt |
| How did Andy Dufresne escape from Shawshank? | movie_plots.txt |
| What is the fastest wind speed in the solar system? | space_facts.txt |
| When was the ISS first continuously inhabited? | space_facts.txt |
| How do you handle errors in Python? | python_basics.txt |
| What is a dictionary in Python? | python_basics.txt |
