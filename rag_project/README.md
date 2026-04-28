# Document Intelligence API

A **FastAPI-based RAG (Retrieval-Augmented Generation)** engine that acts as a
specialised Q&A bot. It answers questions **exclusively from a local set of documents**.
It will never use outside knowledge — only what is inside the `data/` folder.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Functional Flow](#functional-flow)
3. [Project Structure](#project-structure)
4. [Tech Stack](#tech-stack)
5. [Setup Instructions](#setup-instructions)
6. [How to Run](#how-to-run)
7. [Using the API](#using-the-api)
8. [Sample Questions](#sample-questions)
9. [Where is the Vector Data Stored?](#where-is-the-vector-data-stored)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser / curl)                      │
│                     POST /v1/query  {"user_query": "..."}           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  HTTP Request
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py — FastAPI                           │
│                                                                     │
│   • Pydantic validates the request body (QueryRequest)              │
│   • Checks vector store is ready                                    │
│   • Calls get_rag_answer() from rag.py                              │
│   • Returns QueryResponse (Pydantic)                                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       rag.py — RAG Pipeline                         │
│                                                                     │
│   Step 1 │ FAISS Retriever → top-4 relevant chunks                  │
│   Step 2 │ Format chunks into numbered context block                │
│   Step 3 │ Build full prompt: SYSTEM_PROMPT + context + question    │
│   Step 4 │ await llm.ainvoke() → Groq API (llama-3.3-70b)          │
│   Step 5 │ Package answer + sources into QueryResponse              │
└──────┬──────────────────────────────────────────┬───────────────────┘
       │                                          │
       │ similarity search                        │ async LLM call
       ▼                                          ▼
┌─────────────────┐                   ┌──────────────────────────────┐
│  FAISS Index    │                   │   Groq Cloud API             │
│  (in RAM only)  │                   │   llama-3.3-70b-versatile    │
│                 │                   │   temperature = 0            │
│  Built once at  │                   │   Grounded by system prompt  │
│  server startup │                   └──────────────────────────────┘
└─────────────────┘
       ▲
       │ built at startup
       │
┌─────────────────────────────────────────────────────────────────────┐
│                    indexer.py — Indexing Pipeline                   │
│                                                                     │
│   Step 1 │ load_documents()                                         │
│           │  TextLoader  → .txt files                               │
│           │  PyPDFLoader → .pdf files                               │
│                                                                     │
│   Step 2 │ RecursiveCharacterTextSplitter                           │
│           │  chunk_size=500, chunk_overlap=50                       │
│                                                                     │
│   Step 3 │ HuggingFaceEmbeddings('all-MiniLM-L6-v2')               │
│           │  Runs locally — no API key needed                       │
│                                                                     │
│   Step 4 │ FAISS.from_documents()  → in-memory vector index        │
│           │  asyncio.to_thread() keeps event loop free              │
└─────────────────────────────────────────────────────────────────────┘
       ▲
       │ reads files from
       │
┌─────────────────┐
│   data/         │
│  *.txt  *.pdf   │
└─────────────────┘
```

---

## Functional Flow

### Startup Flow (runs once when server starts)

```
uvicorn main:app --reload
        │
        ▼
lifespan() called
        │
        ▼
build_vector_store("data")
        │
        ├── load_documents()
        │       ├── TextLoader  → reads every .txt file
        │       └── PyPDFLoader → reads every .pdf file
        │
        ├── RecursiveCharacterTextSplitter
        │       └── splits pages into 500-char chunks (50-char overlap)
        │
        ├── HuggingFaceEmbeddings("all-MiniLM-L6-v2")
        │       └── converts each chunk into a numeric vector (embedding)
        │
        └── FAISS.from_documents()
                └── stores all vectors in RAM as a searchable index
                    → saved in app_state["vector_store"]

Server is now ready to accept requests
```

### Query Flow (runs on every POST /v1/query)

```
Client sends:  POST /v1/query  {"user_query": "What is annual leave?"}
        │
        ▼
Pydantic validates QueryRequest
        │
        ▼
get_rag_answer(user_query, vector_store)
        │
        ├── STEP 1: FAISS similarity search
        │       └── finds top-4 chunks most similar to the question
        │
        ├── STEP 2: Context assembly
        │       └── formats chunks as:
        │               [Context 1]  <chunk text>
        │               [Context 2]  <chunk text>  ...
        │
        ├── STEP 3: Prompt construction
        │       └── SYSTEM_PROMPT + context block + user question
        │
        ├── STEP 4: Groq LLM call (async)
        │       └── await llm.ainvoke(full_prompt)
        │           Model is grounded — answers ONLY from context
        │           If context insufficient → returns exact fallback sentence
        │
        └── STEP 5: Response packaging
                └── QueryResponse(answer=..., sources=[...])

Client receives:
{
    "answer": "Employees are entitled to 20 days of annual leave.",
    "sources": [
        { "source": "data/company_handbook.txt", "excerpt": "..." }
    ]
}
```

---

## Project Structure

```
rag_project/
│
├── app/                        ← Python package — all business logic
│   ├── __init__.py             ← marks app/ as a package
│   ├── config.py               ← loads GROQ_API_KEY from .env
│   ├── schemas.py              ← Pydantic models: QueryRequest, QueryResponse
│   ├── indexer.py              ← document loading, chunking, FAISS indexing
│   └── rag.py                  ← retrieval + LLM call pipeline
│
├── data/                       ← knowledge base (your documents go here)
│   ├── company_handbook.txt
│   ├── medical_guidelines.txt
│   ├── movie_plots.txt
│   ├── python_basics.txt
│   └── space_facts.txt
│
├── main.py                     ← FastAPI app, lifespan, /v1/query endpoint
├── requirements.txt            ← all Python dependencies
├── .env.example                ← template — copy to .env and add your key
├── .gitignore                  ← excludes .env and __pycache__
└── README.md                   ← this file
```

### What each file does

| File | Single Responsibility |
|---|---|
| `app/config.py` | Read `GROQ_API_KEY` from `.env` — nothing else |
| `app/schemas.py` | Define the shape of API request and response JSON |
| `app/indexer.py` | Load files → split → embed → build FAISS index |
| `app/rag.py` | Retrieve chunks → build prompt → call LLM → return answer |
| `main.py` | Register FastAPI app, lifespan handler, and HTTP endpoints |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API Framework | FastAPI + Uvicorn | Async-native, auto Swagger docs, Pydantic built-in |
| LLM | Groq — llama-3.3-70b-versatile | Fast inference, free API key available |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Local, free, no API key, ~80 MB |
| Vector Store | FAISS (in-memory) | Fast similarity search, no database setup needed |
| RAG Framework | LangChain | Document loaders, splitters, retriever abstraction |
| Validation | Pydantic v2 | Request/response schema, automatic error messages |
| Config | python-dotenv | Load secrets from .env, never hardcode keys |

---

## Setup Instructions

### Prerequisites

- Python 3.11 or newer
- A free Groq API key from https://console.groq.com

### Step 1 — Check Python version

```bash
python --version
# Should show Python 3.11.x or higher
```

### Step 2 — Create a virtual environment

```bash
# Create
python -m venv venv

# Activate on Mac / Linux
source venv/bin/activate

# Activate on Windows (PowerShell)
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> Note: On first run, `sentence-transformers` will download the
> `all-MiniLM-L6-v2` model (~80 MB). This only happens once.

### Step 4 — Set your API key

```bash
# Mac / Linux
cp .env.example .env

# Windows (PowerShell)
copy .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_your-real-key-here
```

> Get a free key at https://console.groq.com → API Keys → Create API Key

> ⚠️ Never commit your `.env` file. It is already in `.gitignore`.

---

## How to Run

```bash
uvicorn main:app --reload
```

### Expected startup output

```
Server starting up... indexing documents from data/ folder.
  [indexer] Loaded: company_handbook.txt
  [indexer] Loaded: PRINCE NSS.pdf
  [indexer] Loaded: Seminar.pdf
  [indexer] Loaded: python_basics.txt
  [indexer] Created 32 chunks from 5 document pages.
  [indexer] FAISS vector store is ready.
Startup complete. Server is ready to accept requests.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## Using the API

### Option 1 — Swagger UI (easiest)

Open your browser: **http://127.0.0.1:8000/docs**

Click `POST /v1/query` → **Try it out** → type your question → **Execute**

---

### Option 2 — PowerShell (Windows)

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/v1/query" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"user_query": "What is the annual leave policy?"}'
```

---

### Option 3 — curl (Mac / Linux / Git Bash)

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What is the annual leave policy?"}'
```

---

### Successful response

```json
{
  "answer": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
  "sources": [
    {
      "source": "data/company_handbook.txt",
      "excerpt": "Full-time employees are entitled to 20 days of paid annual leave..."
    }
  ]
}
```

### Grounding fallback (answer not in documents)

```json
{
  "answer": "I'm sorry, but the provided documents do not contain information to answer this query.",
  "sources": [...]
}
```

### Health check

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

---

## Sample Questions

| Question | Source Document |
|---|---|
| What is the annual leave policy? | company_handbook.txt |
| How many sick days do employees get? | company_handbook.txt |
| What benefits does the company provide? | company_handbook.txt |
| What is the plot of Inception? | movie_plots.txt |
| How did Andy Dufresne escape from Shawshank? | movie_plots.txt |
| What are the themes of Parasite? | movie_plots.txt |
| What is Prince Kushwaha's USN number? | PRINCE_NSS.pdf |
| What was the objective of the water management system activity? | PRINCE_NSS.pdf |
| Which school did the educational outreach program visit? | PRINCE_NSS.pdf |
| Where is ACS College of Engineering located? | Seminar.pdf |
| What is the time complexity of Grover's algorithm? | Seminar.pdf |
| How do you handle errors in Python? | python_basics.txt |
| What is a dictionary in Python? | python_basics.txt |
| What is the capital of France? *(not in docs)* | → fallback message |

---

## Where is the Vector Data Stored?

**Nowhere on disk — only in RAM.**

The FAISS index lives inside the `app_state` Python dictionary in `main.py`.
When the server stops, the index is gone. When it restarts, it is rebuilt
from scratch by reading the `data/` folder again.

This is intentional — the assessment specifies an in-memory index is acceptable.

If you ever want to persist it to disk, you can add:

```python
# Save
vector_store.save_local("faiss_index")

# Load
vector_store = FAISS.load_local("faiss_index", embeddings)
```

This would create a `faiss_index/` folder. But this is not needed here.
