
"""
config.py
---------
Centralised configuration module for the Document Intelligence API.

Responsibility:
    Load environment variables from the .env file and expose them
    as typed constants to the rest of the application.

Usage:
    from app.config import GROQ_API_KEY

Note:
    Never hardcode secrets. All sensitive values must live in the .env file.
    The .env file must never be committed to version control.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# Read the API key from environment — never hardcode secrets
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")
