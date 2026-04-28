import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# Read the API key from environment — never hardcode secrets
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")
