import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()

def get_llm():
    ollama = OllamaLLM(
        model= os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", 0.1))
    )
    
    return ollama