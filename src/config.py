import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file (primarily for local development)
load_dotenv()

class Config:
    # 1. Base Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_PATH = BASE_DIR / "dataset" / "campushire_jd_dataset.csv"
    
    # FAISS configuration
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    FAISS_INDEX_PATH = BASE_DIR / FAISS_INDEX_DIR

    # Model parameters
    LLM_MODEL_NAME = "meta/llama-3.3-70b-instruct"  # High-reasoning NVIDIA model
    EMBEDDING_MODEL_NAME = "gemini-embedding-2-preview" # Handled separately via Google

    # 2. Key validation and Fallbacks (System Env ➔ Streamlit Secrets)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        try:
            import streamlit as st
            GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    if not NVIDIA_API_KEY:
        try:
            import streamlit as st
            NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY")
        except Exception:
            pass

    @classmethod
    def get_llm(cls, temperature: float = 0.0):
        """
        Centralized LLM Factory. Imports and returns the NVIDIA ChatNVIDIA model.
        Changing models in the future only requires editing this single function.
        """
        if not cls.NVIDIA_API_KEY:
            raise ValueError(
                "CRITICAL ERROR: NVIDIA_API_KEY is not set. Please check your .env file "
                "or your Streamlit Cloud Secrets settings."
            )
            
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        
        return ChatNVIDIA(
            model=cls.LLM_MODEL_NAME,
            api_key=cls.NVIDIA_API_KEY,
            temperature=temperature
        )

    @classmethod
    def validate_paths(cls):
        """Ensures directories exist or warns if resources are missing."""
        if not cls.DATASET_PATH.exists():
            print(f"Warning: Dataset not found at {cls.DATASET_PATH}. "
                  f"Please place 'campushire_jd_dataset.csv' inside the 'dataset' directory.")
        else:
            print("Dataset verified successfully.")