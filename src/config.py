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

    # Model parameters (Switched back to Google's highly stable 2.0 Flash)
    # Model parameters (Switched to high-RPM 2.0 Flash Lite)
    LLM_MODEL_NAME = "gemini-2.0-flash-lite"  
    EMBEDDING_MODEL_NAME = "gemini-embedding-2-preview"

    # 2. Key validation and Fallbacks (System Env ➔ Streamlit Secrets)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        try:
            import streamlit as st
            GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass

    @classmethod
    def get_llm(cls, temperature: float = 0.0):
        """
        Centralized LLM Factory. Imports and returns the Google ChatGoogleGenerativeAI model.
        Decoupled from NVIDIA's temporary network outages.
        """
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "CRITICAL ERROR: GOOGLE_API_KEY is not set. Please check your .env file "
                "or your Streamlit Cloud Secrets settings."
            )
            
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        return ChatGoogleGenerativeAI(
            model=cls.LLM_MODEL_NAME,
            google_api_key=cls.GOOGLE_API_KEY,
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