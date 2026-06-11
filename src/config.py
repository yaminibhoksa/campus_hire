import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file (primarily for local development)
load_dotenv()

class Config:
    # Key validation
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError(
            "CRITICAL ERROR: GOOGLE_API_KEY is not set. Please check your .env file "
            "or host-provided environment variables."
        )

    # Base Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_PATH = BASE_DIR / "dataset" / "campushire_jd_dataset.csv"
    
    # FAISS configuration
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    FAISS_INDEX_PATH = BASE_DIR / FAISS_INDEX_DIR

   
    # Model parameters
    LLM_MODEL_NAME = "gemini-2.5-flash"  # <-- Update to the active 2.5 Flash model
    EMBEDDING_MODEL_NAME = "gemini-embedding-2-preview"

    @classmethod
    def validate_paths(cls):
        """Ensures directories exist or warns if resources are missing."""
        if not cls.DATASET_PATH.exists():
            print(f"Warning: Dataset not found at {cls.DATASET_PATH}. "
                  f"Please place 'campushire_jd_dataset.csv' inside the 'dataset' directory.")
        else:
            print("Dataset verified successfully.")