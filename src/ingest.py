import os
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config

def load_csv_and_build_documents() -> list[Document]:
    """
    Reads the CSV dataset and transforms each row into a LangChain Document.
    """
    if not Config.DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset missing at {Config.DATASET_PATH}")
    
    df = pd.read_csv(Config.DATASET_PATH)
    df = df.fillna("")
    
    documents = []
    for idx, row in df.iterrows():
        page_content = (
            f"Job Title: {row['Title']}\n"
            f"Experience Level: {row['ExperienceLevel']}\n"
            f"Years Of Experience Required: {row['YearsOfExperience']}\n"
            f"Required Skills: {row['Skills']}\n"
            f"Key Responsibilities: {row['Responsibilities']}\n"
            f"Keywords: {row['Keywords']}"
        )
        
        metadata = {
            "id": idx,
            "title": row["Title"],
            "experience_level": row["ExperienceLevel"],
            "years_of_experience": row["YearsOfExperience"],
            "skills": row["Skills"],
            "keywords": row["Keywords"]
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
        
    return documents

def generate_vector_store():
    """
    Creates and saves a FAISS vector database locally with safe batching
    to prevent hitting Gemini API Free Tier rate limits (429 errors).
    """
    print("Step 1: Parsing CSV rows into structured documents...")
    docs = load_csv_and_build_documents()
    total_docs = len(docs)
    print(f"Parsed {total_docs} job descriptions.")

    print("\nStep 2: Initializing Google Generative AI Embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDING_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY
    )

    # We will process in batches of 30 documents
    BATCH_SIZE = 15
    SLEEP_TIME = 15  # seconds to pause between batches

    print(f"\nStep 3: Generating embeddings in batches of {BATCH_SIZE} with a {SLEEP_TIME}s delay...")
    
    # 1. Initialize FAISS with the first batch
    first_batch = docs[:BATCH_SIZE]
    print(f"Embedding batch 1/{((total_docs - 1) // BATCH_SIZE) + 1} (Docs 0 to {len(first_batch)})...")
    db = FAISS.from_documents(first_batch, embeddings)
    
    # 2. Add remaining batches sequentially
    for i in range(BATCH_SIZE, total_docs, BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = ((total_docs - 1) // BATCH_SIZE) + 1
        
        print(f"Waiting {SLEEP_TIME} seconds to stay within free-tier API limits...")
        time.sleep(SLEEP_TIME)
        
        print(f"Embedding batch {batch_num}/{total_batches} (Docs {i} to {i + len(batch_docs)})...")
        db.add_documents(batch_docs)

    # Create output directory path if it doesn't exist
    os.makedirs(Config.FAISS_INDEX_PATH, exist_ok=True)
    
    print(f"\nStep 4: Saving FAISS index to storage at '{Config.FAISS_INDEX_PATH}'...")
    db.save_local(str(Config.FAISS_INDEX_PATH))
    print("Database indexing complete and saved successfully!")

if __name__ == "__main__":
    try:
        generate_vector_store()
    except Exception as e:
        print(f"\nAn error occurred during ingestion: {e}")