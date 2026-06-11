from typing import List
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from src.config import Config
from src.parser import ParsedResume

# ==========================================
# 1. DEFINE MATCH EVALUATION SCHEMAS
# ==========================================

class JobMatchEvaluation(BaseModel):
    jd_id: int = Field(description="The unique metadata ID of the matching job description")
    title: str = Field(description="Job Title from the matching job description")
    match_score: int = Field(description="A numeric score from 0 to 100 based on how well the candidate's profile matches the JD requirements")
    matched_skills: List[str] = Field(description="List of required job skills that the candidate already has in their resume")
    missing_skills: List[str] = Field(description="List of critical required skills or technologies from the JD that are missing from the candidate's resume")
    reasoning: str = Field(description="A concise, professional explanation summarizing why this match score was assigned, referring to experience and projects")

class MatchResults(BaseModel):
    matches: List[JobMatchEvaluation] = Field(description="List of matching job evaluations sorted by highest match score first")


# ==========================================
# 2. MATCH ENGINE IMPLEMENTATION
# ==========================================

def retrieve_matching_jds(query_text: str, k: int = 3) -> list:
    """
    Loads the local FAISS vector store and retrieves the top k similar Job Documents,
    programmatically filtering out duplicate job titles.
    """
    if not Config.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index folder not found at {Config.FAISS_INDEX_PATH}. "
            f"Please run the ingestion pipeline ('python -m src.ingest') first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDING_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY
    )

    # Load FAISS locally. allow_dangerous_deserialization=True is safe here
    # since we compiled and serialized the database ourselves locally.
    db = FAISS.load_local(
        folder_path=str(Config.FAISS_INDEX_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    # Ask FAISS for a larger batch of records (e.g. 10) so we have a pool to deduplicate
    matched_docs = db.similarity_search(query_text, k=10)
    
    # Filter out duplicate job titles programmatically
    unique_docs = []
    seen_titles = set()
    
    for doc in matched_docs:
        title = doc.metadata.get("title", "").strip().lower()
        if title not in seen_titles:
            seen_titles.add(title)
            unique_docs.append(doc)
            
        # Stop collecting once we have the desired 'k' unique matching jobs
        if len(unique_docs) == k:
            break
            
    return unique_docs


def evaluate_matches(resume: ParsedResume, matched_docs: list) -> MatchResults:
    """
    Feeds matched job documents and the parsed resume to Gemini to calculate
    scores and perform granular matching metrics.
    """
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.0
    )
    
    # Configure structured output to return Pydantic MatchResults
    structured_llm = llm.with_structured_output(MatchResults)

    # Prepare inputs for the LLM
    resume_dump = resume.model_dump_json(indent=2)
    jds_context = ""
    for doc in matched_docs:
        meta_id = doc.metadata.get("id", -1)
        jds_context += f"--- JOB DESCRIPTION ID: {meta_id} ---\n{doc.page_content}\n\n"

    system_prompt = (
        "You are an expert technical recruiter. Your task is to compare a candidate's "
        "parsed resume profile against a list of retrieved Job Descriptions (JDs). "
        "For each job description provided, calculate a match score from 0 to 100 based on "
        "skill alignment, experience levels, academic background, and relevant projects. "
        "Extract matched skills, identify key missing skills, and provide a constructive, "
        "evidence-backed reasoning summary for each evaluation. Sort matches by score in descending order."
    )

    user_prompt = (
        "Here is the candidate's resume profile:\n{resume_profile}\n\n"
        "Here are the matched Job Descriptions:\n{jds}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    chain = prompt | structured_llm

    # Invoke the model
    results = chain.invoke({
        "resume_profile": resume_dump,
        "jds": jds_context
    })

    return results

if __name__ == "__main__":
    from src.parser import ContactInfo, Education, Experience, Project
    
    # Define a mock resume to simulate a parsed candidate
    mock_resume = ParsedResume(
        name="Yamini Bhoksa",
        contact=ContactInfo(email="yamini@example.com", phone="+91 9876543210"),
        skills=["Python", "SQL", "LangChain", "Streamlit", "FAISS", "Git", "Machine Learning", "Pandas"],
        education=[Education(degree="B.Tech in Computer Science", institution="JNTU", year="2021 - 2025", gpa="9.1")],
        experience=[
            Experience(
                role="Data Science Intern",
                company="TechCorp",
                duration="June 2024 - Present",
                description=["Developed custom dashboards using Streamlit.", "Integrated Pandas data manipulation routines."]
            )
        ],
        projects=[
            Project(
                title="CampusHire AI Platform",
                technologies=["Gemini LLM", "LangChain", "FAISS"],
                description="Designed an automatic match recommendation engine using vector stores."
            )
        ],
        target_role="Data Scientist"
    )

    print("Step 1: Constructing query from parsed resume details...")
    query = f"Target Role: {mock_resume.target_role}. Skills: {', '.join(mock_resume.skills)}"
    print(f"Generated query text: {query}\n")

    print("Step 2: Retrieving top 3 matching job descriptions from FAISS database (with deduplication)...")
    try:
        matched_job_docs = retrieve_matching_jds(query, k=3)
        print(f"Retrieved {len(matched_job_docs)} matching jobs.")
        for idx, doc in enumerate(matched_job_docs):
            print(f"  Match {idx+1}: {doc.metadata.get('title')} (ID: {doc.metadata.get('id')})")
        
        print("\nStep 3: Evaluating matches using Gemini model evaluation pipeline...")
        evaluation = evaluate_matches(mock_resume, matched_job_docs)
        print("\nEvaluation results compiled successfully:")
        print(evaluation.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error during matching execution: {e}")