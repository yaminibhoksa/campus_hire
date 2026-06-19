from typing import List
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config
from src.parser import ParsedResume

class JobMatchEvaluation(BaseModel):
    jd_id: int = Field(description="The unique metadata ID of the matching job description")
    title: str = Field(description="Job Title from the matching job description")
    match_score: int = Field(description="A numeric score from 0 to 100 based on alignment")
    matched_skills: List[str] = Field(description="Required job skills the candidate has")
    missing_skills: List[str] = Field(description="Critical required skills the candidate lacks")
    reasoning: str = Field(description="A concise summary explaining why this match score was assigned")

class MatchResults(BaseModel):
    matches: List[JobMatchEvaluation] = Field(description="List of matching job evaluations")

def retrieve_matching_jds(query_text: str, k: int = 3) -> list:
    if not Config.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index folder not found at {Config.FAISS_INDEX_PATH}.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDING_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY
    )

    db = FAISS.load_local(
        folder_path=str(Config.FAISS_INDEX_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    matched_docs = db.similarity_search(query_text, k=10)
    
    unique_docs = []
    seen_titles = set()
    for doc in matched_docs:
        title = doc.metadata.get("title", "").strip().lower()
        if title not in seen_titles:
            seen_titles.add(title)
            unique_docs.append(doc)
        if len(unique_docs) == k:
            break
            
    return unique_docs

def evaluate_matches(resume: ParsedResume, matched_docs: list) -> MatchResults:
    # Call Centralized LLM Factory
    llm = Config.get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(MatchResults)

    resume_dump = resume.model_dump_json(indent=2)
    jds_context = ""
    for doc in matched_docs:
        meta_id = doc.metadata.get("id", -1)
        jds_context += f"--- JOB DESCRIPTION ID: {meta_id} ---\n{doc.page_content}\n\n"

    system_prompt = (
        "You are an expert technical recruiter. Compare the candidate's resume "
        "against the JDs. Calculate a match score from 0 to 100 based on skills, "
        "projects, and education. Sort matches by score in descending order."
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
    return chain.invoke({"resume_profile": resume_dump, "jds": jds_context})