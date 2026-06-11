from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config
from src.parser import ParsedResume

# ==========================================
# 1. DEFINE SKILL GAP SCHEMAS
# ==========================================

class SkillGapItem(BaseModel):
    skill_name: str = Field(description="The name of the missing skill, tool, framework, or concept")
    priority: str = Field(description="How critical it is to learn this skill for the role: High (blocking placement), Medium (nice-to-have), Low (bonus)")
    importance_reason: str = Field(description="A brief explanation of how this skill is applied in the target role")
    difficulty_to_learn: str = Field(description="Estimated learning difficulty for a beginner: Easy, Medium, or Hard")

class SkillGapAnalysis(BaseModel):
    target_role: str = Field(description="The job title or role evaluated")
    match_percentage: int = Field(description="Calculated overall skill alignment percentage (0-100%)")
    strengths: List[str] = Field(description="Key relevant skills the candidate already possesses that fit this role")
    missing_skills: List[SkillGapItem] = Field(description="Categorized and prioritized list of skills the candidate lacks")
    key_recommendation: str = Field(description="A structured, supportive professional summary indicating where the candidate should start learning first")


# ==========================================
# 2. ANALYZER ENGINE IMPLEMENTATION
# ==========================================

def analyze_skill_gaps(resume: ParsedResume, target_jd_text: str, target_jd_title: str) -> SkillGapAnalysis:
    """
    Compares the candidate's parsed profile against a selected Job Description
    to isolate skill gaps and construct a structured analysis report.
    """
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.0
    )
    
    # Instruct Gemini to output structured results using our Pydantic schema
    structured_llm = llm.with_structured_output(SkillGapAnalysis)

    system_prompt = (
        "You are an expert technical interviewer and placement mentor. Your task is "
        "to perform a meticulous gap analysis by comparing a candidate's resume "
        "against their target Job Description (JD). "
        "1. Identify overlaps (strengths) and highlight critical gaps (missing skills).\n"
        "2. Categorize each missing skill into high, medium, or low priority based "
        "on core requirements needed to clear a placement technical interview.\n"
        "3. Keep descriptions factual, encouraging, and clear."
    )

    user_prompt = (
        "Target Job Title: {jd_title}\n\n"
        "Target Job Description:\n{jd_text}\n\n"
        "Candidate Resume Details:\n{resume_details}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    chain = prompt | structured_llm

    # Run the chain
    analysis_report = chain.invoke({
        "jd_title": target_jd_title,
        "jd_text": target_jd_text,
        "resume_details": resume.model_dump_json(indent=2)
    })

    return analysis_report

if __name__ == "__main__":
    from src.parser import ContactInfo, Education, Experience, Project
    
    # Simulate a parsed candidate profile
    mock_resume = ParsedResume(
        name="Yamini Bhoksa",
        contact=ContactInfo(email="yamini@example.com", phone="+91 9876543210"),
        skills=["Python", "SQL", "LangChain", "Streamlit", "FAISS", "Git"],
        education=[Education(degree="B.Tech in Computer Science", institution="JNTU", year="2021 - 2025", gpa="9.1")],
        experience=[
            Experience(
                role="Data Science Intern",
                company="TechCorp",
                duration="June 2024 - Present",
                description=["Developed custom dashboards using Streamlit.", "Integrated Pandas data manipulation routines."]
            )
        ],
        projects=[],
        target_role="Data Scientist"
    )

    # Simulate a Job Description retrieved from your FAISS indexing
    mock_target_jd_title = "Associate ML Engineer / Data Scientist"
    mock_target_jd_text = (
        "Role Overview:\n"
        "We are looking for an Associate Data Scientist with strong Python programming fundamentals. "
        "Key requirements include experience with SQL databases and hands-on exposure to Machine Learning "
        "frameworks (such as Scikit-Learn or PyTorch). Experience with Docker containerization, AWS, "
        "and building REST APIs (FastAPI) is highly preferred for production deployments. Candidates must "
        "be able to design and evaluate ML models."
    )

    print("Step 1: Commencing analysis comparing mock candidate vs selected JD...")
    try:
        report = analyze_skill_gaps(mock_resume, mock_target_jd_text, mock_target_jd_title)
        print("\nSkill Gap Analysis Report completed successfully:")
        print(report.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error during gap analysis: {e}")