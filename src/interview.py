from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config
from src.parser import ParsedResume
from src.analyzer import SkillGapAnalysis

# ==========================================
# 1. DEFINE INTERVIEW SCHEMAS
# ==========================================

class InterviewQuestion(BaseModel):
    question_id: int = Field(description="Unique incremental ID for the question (e.g. 1, 2, 3...)")
    category: str = Field(description="Category of the question: Technical (Strength), Technical (Gap), Project-Based, or Behavioral")
    question: str = Field(description="The actual interview question, styled to resemble real placement panel evaluations")
    difficulty: str = Field(description="Difficulty level of the question: Easy, Medium, or Hard")
    concept_tested: str = Field(description="The specific core concept, framework, or soft skill being evaluated")
    hint_or_ideal_response: str = Field(description="A professional guideline, key talking points, or structural schema of a stellar response")

class MockInterviewSet(BaseModel):
    role: str = Field(description="The target job title for this mock interview session")
    total_questions: int = Field(description="Total number of interview questions generated")
    questions: List[InterviewQuestion] = Field(description="List of mock interview questions and answer guidelines")


# ==========================================
# 2. INTERVIEW GENERATOR IMPLEMENTATION
# ==========================================

def generate_mock_interview(resume: ParsedResume, gap_analysis: SkillGapAnalysis, num_questions: int = 6) -> MockInterviewSet:
    """
    Inputs parsed candidate details and identified gaps to synthesize
    a set of personalized mock interview questions with guide hints.
    """
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.4  # Moderate temperature for realistic variety in questions
    )

    # Instruct Gemini to output structured results mapping to MockInterviewSet
    structured_llm = llm.with_structured_output(MockInterviewSet)

    system_prompt = (
        "You are an elite corporate technical interviewer and HR panel lead. Your job is "
        "to draft a customized set of mock interview questions designed to prepare a candidate "
        "for a technical placement round.\n\n"
        "Guidelines:\n"
        "1. Question Distribution: Create a balanced set of questions:\n"
        "   - Project-Based: Ask about the candidate's specific listed projects.\n"
        "   - Technical (Strength): Query technical tools they already have in their resume.\n"
        "   - Technical (Gap): Query fundamental concepts from their missing/gap skills list.\n"
        "   - Behavioral: Ask situational questions mapping to corporate expectations.\n"
        "2. Keep the hints detailed and practical, mentioning core keywords the candidate should use."
    )

    user_prompt = (
        "Target Role: {target_role}\n"
        "Candidate Resume Details:\n{resume_details}\n\n"
        "Identified Skill Gaps:\n{gaps_details}\n\n"
        "Desired Number of Questions: {num_qs}\n\n"
        "Generate a structured interview question set."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    chain = prompt | structured_llm

    # Execute and parse
    interview_set = chain.invoke({
        "target_role": gap_analysis.target_role,
        "resume_details": resume.model_dump_json(indent=2),
        "gaps_details": gap_analysis.model_dump_json(indent=2),
        "num_qs": num_questions
    })

    return interview_set


def evaluate_candidate_mock_response(question: str, ideal_rubric: str, user_response: str) -> str:
    """
    Grades the candidate's typed response against the ideal rubric.
    Returns a brief scorecard out of 10, strengths, and suggestions for improvement.
    """
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.2  # Low temperature for analytical grading
    )
    
    system_prompt = (
        "You are an elite corporate technical interviewer. Your task is to evaluate "
        "a candidate's typed response against a question's ideal response rubric.\n\n"
        "Provide a structured assessment containing:\n"
        "1. Score: Rate the response out of 10.\n"
        "2. Strengths: What did the candidate address correctly?\n"
        "3. Weaknesses/Gaps: What important keywords, frameworks, or concepts did they miss?\n"
        "4. Perfect Answer Sample: A brief, single-paragraph version of how they could state it perfectly."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            "Interview Question: {question}\n"
            "Ideal Rubric/Hint: {rubric}\n"
            "Candidate's Typed Response: {response}\n\n"
            "Evaluate this response."
        ))
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "question": question,
        "rubric": ideal_rubric,
        "response": user_response
    })
    
    return result.content


if __name__ == "__main__":
    from src.parser import ContactInfo, Education, Experience, Project
    from src.analyzer import SkillGapItem
    
    # Simulate parsed candidate profile
    mock_resume = ParsedResume(
        name="Yamini Bhoksa",
        contact=ContactInfo(email="yamini@example.com", phone="+91 9876543210"),
        skills=["Python", "SQL", "LangChain", "Streamlit", "FAISS", "Git"],
        education=[Education(degree="B.Tech in Computer Science", institution="JNTU", year="2021 - 2025", gpa="9.1")],
        experience=[],
        projects=[
            Project(
                title="CampusHire AI Platform",
                technologies=["Gemini LLM", "LangChain", "FAISS"],
                description="Designed an automatic match recommendation engine using vector stores."
            )
        ],
        target_role="Data Scientist"
    )

    # Simulate Gap Analysis results from Phase 5
    mock_gap_analysis = SkillGapAnalysis(
        target_role="Associate ML Engineer / Data Scientist",
        match_percentage=50,
        strengths=["Python", "SQL", "Streamlit"],
        missing_skills=[
            SkillGapItem(
                skill_name="Scikit-Learn",
                priority="High",
                importance_reason="Core library for training supervised ML models.",
                difficulty_to_learn="Medium"
            ),
            SkillGapItem(
                skill_name="FastAPI",
                priority="Medium",
                importance_reason="Standard framework for writing REST API model wrappers.",
                difficulty_to_learn="Easy"
            )
        ],
        key_recommendation="Prioritize ML fundamentals via Scikit-Learn."
    )

    print("Step 1: Ingesting candidate information and generating customized mock interview questions...")
    try:
        mock_interview = generate_mock_interview(mock_resume, mock_gap_analysis, num_questions=5)
        print("\nMock Interview Questions compiled successfully:")
        print(mock_interview.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error during mock interview creation: {e}")