from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config
from src.parser import ParsedResume
from src.analyzer import SkillGapAnalysis

# ==========================================
# 1. DEFINE ROADMAP SCHEMAS
# ==========================================

class RoadmapTask(BaseModel):
    task_title: str = Field(description="The name of the concept, technology, or theoretical topic to master")
    estimated_hours: int = Field(description="Approximate study and practice hours required for this task")
    resources: List[str] = Field(description="Recommended online resources, tutorial search paths, or documentation sites")

class RoadmapWeek(BaseModel):
    week_number: int = Field(description="The sequential week number (e.g., 1, 2, 3...)")
    theme: str = Field(description="The primary focus theme of this week (e.g., Mastering FastAPI and REST APIs)")
    tasks: List[RoadmapTask] = Field(description="Granular topics and subtasks to study during this week")
    hands_on_project: str = Field(description="A highly actionable hands-on mini-project to build to apply this week's learnings")
    milestone: str = Field(description="Expected learning milestone or verification check for the end of the week")

class PersonalizedRoadmap(BaseModel):
    role: str = Field(description="The target role analyzed")
    duration_weeks: int = Field(description="Total weeks in this customized roadmap")
    weeks: List[RoadmapWeek] = Field(description="Chronological, week-by-week learning syllabus")
    general_tips: List[str] = Field(default=[], description="Broad strategic interview preparation tips for this specific role")


# ==========================================
# 2. ROADMAP GENERATOR IMPLEMENTATION
# ==========================================

def generate_personalized_roadmap(resume: ParsedResume, gap_analysis: SkillGapAnalysis) -> PersonalizedRoadmap:
    """
    Inputs parsed profile details and identified gaps to synthesize
    a structured week-by-week roadmap.
    """
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.2  # Slightly higher temperature for creative planning logic
    )

    # Instruct Gemini to output structured results mapping to PersonalizedRoadmap
    structured_llm = llm.with_structured_output(PersonalizedRoadmap)

    system_prompt = (
        "You are an expert technical curriculum designer and career mentor. Your job is "
        "to formulate a highly customized, realistic week-by-week technical roadmap. "
        "The timeline must be tailored to help a candidate bridge their specific skill gaps "
        "and prepare for technical placement interviews for their target role.\n\n"
        "Requirements:\n"
        "1. Prioritize learning of 'High' priority missing skills early in the schedule.\n"
        "2. Provide highly practical, build-to-learn mini-projects for each week.\n"
        "3. Provide search paths or documentation paths for tutorials, rather than "
        "broken or static URL links."
    )

    user_prompt = (
        "Target Role: {target_role}\n\n"
        "Current Candidate Strengths: {strengths}\n\n"
        "Identified Skill Gaps:\n{gaps_json}\n\n"
        "Generate a structured, logical weekly roadmap."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    chain = prompt | structured_llm

    # Execute and parse
    roadmap = chain.invoke({
        "target_role": gap_analysis.target_role,
        "strengths": ", ".join(gap_analysis.strengths),
        "gaps_json": gap_analysis.model_dump_json(indent=2)
    })

    return roadmap

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
        projects=[],
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
            ),
            SkillGapItem(
                skill_name="Docker",
                priority="Medium",
                importance_reason="Essential for lightweight microservices container packaging.",
                difficulty_to_learn="Medium"
            )
        ],
        key_recommendation="Prioritize ML fundamentals via Scikit-Learn."
    )

    print("Step 1: Commencing personalized roadmap generation using mock profile data...")
    try:
        custom_roadmap = generate_personalized_roadmap(mock_resume, mock_gap_analysis)
        print("\nRoadmap compiled successfully:")
        print(custom_roadmap.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error during roadmap creation: {e}")