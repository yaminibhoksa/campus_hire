from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.parser import ParsedResume
from src.analyzer import SkillGapAnalysis

class RoadmapTask(BaseModel):
    task_title: str = Field(description="The concept or framework to master")
    estimated_hours: int = Field(description="Study and practice hours needed")
    resources: List[str] = Field(description="Recommended online resources or search terms")

class RoadmapWeek(BaseModel):
    week_number: int = Field(description="The sequential week number")
    theme: str = Field(description="The focus theme of this week")
    tasks: List[RoadmapTask] = Field(description="Syllabus topics to study")
    hands_on_project: str = Field(description="A mini-project to apply this week's learnings")
    milestone: str = Field(description="Weekly verification milestone")

class PersonalizedRoadmap(BaseModel):
    role: str = Field(description="The target role analyzed")
    duration_weeks: int = Field(description="Total weeks in this roadmap")
    weeks: List[RoadmapWeek] = Field(description="Week-by-week learning syllabus")
    general_tips: List[str] = Field(default=[], description="Strategic prep tips")

def generate_personalized_roadmap(resume: ParsedResume, gap_analysis: SkillGapAnalysis) -> PersonalizedRoadmap:
    # Call Centralized LLM Factory with moderate temperature
    llm = Config.get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(PersonalizedRoadmap)

    system_prompt = (
        "You are an expert curriculum designer. Formulate a realistic "
        "week-by-week technical roadmap helping the candidate bridge "
        "their gaps. Prioritize High-priority gaps first. Provide search "
        "paths/keywords for free resources instead of broken static URLs."
    )

    user_prompt = (
        "Target Role: {target_role}\n\n"
        "Current Candidate Strengths: {strengths}\n\n"
        "Identified Skill Gaps:\n{gaps_json}\n\n"
        "Generate a structured weekly roadmap."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    chain = prompt | structured_llm
    return chain.invoke({
        "target_role": gap_analysis.target_role,
        "strengths": ", ".join(gap_analysis.strengths),
        "gaps_json": gap_analysis.model_dump_json(indent=2)
    })