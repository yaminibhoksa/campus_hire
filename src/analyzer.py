from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.parser import ParsedResume

class SkillGapItem(BaseModel):
    skill_name: str = Field(description="The name of the missing skill or technology")
    priority: str = Field(description="Priority: High, Medium, or Low")
    importance_reason: str = Field(description="How this skill is applied in the target role")
    difficulty_to_learn: str = Field(description="Estimated difficulty: Easy, Medium, or Hard")

class SkillGapAnalysis(BaseModel):
    target_role: str = Field(description="The job title evaluated")
    match_percentage: int = Field(description="Calculated skill alignment (0-100%)")
    strengths: List[str] = Field(description="Key relevant skills the candidate possesses")
    missing_skills: List[SkillGapItem] = Field(description="List of skills the candidate lacks")
    key_recommendation: str = Field(description="A structured professional upskilling recommendation")

def analyze_skill_gaps(resume: ParsedResume, target_jd_text: str, target_jd_title: str) -> SkillGapAnalysis:
    # Call Centralized LLM Factory
    llm = Config.get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(SkillGapAnalysis)

    system_prompt = (
        "You are an expert technical placement mentor. Perform a meticulous "
        "gap analysis comparing the resume against the target JD. Organize "
        "missing skills by priority needed to clear placements."
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
    return chain.invoke({
        "jd_title": target_jd_title,
        "jd_text": target_jd_text,
        "resume_details": resume.model_dump_json(indent=2)
    })