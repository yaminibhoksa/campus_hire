from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.parser import ParsedResume
from src.analyzer import SkillGapAnalysis

class InterviewQuestion(BaseModel):
    question_id: int = Field(description="Unique incremental ID for the question")
    category: str = Field(description="Technical (Strength), Technical (Gap), Project-Based, or Behavioral")
    question: str = Field(description="The mock interview question")
    difficulty: str = Field(description="Difficulty: Easy, Medium, or Hard")
    concept_tested: str = Field(description="The concept being evaluated")
    hint_or_ideal_response: str = Field(description="Ideal response guideline and key terms")

class MockInterviewSet(BaseModel):
    role: str = Field(description="The target job title")
    total_questions: int = Field(description="Total questions generated")
    questions: List[InterviewQuestion] = Field(description="List of mock interview questions")

def generate_mock_interview(resume: ParsedResume, gap_analysis: SkillGapAnalysis, num_questions: int = 6) -> MockInterviewSet:
    # Call Centralized LLM Factory
    llm = Config.get_llm(temperature=0.4)
    structured_llm = llm.with_structured_output(MockInterviewSet)

    system_prompt = (
        "You are an elite corporate technical interviewer. Draft a customized "
        "mock question set balancing project-based questions, core technical "
        "strengths, conceptual skill gap topics, and behavioral parameters."
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
    return chain.invoke({
        "target_role": gap_analysis.target_role,
        "resume_details": resume.model_dump_json(indent=2),
        "gaps_details": gap_analysis.model_dump_json(indent=2),
        "num_qs": num_questions
    })

def evaluate_candidate_mock_response(question: str, ideal_rubric: str, user_response: str) -> str:
    # Call Centralized LLM Factory with low temperature for grading
    llm = Config.get_llm(temperature=0.2)
    
    system_prompt = (
        "You are an elite interviewer. Evaluate the user's typed response "
        "against the ideal rubric. Provide: 1. Score out of 10, 2. Strengths, "
        "3. Gaps/Weaknesses, 4. Perfect Sample Answer."
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
    return chain.invoke({
        "question": question,
        "rubric": ideal_rubric,
        "response": user_response
    }).content