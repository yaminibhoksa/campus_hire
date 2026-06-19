from typing import List, Optional
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config

# ==========================================
# 1. DEFINE PYDANTIC SCHEMAS FOR VALIDATION
# ==========================================

class ContactInfo(BaseModel):
    email: Optional[str] = Field(default="", description="Email address of the candidate")
    phone: Optional[str] = Field(default="", description="Phone number of the candidate")
    linkedin: Optional[str] = Field(default="", description="LinkedIn profile URL")
    github: Optional[str] = Field(default="", description="GitHub profile URL")

class Education(BaseModel):
    degree: str = Field(description="Degree or specialization name (e.g., B.Tech in Computer Science)")
    institution: str = Field(description="Name of the school, college, or university")
    year: Optional[str] = Field(default="", description="Year of completion or duration range")
    gpa: Optional[str] = Field(default="", description="GPA or percentage achieved")

class Experience(BaseModel):
    role: str = Field(description="Job title or role")
    company: str = Field(description="Name of the company or organization")
    duration: Optional[str] = Field(default="", description="Duration of employment")
    description: List[str] = Field(default=[], description="Bullet points detailing achievements")

class Project(BaseModel):
    title: str = Field(description="Name of the project")
    technologies: List[str] = Field(default=[], description="List of technologies used")
    description: str = Field(description="A concise summary of what was built")

class ParsedResume(BaseModel):
    name: str = Field(description="Full name of the candidate")
    contact: ContactInfo = Field(description="Contact details")
    skills: List[str] = Field(description="Complete list of technologies, tools, and languages")
    education: List[Education] = Field(default=[], description="Academic history")
    experience: List[Experience] = Field(default=[], description="Work history")
    projects: List[Project] = Field(default=[], description="Personal or academic projects")
    target_role: Optional[str] = Field(default="", description="Deduced target role of the candidate")


# ==========================================
# 2. CORE PARSER FUNCTIONS
# ==========================================

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        for page in doc:
            text_content.append(page.get_text())
        doc.close()
        return "\n".join(text_content).strip()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document. Reason: {str(e)}")

def parse_resume_to_json(raw_text: str) -> ParsedResume:
    # Call Centralized LLM Factory
    llm = Config.get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ParsedResume)
    
    system_prompt = (
        "You are an expert ATS (Applicant Tracking System) parser. "
        "Analyze the provided resume text and extract structured fields. "
        "Strictly adhere to the formatting requested. If an element "
        "is missing, populate it with an empty string or empty list."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Analyze and parse the following resume:\n\n{resume_text}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({"resume_text": raw_text})