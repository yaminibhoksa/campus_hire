from typing import List, Optional
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
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
    year: Optional[str] = Field(default="", description="Year of completion or duration range (e.g., 2020 - 2024)")
    gpa: Optional[str] = Field(default="", description="GPA or percentage achieved (e.g., 8.9 CGPA or 85%)")

class Experience(BaseModel):
    role: str = Field(description="Job title or role (e.g., Software Engineering Intern)")
    company: str = Field(description="Name of the company or organization")
    duration: Optional[str] = Field(default="", description="Duration of employment (e.g., June 2023 - Aug 2023)")
    description: List[str] = Field(default=[], description="Bullet points detailing achievements, responsibilities, and tasks")

class Project(BaseModel):
    title: str = Field(description="Name of the project")
    technologies: List[str] = Field(default=[], description="List of technologies, tools, and languages used in this project")
    description: str = Field(description="A concise summary of what was built and the impact of the project")

class ParsedResume(BaseModel):
    name: str = Field(description="Full name of the candidate")
    contact: ContactInfo = Field(description="Contact details of the candidate")
    skills: List[str] = Field(description="Complete list of technologies, databases, programming languages, and tools")
    education: List[Education] = Field(default=[], description="Academic history")
    experience: List[Experience] = Field(default=[], description="Work or internship history")
    projects: List[Project] = Field(default=[], description="Personal or academic projects")
    target_role: Optional[str] = Field(default="", description="Deduced target role of the candidate (e.g., Frontend Developer, Data Scientist, Java Developer)")


# ==========================================
# 2. CORE PARSER FUNCTIONS
# ==========================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all raw text characters page-by-page from a PDF file.
    """
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
    """
    Applies Gemini structured schema parsing to translate raw text into
    a validated Pydantic object.
    """
    # Initialize the LLM (temperature set to 0.0 for maximum precision)
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.0
    )
    
    # Configure the structured output model using Pydantic
    structured_llm = llm.with_structured_output(ParsedResume)
    
    # Build System Prompt and Template
    system_prompt = (
        "You are an expert ATS (Applicant Tracking System) parser and resume engineer. "
        "Your job is to thoroughly analyze the provided resume text and extract structured fields. "
        "Strictly adhere to the formatting requested. If an element (like contact details, GPA, or duration) "
        "is missing, populate it with an empty string or empty list, do not make things up."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Analyze and parse the following resume:\n\n{resume_text}")
    ])
    
    # Chain the prompt to the structured LLM
    chain = prompt | structured_llm
    
    # Run the model
    result = chain.invoke({"resume_text": raw_text})
    return result

if __name__ == "__main__":
    # Test text
    sample_text = """
    Yamini Bhoksa
    Email: yamini@example.com | Phone: +91 9876543210
    LinkedIn: linkedin.com/in/yamini-bhoksa | GitHub: github.com/yamini
    
    Education:
    B.Tech in Computer Science Engineering (GPA: 9.1)
    Jawaharlal Nehru Technological University (2021 - 2025)
    
    Skills:
    Python, SQL, HTML, CSS, JavaScript, LangChain, Streamlit, FAISS, Git, Machine Learning
    
    Experience:
    Data Science Intern at TechCorp (June 2024 - Present)
    - Developed custom dashboards using Streamlit to monitor dataset ingestion metrics.
    - Integrated Pandas and NumPy data manipulation routines to preprocess CSV inputs.
    
    Projects:
    CampusHire AI Platform
    - Used Gemini LLM and LangChain vector stores to design an automatic match recommendation engine.
    - Achieved highly accurate skill gap analysis matching by parsing user profiles.
    """
    
    print("Testing parser using a mock text sample...")
    try:
        parsed_result = parse_resume_to_json(sample_text)
        print("\nSuccessfully Parsed Profile Struct:")
        print(parsed_result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Extraction failed: {e}")