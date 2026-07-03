import json
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import Config
from src.parser import parse_resume_to_json, ParsedResume
from src.matcher import retrieve_matching_jds, evaluate_matches
from src.analyzer import analyze_skill_gaps
from src.roadmap import generate_personalized_roadmap
from src.interview import generate_mock_interview

# ==========================================
# 1. WRAP OUR PIPELINES AS LANGCHAIN TOOLS
# ==========================================

@tool
def resume_analyzer_tool(resume_text: str) -> str:
    """Parses raw resume text and extracts structured candidate fields."""
    try:
        parsed = parse_resume_to_json(resume_text)
        return parsed.model_dump_json(indent=2)
    except Exception as e:
        return f"Error analyzing resume: {str(e)}"

@tool
def jd_matcher_tool(skills: list[str], target_role: str) -> str:
    """Searches FAISS and returns matching job evaluations."""
    try:
        query_text = f"Target Role: {target_role}. Skills: {', '.join(skills)}"
        matched_docs = retrieve_matching_jds(query_text, k=3)
        
        from src.parser import ContactInfo
        temp_resume = ParsedResume(
            name="Candidate",
            contact=ContactInfo(),
            skills=skills,
            target_role=target_role
        )
        evaluation = evaluate_matches(temp_resume, matched_docs)
        return evaluation.model_dump_json(indent=2)
    except Exception as e:
        return f"Error matching jobs: {str(e)}"

@tool
def gap_analyzer_tool(resume_json_str: str, target_jd_text: str, target_jd_title: str) -> str:
    """Performs comparative skill gap analysis between parsed resume and target job description."""
    try:
        resume_dict = json.loads(resume_json_str)
        resume_obj = ParsedResume(**resume_dict)
        report = analyze_skill_gaps(resume_obj, target_jd_text, target_jd_title)
        return report.model_dump_json(indent=2)
    except Exception as e:
        return f"Error analyzing gaps: {str(e)}"

@tool
def roadmap_generator_tool(gap_analysis_json_str: str) -> str:
    """Generates a week-by-week learning roadmap based on a gap analysis report."""
    try:
        from src.analyzer import SkillGapAnalysis
        gap_dict = json.loads(gap_analysis_json_str)
        gap_obj = SkillGapAnalysis(**gap_dict)
        
        from src.parser import ContactInfo
        temp_resume = ParsedResume(
            name="Candidate",
            contact=ContactInfo(),
            skills=gap_obj.strengths
        )
        roadmap = generate_personalized_roadmap(temp_resume, gap_obj)
        return roadmap.model_dump_json(indent=2)
    except Exception as e:
        return f"Error generating roadmap: {str(e)}"

@tool
def mock_interview_tool(resume_json_str: str, gap_analysis_json_str: str) -> str:
    """Generates tailored mock interview questions based on candidate profile and gap report."""
    try:
        from src.analyzer import SkillGapAnalysis
        resume_dict = json.loads(resume_json_str)
        resume_obj = ParsedResume(**resume_dict)
        gap_dict = json.loads(gap_analysis_json_str)
        gap_obj = SkillGapAnalysis(**gap_dict)
        
        interview = generate_mock_interview(resume_obj, gap_obj, num_questions=4)
        return interview.model_dump_json(indent=2)
    except Exception as e:
        return f"Error generating mock interview: {str(e)}"


# ==========================================
# 2. AGENT INITIALIZATION
# ==========================================

def get_placement_mentor_agent():
    # Call Centralized LLM Factory
    llm = Config.get_llm(temperature=0.3)

    tools = [
        resume_analyzer_tool,
        jd_matcher_tool,
        gap_analyzer_tool,
        roadmap_generator_tool,
        mock_interview_tool
    ]

    system_prompt = (
        "You are 'CampusHire AI Mentor', a supportive and brilliant career coach. "
        "Your job is to guide students step-by-step through their placement preparation.\n\n"
        "IMPORTANT RULES FOR TOOL CALLING:\n"
        "1. For simple greetings (such as 'hi', 'hello', 'hey', 'good morning'), general conversation, or small talk, "
        "DO NOT CALL ANY TOOLS. Simply reply to the user with a short, friendly greeting and ask how you can help them.\n"
        "2. ONLY call a tool when the user explicitly requests an operation that requires that specific tool's output.\n\n"
        "IMPORTANT LENGTH LIMITATION: Your responses must be extremely concise, brief, and "
        "straight to the point (no more than 1 or 2 short paragraphs). Avoid outputting extremely long lists. "
        "Keep formatting clean and compact.\n\n"
        "You have access to a suite of highly specialized tools:\n"
        "- Use `resume_analyzer_tool` to parse raw resume text.\n"
        "- Use `jd_matcher_tool` to search for jobs and match scores.\n"
        "- Use `gap_analyzer_tool` to figure out what skills are missing.\n"
        "- Use `roadmap_generator_tool` to create weekly learning timetables.\n"
        "- Use `mock_interview_tool` to generate custom interview practice questions."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )