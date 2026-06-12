import json
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

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
    """
    Parses a raw resume text string and extracts structured fields 
    such as name, skills, education, experience, and projects.
    """
    try:
        parsed = parse_resume_to_json(resume_text)
        return parsed.model_dump_json(indent=2)
    except Exception as e:
        return f"Error analyzing resume: {str(e)}"

@tool
def jd_matcher_tool(skills: list[str], target_role: str) -> str:
    """
    Takes a list of candidate skills and a target role, searches the FAISS database 
    for similar jobs, and returns structured match scores and evaluations.
    """
    try:
        query_text = f"Target Role: {target_role}. Skills: {', '.join(skills)}"
        matched_docs = retrieve_matching_jds(query_text, k=3)
        
        # Build a temporary ParsedResume object to fit the matcher's parameters
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
    """
    Performs a comparative skill gap analysis between a parsed resume (JSON string) 
    and a specific target job description text.
    """
    try:
        resume_dict = json.loads(resume_json_str)
        resume_obj = ParsedResume(**resume_dict)
        report = analyze_skill_gaps(resume_obj, target_jd_text, target_jd_title)
        return report.model_dump_json(indent=2)
    except Exception as e:
        return f"Error analyzing gaps: {str(e)}"

@tool
def roadmap_generator_tool(gap_analysis_json_str: str) -> str:
    """
    Generates a personalized, week-by-week learning roadmap based on 
    an identified gap analysis report (JSON string).
    """
    try:
        from src.analyzer import SkillGapAnalysis
        gap_dict = json.loads(gap_analysis_json_str)
        gap_obj = SkillGapAnalysis(**gap_dict)
        
        # Build a temporary ParsedResume for compatibility
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
    """
    Generates tailored mock interview questions with answers based on the parsed resume (JSON string) 
    and identified skill gap analysis (JSON string).
    """
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
    """
    Initializes and returns the LangChain AgentExecutor for the Placement Mentor.
    """
    # 1. Define LLM
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL_NAME,
        google_api_key=Config.GOOGLE_API_KEY,
        temperature=0.3
    )

    # 2. Gather Tools
    tools = [
        resume_analyzer_tool,
        jd_matcher_tool,
        gap_analyzer_tool,
        roadmap_generator_tool,
        mock_interview_tool
    ]

    # 3. Create the prompt template
    system_prompt = (
        "You are 'CampusHire AI Mentor', a supportive and brilliant conversational career coach "
        "and placement coordinator. Your job is to guide students step-by-step through their placement "
        "preparation journey.\n\n"
        "IMPORTANT LENGTH LIMITATION: Your responses must be extremely concise, brief, and "
        "straight to the point (no more than 1 or 2 short paragraphs). Avoid outputting extremely long lists "
        "or repeating yourself. Since you are displayed in a narrow sidebar chatbot panel, "
        "keep formatting clean, compact, and highly scannable.\n\n"
        "You have access to a suite of highly specialized tools:\n"
        "- Use `resume_analyzer_tool` to parse raw resume text.\n"
        "- Use `jd_matcher_tool` to search for jobs and match scores.\n"
        "- Use `gap_analyzer_tool` to figure out what skills are missing.\n"
        "- Use `roadmap_generator_tool` to create weekly learning timetables.\n"
        "- Use `mock_interview_tool` to generate custom interview practice questions.\n\n"
        "Always communicate in a professional, encouraging, and structured mentor tone. "
        "If a student asks a broad question, call the appropriate tool to get data and present "
        "the findings cleanly."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. Construct the Tool-Calling Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. Build the Agent Executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    return agent_executor

if __name__ == "__main__":
    print("Testing Central Mentor Agent routing capabilities...")
    agent_exec = get_placement_mentor_agent()
    
    # Test conversational query that requires search tool matching
    test_input = "My current skills are Python, SQL, and Git. I want to match with jobs for a Data Scientist."
    
    try:
        response = agent_exec.invoke({
            "input": test_input,
            "chat_history": []
        })
        print("\nAgent Conversation Output:")
        print(response["output"])
    except Exception as e:
        print(f"Agent test failed: {e}")