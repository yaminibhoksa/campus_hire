import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import time

# Import our custom backend modules
from src.config import Config
from src.parser import parse_resume_to_json, ParsedResume
from src.matcher import retrieve_matching_jds, evaluate_matches, MatchResults
from src.analyzer import analyze_skill_gaps, SkillGapAnalysis
from src.roadmap import generate_personalized_roadmap, PersonalizedRoadmap
from src.interview import generate_mock_interview, MockInterviewSet, evaluate_candidate_mock_response
from src.agents import get_placement_mentor_agent

# ==========================================
# 1. SETUP PAGE CONFIG & TITLE
# ==========================================
st.set_page_config(
    page_title="CampusHire | AI Placement Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 CampusHire")
st.caption("Agentic AI-Powered Placement Preparation Platform")

# ==========================================
# 2. STATE INITIALIZATION
# ==========================================
# Initialize Session State values to persist data across page interactions
if "parsed_resume" not in st.session_state:
    st.session_state.parsed_resume = None
if "matching_results" not in st.session_state:
    st.session_state.matching_results = None
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "gap_analysis" not in st.session_state:
    st.session_state.gap_analysis = None
if "roadmap" not in st.session_state:
    st.session_state.roadmap = None
if "interview_set" not in st.session_state:
    st.session_state.interview_set = None
if "completed_weeks" not in st.session_state:
    st.session_state.completed_weeks = {}
if "completed_questions" not in st.session_state:
    st.session_state.completed_questions = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mock_answers" not in st.session_state:
    st.session_state.mock_answers = {}
if "mock_feedback" not in st.session_state:
    st.session_state.mock_feedback = {}
if "pending_chat_msg" not in st.session_state:
    st.session_state.pending_chat_msg = ""


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def extract_text_from_uploaded_pdf(uploaded_file) -> str:
    """
    Reads the PDF bytes directly in-memory and extracts raw text
    without saving files to local disk storage.
    """
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_content = []
    for page in doc:
        text_content.append(page.get_text())
    doc.close()
    return "\n".join(text_content).strip()


def clean_agent_output(output) -> str:
    """
    Cleans up the raw Agent output. If it is a string, returns it.
    If it is a list of blocks/dicts, extracts and joins text segments.
    If a raw python_tag escaped due to a model hallucination, it filters it out gracefully.
    """
    # 1. Handle lists
    if isinstance(output, list):
        text_parts = []
        for block in output:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        output_str = "".join(text_parts).strip()
    else:
        output_str = str(output)

    # 2. Intercept raw python_tag escapes
    if "<|python_tag|>" in output_str:
        return (
            "Hi there! I am ready to guide you on your placement journey. "
            "Please upload your resume in the sidebar and enter your target role so we can get started!"
        )
        
    return output_str


def handle_chat_submit():
    """
    Callback function that safely grabs the chat input, stashes it,
    and clears the text input widget to prevent infinite looping reruns.
    """
    user_msg = st.session_state.agent_chat_input.strip()
    if user_msg:
        st.session_state.pending_chat_msg = user_msg
    # Clear the input box value instantly in session state
    st.session_state.agent_chat_input = ""


# ==========================================
# 4. SIDEBAR - FILE UPLOAD & CONFIG
# ==========================================
with st.sidebar:
    st.header("📄 Resume Upload")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    st.header("🎯 Target Role")
    target_role_input = st.text_input(
        "Enter your Target Role", 
        placeholder="e.g., Associate Software Engineer"
    )

    analyze_btn = st.button("🚀 Analyze & Match Resume", use_container_width=True)

    # ==========================================
    # INTEGRATE AGENT CHATBOT INTO SIDEBAR
    # ==========================================
    st.divider()
    st.subheader("💬 Chat with AI Mentor")
    st.caption("Ask questions about your resume, matched roles, or general upskilling advice.")

    # Render previous chat history bubbles
    for role, text in st.session_state.chat_history[-4:]:  # Show last 4 messages for space
        if role == "human":
            st.markdown(f"**👤 You:** {text}")
        else:
            st.markdown(f"**🤖 Mentor:** {text}")

    # Chat text input field (Using the Callback Handler)
    st.text_input(
        "Type your message and press Enter...", 
        key="agent_chat_input",
        on_change=handle_chat_submit
    )

    # Process the chat input safely using the stashed message
    if st.session_state.pending_chat_msg:
        user_msg = st.session_state.pending_chat_msg
        # Clear the stashed message immediately so it can never trigger again on next clicks
        st.session_state.pending_chat_msg = ""
        
        with st.spinner("Mentor is typing..."):
            try:
                # Retrieve the LangChain agent executor
                agent_exec = get_placement_mentor_agent()
                
                # Execute the agent
                response = agent_exec.invoke({
                    "input": user_msg,
                    "chat_history": st.session_state.chat_history
                })
                
                # Clean the raw output blocks into a human-readable string
                clean_response = clean_agent_output(response["output"])
                
                # Store cleaned message in history
                st.session_state.chat_history.append(("human", user_msg))
                st.session_state.chat_history.append(("ai", clean_response))
                
                # Force UI reload to display messages instantly
                st.rerun()
                
            except Exception as e:
                # Check for temporary Gemini server demand errors (503)
                if "503" in str(e):
                    st.error("Google's servers are experiencing high demand right now. Please try sending your chat message again in a moment.")
                else:
                    st.error(f"Chat error: {e}")

# Processing Trigger Block for Dashboard Ingestion
if analyze_btn:
    if not uploaded_file:
        st.sidebar.error("Please upload your PDF resume first.")
    elif not target_role_input.strip():
        st.sidebar.error("Please enter a target role.")
    else:
        with st.spinner("Step 1/2: Parsing and extracting resume content..."):
            try:
                raw_text = extract_text_from_uploaded_pdf(uploaded_file)
                st.session_state.parsed_resume = parse_resume_to_json(raw_text)
                st.session_state.parsed_resume.target_role = target_role_input.strip()
                st.toast("Resume parsed successfully!", icon="✅")
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.error("Google API Rate limit reached. Please wait a moment and try again.")
                else:
                    st.error(f"Error parsing resume: {e}")
                st.stop()

        with st.spinner("Step 2/2: Retrieving matches and calculating scores..."):
            try:
                skills_str = ", ".join(st.session_state.parsed_resume.skills)
                query_text = f"Target Role: {target_role_input.strip()}. Skills: {skills_str}"
                matched_docs = retrieve_matching_jds(query_text, k=3)
                
                st.session_state.matching_results = evaluate_matches(
                    st.session_state.parsed_resume, 
                    matched_docs
                )
                
                # Reset downstream selected state to prevent stale data display
                st.session_state.selected_job = None
                st.session_state.gap_analysis = None
                st.session_state.roadmap = None
                st.session_state.interview_set = None
                st.session_state.mock_answers = {}
                st.session_state.mock_feedback = {}
                
                st.toast("Matches calculated successfully!", icon="🏆")
                st.rerun()
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.error("Google API Rate limit reached. Please wait a moment and click match again.")
                else:
                    st.error(f"Error calculating matches: {e}")


# ==========================================
# 5. MAIN PAGE LAYOUT
# ==========================================
if st.session_state.parsed_resume is None:
    # If no data exists, display a welcoming dashboard
    st.info("👋 Welcome to CampusHire! Please upload your PDF resume and enter a target role in the sidebar to get started.")
    
    # Showcase simple instructions
    st.subheader("How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🎯 **1. Upload & Specify**\nUpload your PDF resume and provide the target role you want to crack.")
    with col2:
        st.markdown("🔎 **2. Match JDs**\nOur FAISS vector database scans real job descriptions to calculate alignment scores.")
    with col3:
        st.markdown("📈 **3. Upskill & Practice**\nGet custom gap reports, week-by-week roadmaps, and mock questions.")

else:
    # Build main working dashboard tab interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Profile Summary", 
        "💼 Job Matches", 
        "📊 Gap & Roadmap", 
        "🎙️ Mock Interview", 
        "✅ Learning Tracker"
    ])

    # -----------------------------
    # TAB 1: PROFILE SUMMARY
    # -----------------------------
    with tab1:
        st.header(f"Profile: {st.session_state.parsed_resume.name}")
        
        contact = st.session_state.parsed_resume.contact
        st.write(f"📧 **Email:** {contact.email} | 📞 **Phone:** {contact.phone}")
        st.write(f"🌐 **LinkedIn:** {contact.linkedin} | 💻 **GitHub:** {contact.github}")
        st.write(f"🎯 **Deduced Core Focus:** {st.session_state.parsed_resume.target_role}")
        st.divider()

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🛠️ Extracted Skills")
            for skill in st.session_state.parsed_resume.skills:
                st.markdown(f"- {skill}")

            st.subheader("🎓 Education Details")
            for edu in st.session_state.parsed_resume.education:
                st.markdown(f"**{edu.degree}**")
                st.markdown(f"*{edu.institution}* ({edu.year or 'N/A'}) - GPA: {edu.gpa or 'N/A'}")

        with col2:
            st.subheader("💼 Work & Internship History")
            if not st.session_state.parsed_resume.experience:
                st.info("No work experience listed.")
            else:
                for exp in st.session_state.parsed_resume.experience:
                    st.markdown(f"**{exp.role}** at *{exp.company}* ({exp.duration or 'N/A'})")
                    for pt in exp.description:
                        st.markdown(f"- {pt}")
                    st.write("")

            st.subheader("🚀 Projects")
            if not st.session_state.parsed_resume.projects:
                st.info("No projects listed.")
            else:
                for proj in st.session_state.parsed_resume.projects:
                    st.markdown(f"**{proj.title}**")
                    st.caption(f"Technologies: {', '.join(proj.technologies)}")
                    st.write(proj.description)
                    st.write("")

    # -----------------------------
    # TAB 2: JOB MATCHES
    # -----------------------------
    with tab2:
        st.header("🏆 Recommended Job Recommendations")
        st.write("These jobs were extracted from your database using vector similarity and graded by Llama.")
        
        if st.session_state.matching_results:
            for idx, match in enumerate(st.session_state.matching_results.matches):
                # Clean up score formatting (strips '%' if Llama included it)
                clean_score = str(match.match_score).replace("%", "").strip()
                
                with st.expander(f"📌 {match.title} | Match Score: {clean_score}%", expanded=(idx==0)):
                    col_score, col_details = st.columns([1, 4])
                    
                    with col_score:
                        st.metric(label="Match Alignment", value=f"{clean_score}%")
                        
                        if st.button("🎯 Select This Role", key=f"sel_role_{match.jd_id}"):
                            # We process the generation safely, and only apply state variables
                            # once ALL three elements have generated successfully.
                            with st.spinner("Generating detailed skill gap report and roadmap..."):
                                try:
                                    docs = retrieve_matching_jds(match.title, k=5)
                                    target_doc = next((d for d in docs if d.metadata.get("id") == match.jd_id), docs[0])
                                    
                                    # Call 1: Gap Analysis
                                    gap_analysis = analyze_skill_gaps(
                                        st.session_state.parsed_resume,
                                        target_doc.page_content,
                                        match.title
                                    )
                                    
                                    # Introduce a small 2.5-second sleep to respect Google 15/20 RPM limits
                                    time.sleep(2.5)
                                    
                                    # Call 2: Roadmap Generation
                                    roadmap = generate_personalized_roadmap(
                                        st.session_state.parsed_resume,
                                        gap_analysis
                                    )
                                    
                                    # Introduce a small 2.5-second sleep to respect Google 15/20 RPM limits
                                    time.sleep(2.5)
                                    
                                    # Call 3: Interview Question Sets
                                    interview_set = generate_mock_interview(
                                        st.session_state.parsed_resume,
                                        gap_analysis,
                                        num_questions=6
                                    )
                                    
                                    # Committing to State (Atomic/All-or-Nothing to prevent NoneType crashes)
                                    st.session_state.selected_job = match
                                    st.session_state.gap_analysis = gap_analysis
                                    st.session_state.roadmap = roadmap
                                    st.session_state.interview_set = interview_set
                                    
                                    # Initialize progress checklists and text responses (Use STR keys for Llama-8B safety)
                                    st.session_state.completed_weeks = {str(w.week_number): False for w in roadmap.weeks}
                                    st.session_state.completed_questions = {str(q.question_id): False for q in interview_set.questions}
                                    st.session_state.mock_answers = {}
                                    st.session_state.mock_feedback = {}
                                    
                                    st.toast(f"Preparation Pipeline generated for {match.title}!", icon="🎯")
                                    st.rerun()
                                    
                                except Exception as e:
                                    # If any generation step failed, wipe variables to prevent rendering broken states
                                    st.session_state.selected_job = None
                                    st.session_state.gap_analysis = None
                                    st.session_state.roadmap = None
                                    st.session_state.interview_set = None
                                    st.session_state.mock_answers = {}
                                    st.session_state.mock_feedback = {}
                                    
                                    # Check for specific 429 rate limit errors
                                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                                        st.error(
                                            "Google API Rate limit reached. To respect free-tier limitations, "
                                            "please wait 30 seconds and click 'Select This Role' again."
                                        )
                                    else:
                                        st.error(f"Failed to generate custom timeline assets: {e}")
                                    
                    with col_details:
                        st.markdown("**Matched Skills:**")
                        st.write(", ".join(match.matched_skills) if match.matched_skills else "None")
                        
                        st.markdown("**Missing Skills:**")
                        st.write(", ".join(match.missing_skills) if match.missing_skills else "None")
                        
                        st.markdown("**Reasoning:**")
                        st.write(match.reasoning)

    # -----------------------------
    # TAB 3: GAP & ROADMAP
    # -----------------------------
    with tab3:
        # Require both the selected job AND the generated assets to be present
        if not st.session_state.selected_job or not st.session_state.gap_analysis or not st.session_state.roadmap:
            st.info("👈 Please go to the **Job Matches** tab and click 'Select This Role' on a job to generate your gap report and roadmap.")
        else:
            st.header(f"📈 Prep Analytics for {st.session_state.selected_job.title}")
            
            st.subheader("🎯 Key Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.success("💪 **Your Top Strengths:**")
                for s in st.session_state.gap_analysis.strengths:
                    st.markdown(f"- {s}")
            with col2:
                st.info("💡 **Core Recommendations:**")
                st.write(st.session_state.gap_analysis.key_recommendation)
            
            st.divider()
            
            st.subheader("⚠️ Core Skill Gaps")
            if not st.session_state.gap_analysis.missing_skills:
                st.success("No critical skill gaps identified for this role!")
            else:
                gap_data = []
                for gap in st.session_state.gap_analysis.missing_skills:
                    gap_data.append({
                        "Skill": gap.skill_name,
                        "Priority": gap.priority,
                        "Difficulty": gap.difficulty_to_learn,
                        "Why It Matters": gap.importance_reason
                    })
                df_gap = pd.DataFrame(gap_data)
                st.dataframe(df_gap, use_container_width=True, hide_index=True)
                
            st.divider()

            # Show Weekly Roadmap (Flat loop rendering)
            st.subheader("📅 Personalized Week-by-Week Syllabus")
            for week in st.session_state.roadmap.weeks:
                with st.expander(f"🗓️ Week {week.week_number}: {week.theme}"):
                    st.markdown(f"**🔬 Hands-on Mini Project:** {week.hands_on_project}")
                    st.markdown(f"**🎯 Week Milestone:** *{week.milestone}*")
                    st.write("")
                    st.markdown("**📋 Daily study tasks:**")
                    
                    # Flat-rendered task strings (avoids double-nesting parser crashes)
                    for task in week.tasks:
                        st.markdown(f"- {task}")

    # -----------------------------
    # TAB 4: MOCK INTERVIEW (INTERACTIVE)
    # -----------------------------
    with tab4:
        # Require the interview set to be successfully loaded
        if not st.session_state.selected_job or not st.session_state.interview_set:
            st.info("👈 Please select a job role first inside the **Job Matches** tab.")
        else:
            st.header("🎙️ Dynamic Placement Panel Mock Interview")
            st.write("Type your response inside the text area below, then click 'Submit Response' to receive professional grading from the AI Panel.")

            for q in st.session_state.interview_set.questions:
                with st.container(border=True):
                    st.markdown(f"**Question {q.question_id}** <span style='background-color:rgba(128,128,128,0.1); padding:2px 6px; border-radius:4px; font-size:12px;'>{q.category}</span>", unsafe_allow_html=True)
                    st.subheader(q.question)
                    st.caption(f"🔑 **Concept Tested:** {q.concept_tested} | 📈 **Difficulty:** {q.difficulty}")
                    
                    # 1. Text Area for user to type their answer (using STR key for Pydantic safety)
                    user_draft = st.text_area(
                        "Type your response to practice:",
                        value=st.session_state.mock_answers.get(str(q.question_id), ""),
                        key=f"ta_ans_{q.question_id}",
                        placeholder="e.g., In my experience, I implement this by...",
                        height=120
                    )
                    
                    # Save local edits to session state instantly
                    st.session_state.mock_answers[str(q.question_id)] = user_draft
                    
                    col_submit, col_hint = st.columns([1, 1])
                    
                    with col_submit:
                        # 2. Submit Button to trigger grading
                        if st.button("📝 Submit Response for AI Panel Grading", key=f"btn_grade_{q.question_id}"):
                            if not user_draft.strip():
                                st.warning("Please type out a draft response before submitting.")
                            else:
                                with st.spinner("AI Panel is reviewing your response..."):
                                    try:
                                        # Execute AI evaluation
                                        feedback = evaluate_candidate_mock_response(
                                            q.question,
                                            q.hint_or_ideal_response,
                                            user_draft
                                        )
                                        # Save feedback to state (use STR key)
                                        st.session_state.mock_feedback[str(q.question_id)] = feedback
                                        st.toast(f"Question {q.question_id} response graded!", icon="📝")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to grade answer: {e}")
                                        
                    with col_hint:
                        # 3. Traditional hint expander
                        with st.expander("🔑 View Hint & Ideal Response Guide"):
                            st.write(q.hint_or_ideal_response)
                    
                    # 4. Display AI Feedback panel if graded
                    if str(q.question_id) in st.session_state.mock_feedback:
                        st.divider()
                        st.markdown("### 🤖 AI Panel Evaluation Feedback")
                        st.info(st.session_state.mock_feedback[str(q.question_id)])

    # -----------------------------
    # TAB 5: LEARNING TRACKER
    # -----------------------------
    with tab5:
        # Require all tracked components to be loaded
        if not st.session_state.selected_job or not st.session_state.roadmap or not st.session_state.interview_set:
            st.info("👈 Please select a job role first inside the **Job Matches** tab.")
        else:
            st.header("✅ Interactive Placement Preparation Tracker")
            st.write("Track your progress across roadmap weeks and interview prep questions to measure placement readiness.")
            col_weeks, col_qs = st.columns(2)
            
            with col_weeks:
                st.subheader("📅 Weekly Milestones")
                for week in st.session_state.roadmap.weeks:
                    st.session_state.completed_weeks[str(week.week_number)] = st.checkbox(
                        f"Complete Week {week.week_number}: {week.theme}",
                        value=st.session_state.completed_weeks.get(str(week.week_number), False),
                        key=f"chk_wk_{week.week_number}"
                    )
                    
            with col_qs:
                st.subheader("🎙️ Answered Interview Prep")
                for q in st.session_state.interview_set.questions:
                    st.session_state.completed_questions[str(q.question_id)] = st.checkbox(
                        f"Answered Q{q.question_id}: {q.category}",
                        value=st.session_state.completed_questions.get(str(q.question_id), False),
                        key=f"chk_q_{q.question_id}"
                    )

            st.divider()
            tot_items = len(st.session_state.completed_weeks) + len(st.session_state.completed_questions)
            completed_items = sum(st.session_state.completed_weeks.values()) + sum(st.session_state.completed_questions.values())
            
            if tot_items > 0:
                progress_percentage = int((completed_items / tot_items) * 100)
                st.subheader(f"📊 Overall Preparation Progress: {progress_percentage}%")
                st.progress(completed_items / tot_items)
                if progress_percentage == 100:
                    st.balloons()
                    st.success("🎉 Incredible! You have fully completed your custom placement preparation roadmap! You are ready to crack your interview.")