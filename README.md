# 🎓 CampusHire: Agentic AI Placement Assistant

CampusHire is an Agentic AI-powered placement preparation platform designed to help students analyze their resumes, match them against real job descriptions using semantic vector search, isolate skill gaps, build personalized weekly learning roadmaps, and practice with an interactive, AI-graded mock interview panel.

---

## 🚀 Key Features

- **👤 ATS Resume Parsing:** Extracts skills, education, projects, and work history from uploaded PDF resumes into structured Pydantic structures using `PyMuPDF` and `gemini-2.0-flash`.
- **💼 Semantic Job Matching:** Queries a pre-indexed local FAISS vector database containing 374 real job descriptions to return the top 3 unique matching roles.
- **📈 Granular Skill Gap Analysis:** Meticulously compares candidate skills against the chosen job description, prioritizing missing items as High, Medium, or Low priority.
- **📅 Personalized Prep Roadmap:** Generates a custom week-by-week learning syllabus featuring targeted study concepts, hands-on mini-projects, and progress milestones.
- **🎙️ Interactive Mock Interview Simulator:** Generates role-specific questions spanning behavioral, technical, and project-based topics, featuring interactive text-input responses with dynamic AI-panel grading and scorecards.
- **💬 Conversational AI Mentor Chatbot:** Includes a LangChain-powered autonomous chatbot (`AgentExecutor`) in the sidebar capable of routing conversational student questions to the appropriate backend tool dynamically.

---

## 🛠️ Tech Stack

- **Frontend Interface:** [Streamlit](https://streamlit.io/)
- **Core Orchestration:** [LangChain](https://www.langchain.com/) and `langchain-classic`
- **Embedding Model:** `gemini-embedding-2-preview`
- **Language Model (LLM):** `gemini-2.5-flash`
- **Vector Database:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **PDF Extraction:** [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)

---

## 💻 Local Installation & Setup

To run CampusHire on your local machine, follow these step-by-step instructions:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/campus_hire.git
cd campus_hire