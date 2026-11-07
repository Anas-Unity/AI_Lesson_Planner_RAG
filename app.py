# --- CLOUD DEPLOYMENT COMPATIBILITY FIX FOR CHROMADB ON LINUX ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------------------------------------

import os
import re
import streamlit as st
from dotenv import load_dotenv

# Page Configuration (MUST be the very first Streamlit command)
st.set_page_config(
    page_title="AI Curriculum & Lesson Planner (RAG)",
    page_icon="🎓",
    layout="wide"
)

# Load environment variables locally (.env) or from Streamlit Cloud (st.secrets)
load_dotenv()
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GEMINI_MODEL" in st.secrets:
        os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
except Exception:
    pass

# Auto-build Vector DB on Cloud if it doesn't exist yet
if not os.path.exists("vector_db"):
    with st.spinner("⚙️ First-time Cloud Boot: Indexing PDFs into ChromaDB (takes ~30 seconds)..."):
        from src.ingest import load_and_chunk_pdfs, create_vector_db
        chunks = load_and_chunk_pdfs()
        create_vector_db(chunks)

from src.generator import generate_lesson_plan


def format_chunk_text(text: str) -> str:
    """
    Cleans raw PDF extracted text so it reads cleanly with proper line breaks
    and bullet points instead of a squished wall of text.
    """
    # 1. Put numbered syllabus/module items (e.g., 10.7, 9.12) onto their own bulleted lines
    text = re.sub(r'(?<!\n)\s+(\d{1,2}\.\d{1,2}\s+)', r'\n\n• **\1**', text)
    # Also catch if a chunk starts directly with a section number
    text = re.sub(r'^(\d{1,2}\.\d{1,2}\s+)', r'• **\1**', text)
    
    # 2. Put numbered steps (e.g., "1. ", "2. ", "3. ") onto new lines if squished
    text = re.sub(r'(?<=[.?!])\s+(\d+\.\s+)', r'\n\n**\1**', text)

    # 3. Preserve intentional paragraph breaks from the PDF
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    formatted_lines = []
    
    for line in lines:
        # If line looks like a heading, bullet, or table/diagram row, give it its own paragraph
        if (line.startswith('•') or 
            line.startswith('Module ') or 
            line.startswith('Section ') or
            re.match(r'^\d{1,2}\.\d{1,2}', line) or
            len(line) < 65):
            formatted_lines.append(f"\n\n{line}")
        else:
            formatted_lines.append(line)
            
    cleaned = " ".join(formatted_lines)
    # Clean up excessive newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


# Header Section
st.title("🎓 AI Course Lesson Planner (RAG-Powered)")
st.markdown(
    "Generate strictly timed, source-cited lesson plans, classroom activities, and quizzes "
    "grounded directly in your **Curriculum Objectives** and **Textbook PDFs**."
)
st.divider()

# Sidebar for Configuration & System Info
with st.sidebar:
    st.header("⚙️ System Configuration")
    active_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    st.info(f"**Active LLM (from `.env`):**\n`{active_model}`")
    st.success("**Vector DB:** ChromaDB (`vector_db/`)\n\n**Embeddings:** `all-MiniLM-L6-v2` (384-d)")
    
    st.markdown("---")
    st.subheader("📚 Quick Sample Topics")
    st.caption("Copy-paste any of these to test:")
    st.code("RAG Architecture and Vector Databases", language=None)
    st.code("LoRA and Parameter-Efficient Fine-Tuning", language=None)
    st.code("Transformer Attention Mechanisms", language=None)
    st.code("Prompt Engineering and ReAct Agents", language=None)

# Main Input Form
col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "📖 Enter Lesson Topic:",
        placeholder="e.g., Low-Rank Adaptation (LoRA) and PEFT"
    )

with col2:
    duration = st.number_input(
        "⏱️ Class Duration (Minutes):",
        min_value=15,
        max_value=180,
        value=60,
        step=15
    )

generate_btn = st.button("🚀 Generate Timed Lesson Plan", type="primary", use_container_width=True)

# Execution Logic
if generate_btn:
    if not topic.strip():
        st.warning("⚠️ Please enter a lesson topic first!")
    else:
        with st.spinner(f"🔍 Searching Vector DB & validating curriculum scope for '{topic}'..."):
            try:
                lesson_plan_md, retrieved_docs = generate_lesson_plan(
                    topic=topic.strip(),
                    duration=int(duration)
                )

                # Check if Out-of-Syllabus Guardrail was triggered
                if lesson_plan_md.startswith("OUT_OF_SYLLABUS:"):
                    clean_warning = lesson_plan_md.replace("OUT_OF_SYLLABUS:", "").strip()
                    st.error("🛑 **Out of Curriculum Scope — Hallucination Guardrail Triggered!**")
                    with st.container(border=True):
                        st.markdown(f"### ⚠️ Topic Not Found in Course PDFs\n{clean_warning}")
                        st.info(
                            "💡 **Why did this happen?**\n"
                            "Our RAG pipeline uses a **Two-Layer Hallucination Guardrail**:\n"
                            "1. **Vector Distance Threshold (`MAX_DISTANCE_THRESHOLD = 1.40`):** Discards ChromaDB chunks whose Squared L2 distance is too far from the query.\n"
                            "2. **Semantic Scope Verification:** Prevents the LLM from adapting unrelated chunks when a topic is outside the syllabus."
                        )
                else:
                    st.success("✅ Lesson Plan Generated Successfully!")

                    # Two clean tabs
                    tab1, tab2 = st.tabs([
                        "📄 Generated Lesson Plan", 
                        f"🔍 Retrieved Source Chunks ({len(retrieved_docs)} Chunks)"
                    ])

                    # TAB 1: Formatted Lesson Plan
                    with tab1:
                        with st.container(border=True):
                            st.markdown(lesson_plan_md)
                        
                        st.write("")
                        st.download_button(
                            label="📥 Download Lesson Plan (.md)",
                            data=lesson_plan_md,
                            file_name=f"Lesson_Plan_{topic.replace(' ', '_')}_{duration}min.md",
                            mime="text/markdown",
                            use_container_width=True
                        )

                    # TAB 2: Structured & Readable Retrieved Chunks
                    with tab2:
                        st.markdown("### 🧠 Under the Hood: Retrieved Context from ChromaDB")
                        st.caption(
                            "These are the exact chunks that passed both **Metadata Filtering** and the **L2 Distance Relevance Threshold** before being sent to Gemini."
                        )

                        # Separate Curriculum vs Textbook chunks for cleaner visual hierarchy
                        curr_chunks = [d for d in retrieved_docs if d.metadata.get("doc_type") == "curriculum_objective"]
                        text_chunks = [d for d in retrieved_docs if d.metadata.get("doc_type") == "textbook_passage"]

                        # 1. Display Curriculum Chunks
                        st.subheader(f"🎯 1. Curriculum Objective Chunks ({len(curr_chunks)} Retrieved)")
                        for idx, doc in enumerate(curr_chunks, 1):
                            file_name = doc.metadata.get("file_name", "Unknown File")
                            page_num = doc.metadata.get("page_number", "?")
                            total_pages = doc.metadata.get("total_pages", "?")
                            start_idx = doc.metadata.get("start_index", 0)
                            dist_score = doc.metadata.get("distance_score", "N/A")

                            with st.expander(f"📌 Curriculum Chunk #{idx} — 📄 {file_name} (Page {page_num} of {total_pages}) | 🎯 L2 Distance: {dist_score}", expanded=(idx == 1)):
                                # Clean Metadata Badges
                                m1, m2, m3 = st.columns([2, 1, 1])
                                m1.markdown(f"**📂 Source File:** `{file_name}`")
                                m2.markdown(f"**📖 Page Number:** `Page {page_num} / {total_pages}`")
                                m3.markdown(f"**📐 L2 Distance:** `{dist_score}` *(≤ 1.40)*")
                                st.divider()
                                
                                # Structured & Cleaned Passage Content
                                st.markdown("#### 📜 Extracted Curriculum Content:")
                                st.markdown(format_chunk_text(doc.page_content))
                                
                                st.divider()
                                st.caption(f"🔢 **Chunk Character Start Index:** `{start_idx}` | **Full Path:** `{doc.metadata.get('source')}`")

                        st.write("")

                        # 2. Display Textbook Chunks
                        st.subheader(f"📚 2. Textbook Passage Chunks ({len(text_chunks)} Retrieved)")
                        for idx, doc in enumerate(text_chunks, 1):
                            file_name = doc.metadata.get("file_name", "Unknown File")
                            page_num = doc.metadata.get("page_number", "?")
                            total_pages = doc.metadata.get("total_pages", "?")
                            start_idx = doc.metadata.get("start_index", 0)
                            dist_score = doc.metadata.get("distance_score", "N/A")

                            with st.expander(f"📖 Textbook Chunk #{idx} — 📄 {file_name} (Page {page_num} of {total_pages}) | 🎯 L2 Distance: {dist_score}", expanded=(idx == 1)):
                                # Clean Metadata Badges
                                m1, m2, m3 = st.columns([2, 1, 1])
                                m1.markdown(f"**📂 Source File:** `{file_name}`")
                                m2.markdown(f"**📖 Page Number:** `Page {page_num} / {total_pages}`")
                                m3.markdown(f"**📐 L2 Distance:** `{dist_score}` *(≤ 1.40)*")
                                st.divider()
                                
                                # Structured & Cleaned Passage Content
                                st.markdown("#### 📜 Extracted Textbook Passage:")
                                st.markdown(format_chunk_text(doc.page_content))
                                
                                st.divider()
                                st.caption(f"🔢 **Chunk Character Start Index:** `{start_idx}` | **Full Path:** `{doc.metadata.get('source')}`")

            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")