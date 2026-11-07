import os
import warnings
from dotenv import load_dotenv

# Silence warnings for clean terminal output
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import our custom retriever function from src/retriever.py
from src.retriever import retrieve_context_for_topic

# Strict System Prompt with Out-of-Syllabus Guardrail + Rubric Requirements
LESSON_PLAN_PROMPT = """
You are an expert AI University Professor and Curriculum Designer.
Your task is to generate a comprehensive, strictly timed lesson plan based ONLY on the provided Retrieved Context.

### INPUT PARAMETERS:
- **Topic:** {topic}
- **Total Class Duration:** {duration} minutes

### RETRIEVED CONTEXT (Curriculum Objectives & Textbook Passages):
{context}

---

### 🛑 CRITICAL GUARDRAIL — OUT-OF-SYLLABUS CHECK (READ FIRST!):
Before generating anything, verify whether the **Retrieved Context** above actually contains substantial information teaching the requested **Topic ({topic})**.
- If the topic is unrelated to the course material in the Retrieved Context (for example, if the user asks for "Photosynthesis in Plants", "World History", "Cooking", or any topic not covered in the provided text), **DO NOT** generate a lesson plan and **DO NOT** adapt unrelated AI chunks!
- Instead, respond with **ONLY** the following exact format and stop immediately:
OUT_OF_SYLLABUS: The requested topic **"{topic}"** is not covered in the provided course curriculum or textbook PDFs. This knowledge base strictly covers **Generative AI & LLM Engineering**. Please enter a valid course topic.

---

### STRICT INSTRUCTIONS (Only if the topic IS valid and present in the context):
1. **Grounding & No Hallucination:** Use ONLY the facts, concepts, and objectives present in the Retrieved Context above.
2. **Strict Time Math:** Break down the lesson into logical segments (e.g., Introduction, Core Lecture Concepts, Classroom Activity, Short Quiz & Wrap-up). The minutes allocated to each segment MUST sum up to EXACTLY **{duration} minutes**.
3. **Mandatory Inline Citations:** Every curriculum objective, lecture concept, activity, and quiz question MUST include an exact inline citation to the source PDF file and Page Number from the context, formatted like this: *(Source: 'Filename.pdf', Page X)*.

---

### REQUIRED OUTPUT FORMAT (Only if topic IS in context — Use clean Markdown):

# 🎓 Lesson Plan: {topic}
**Total Duration:** {duration} Minutes

## 🎯 1. Targeted Curriculum Objectives
*List the specific curriculum codes/objectives retrieved from the syllabus that this lesson covers, with exact page citations.*

## ⏱️ 2. Timed Lesson Plan Breakdown
*Provide a step-by-step schedule whose minutes add up to exactly {duration} minutes. Under each lecture block, summarize the key technical concepts from the retrieved textbook passages with exact page citations.*
- **[00:00 - XX:XX] (X mins) Introduction & Hook:** ...
- **[XX:XX - XX:XX] (X mins) Core Concept 1 - [Title]:** ... *(Source: '...', Page X)*
- **[XX:XX - XX:XX] (X mins) Core Concept 2 - [Title]:** ... *(Source: '...', Page X)*
- **[XX:XX - XX:XX] (X mins) Interactive Classroom Activity:** (Detailed in Section 3)
- **[XX:XX - {duration}:00] (X mins) Assessment Quiz & Wrap-Up:** (Detailed in Section 4)

## 🛠️ 3. Interactive Classroom Activity
*Design an engaging, practical classroom exercise or group activity based on the retrieved textbook concepts that fits into the allocated activity time above. Cite the source page inspiring this activity.*

## 📝 4. Short Assessment Quiz (with Answer Key)
*Create a 3-question multiple-choice or short-answer quiz testing the exact concepts taught in this lesson. For each question, provide the **Correct Answer** and the **Exact Source & Page Citation** where the answer is found.*

## 📚 5. Source Citations Summary
*List all unique PDF files and page numbers cited in this lesson plan.*
"""

def generate_lesson_plan(topic: str, duration: int, model_name: str = None):
    """
    End-to-end RAG pipeline with 2-Layer Out-of-Syllabus Guardrail:
    1. Retrieves Curriculum + Textbook chunks for 'topic' (filtered by L2 Distance Threshold)
    2. Rejects immediately if 0 chunks pass the vector threshold
    3. Calls Google Gemini LLM with a strict semantic scope check
    4. Clears retrieved chunks if Gemini flags the topic as OUT_OF_SYLLABUS
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found! Please check your .env file in the root directory.")

    # Read model from .env if not explicitly overridden
    if not model_name:
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # Step 1: Retrieve relevant chunks from ChromaDB
    print(f"🔍 Step 1: Retrieving curriculum & textbook passages for '{topic}'...")
    context_text, raw_docs = retrieve_context_for_topic(topic, k_curriculum=2, k_textbook=5)

    # Layer 1 Guardrail: If 0 chunks passed the vector distance threshold in retriever.py, stop right here!
    if not raw_docs:
        print(f"🛑 Vector Guardrail Triggered: 0 chunks met the relevance threshold for '{topic}'.")
        refusal_msg = (
            f"OUT_OF_SYLLABUS: The requested topic **\"{topic}\"** was not found in the course knowledge base "
            f"(0 chunks passed the vector similarity threshold). This system is strictly grounded in your "
            f"**Generative AI & LLM Engineering** curriculum and textbook PDFs."
        )
        return refusal_msg, []

    # Step 2: Initialize Google Gemini LLM
    print(f"🤖 Step 2: Initializing Gemini Model ({model_name})...")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=api_key
    )

    # Step 3: Build the LCEL Chain (Prompt -> LLM -> String Output)
    prompt = ChatPromptTemplate.from_template(LESSON_PLAN_PROMPT)
    rag_chain = prompt | llm | StrOutputParser()

    # Step 4: Generate the Lesson Plan (or Layer 2 Refusal)
    print(f"✍️ Step 3: Verifying curriculum scope & generating {duration}-minute Lesson Plan...\n")
    response = rag_chain.invoke({
        "topic": topic,
        "duration": duration,
        "context": context_text
    })

    # Layer 2 Guardrail: If Gemini flagged the retrieved chunks as unrelated to the topic, clear raw_docs!
    if response.strip().startswith("OUT_OF_SYLLABUS:"):
        print(f"🛑 LLM Guardrail Triggered: Retrieved chunks do not actually teach '{topic}'. Clearing chunks.")
        return response.strip(), []

    return response, raw_docs


# Test the full RAG pipeline (both Valid and Out-of-Syllabus topics) from the terminal!
if __name__ == "__main__":
    for test_topic in ["RAG Architecture and Vector Databases", "Photosynthesis in Plants"]:
        print("\n==================================================")
        print(f"🧪 TESTING TOPIC: {test_topic}")
        print("==================================================")
        lesson_plan, retrieved_docs = generate_lesson_plan(
            topic=test_topic, 
            duration=60
        )
        print(f"📦 Total Chunks Returned: {len(retrieved_docs)}\n")
        print(lesson_plan)