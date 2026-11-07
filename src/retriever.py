import os
import warnings
from functools import lru_cache

# Silence warnings for clean output
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTOR_DB_PATH = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum Squared L2 Distance allowed for a chunk to be considered relevant.
# Lower distance = closer semantic match. Unrelated topics (like 'Photosynthesis') have distance > 1.45
MAX_DISTANCE_THRESHOLD = 1.40

@lru_cache(maxsize=1)
def get_vector_db():
    """
    Loads the existing Chroma Vector DB and Embedding Model into memory ONCE
    and caches it for instant retrieval on subsequent queries.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(
            f"❌ Vector DB not found at '{VECTOR_DB_PATH}'. Please run 'python src/ingest.py' first!"
        )
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    return vector_db

def retrieve_context_for_topic(topic: str, k_curriculum: int = 2, k_textbook: int = 5):
    """
    Retrieves both Curriculum Objectives and Textbook Passages for a given topic
    using Metadata Filtering + Distance Score Thresholding.
    Discards irrelevant chunks when an out-of-syllabus topic is searched.
    """
    vector_db = get_vector_db()

    # 1. Retrieve Curriculum Objectives WITH Distance Scores
    curr_results = vector_db.similarity_search_with_score(
        query=topic,
        k=k_curriculum,
        filter={"doc_type": "curriculum_objective"}
    )
    curriculum_docs = []
    for doc, score in curr_results:
        if score <= MAX_DISTANCE_THRESHOLD:
            doc.metadata["distance_score"] = round(float(score), 4)
            curriculum_docs.append(doc)

    # 2. Retrieve Textbook Passages WITH Distance Scores
    text_results = vector_db.similarity_search_with_score(
        query=topic,
        k=k_textbook,
        filter={"doc_type": "textbook_passage"}
    )
    textbook_docs = []
    for doc, score in text_results:
        if score <= MAX_DISTANCE_THRESHOLD:
            doc.metadata["distance_score"] = round(float(score), 4)
            textbook_docs.append(doc)

    all_raw_docs = curriculum_docs + textbook_docs

    # If NO chunks passed the relevance threshold, return empty context immediately
    if not all_raw_docs:
        return "", []

    # 3. Format the retrieved chunks into a clean string with Citation Tags for the LLM
    formatted_context = "=== RETRIEVED CURRICULUM OBJECTIVES ===\n"
    if curriculum_docs:
        for i, doc in enumerate(curriculum_docs, 1):
            source_file = doc.metadata.get("file_name", "Unknown File")
            page_num = doc.metadata.get("page_number", "?")
            formatted_context += f"\n[Curriculum Source {i}: '{source_file}', Page {page_num}]\n"
            formatted_context += f"{doc.page_content}\n"
    else:
        formatted_context += "No matching curriculum objectives met the relevance threshold.\n"

    formatted_context += "\n=== RETRIEVED TEXTBOOK PASSAGES ===\n"
    if textbook_docs:
        for i, doc in enumerate(textbook_docs, 1):
            source_file = doc.metadata.get("file_name", "Unknown File")
            page_num = doc.metadata.get("page_number", "?")
            formatted_context += f"\n[Textbook Source {i}: '{source_file}', Page {page_num}]\n"
            formatted_context += f"{doc.page_content}\n"
    else:
        formatted_context += "No matching textbook passages met the relevance threshold.\n"

    return formatted_context, all_raw_docs


if __name__ == "__main__":
    for test_topic in ["RAG and Vector Databases", "Photosynthesis in Plants"]:
        print(f"\n🔎 Testing Retrieval for Topic: '{test_topic}'...")
        context_text, raw_docs = retrieve_context_for_topic(test_topic)
        print(f"✅ Relevant Chunks Passed Threshold: {len(raw_docs)}")