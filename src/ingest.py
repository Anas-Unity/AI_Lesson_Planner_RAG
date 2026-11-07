import os
import shutil
import warnings

# Silence deprecation warnings for a clean Viva terminal output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define paths relative to the Root Directory
CURRICULUM_PATH = "data/curriculum"
TEXTBOOK_PATH = "data/textbooks"
VECTOR_DB_PATH = "vector_db"

# Define our free, local Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_chunk_pdfs():
    documents = []

    # 1. Load Curriculum PDFs and tag them
    if os.path.exists(CURRICULUM_PATH):
        print(f"📂 Loading Curriculum PDFs from '{CURRICULUM_PATH}'...")
        curr_loader = PyPDFDirectoryLoader(CURRICULUM_PATH)
        curr_docs = curr_loader.load()
        for doc in curr_docs:
            doc.metadata["doc_type"] = "curriculum_objective"
            doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
            # Clean file name for nicer citations later
            doc.metadata["file_name"] = os.path.basename(doc.metadata.get("source", "Unknown"))
        documents.extend(curr_docs)

    # 2. Load Textbook/Course PDFs and tag them
    if os.path.exists(TEXTBOOK_PATH):
        print(f"📚 Loading Textbook PDFs from '{TEXTBOOK_PATH}'...")
        text_loader = PyPDFDirectoryLoader(TEXTBOOK_PATH)
        text_docs = text_loader.load()
        for doc in text_docs:
            doc.metadata["doc_type"] = "textbook_passage"
            doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
            doc.metadata["file_name"] = os.path.basename(doc.metadata.get("source", "Unknown"))
        documents.extend(text_docs)

    print(f"✅ Total Pages Loaded across all PDFs: {len(documents)}")

    if len(documents) == 0:
        print("❌ No PDFs found! Please check your data folders.")
        return []

    # 3. Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Total Chunks Created: {len(chunks)}")
    return chunks

def create_vector_db(chunks):
    if not chunks:
        return

    # Clear out existing database if we are re-running ingestion to avoid duplicate chunks
    if os.path.exists(VECTOR_DB_PATH):
        print(f"🧹 Removing old Vector DB at '{VECTOR_DB_PATH}' to build a fresh one...")
        shutil.rmtree(VECTOR_DB_PATH)

    print(f"\n🧠 Initializing Embedding Model ({EMBEDDING_MODEL_NAME})...")
    print("(Note: First run will download the ~90MB model from HuggingFace)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"💾 Converting {len(chunks)} chunks into vectors and saving to ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    print(f"🎉 SUCCESS! Vector Database created and saved at '{VECTOR_DB_PATH}/'!")
    return vector_db

if __name__ == "__main__":
    chunks = load_and_chunk_pdfs()
    create_vector_db(chunks)