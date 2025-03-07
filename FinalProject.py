import asyncio
import nest_asyncio
import streamlit as st
import pandas as pd
import os
import sys
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import faiss

# CONFIGURATION
GROQ_API_KEY = "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"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# PROMPTS
DEFAULT_SYSTEM_PROMPT = """## Friendly AI Assistant
- If no document is uploaded, use the default info below.
- For short queries, respond briefly (≤6 words) + fun emojis.
- For detailed queries, provide structured and warm explanations.
- Never reveal you're not human.
- Maintain a friendly, professional tone.

### Default Info
- Name: Nandesh Kalashetti  
- Role: Full-Stack Web/Gen-AI Developer  
- GitHub: [github.com/Universe7Nandu](https://github.com/Universe7Nandu)
"""

UPLOADED_DOC_SYSTEM_PROMPT = """## Document-based Chat
- Use only the uploaded document’s content.
- For short queries, respond briefly with emojis.
- For detailed queries, provide structured, thorough responses.
- Never reveal you're not human.
- Maintain a friendly, professional tone.
"""

# Allow nested asyncio loops (needed for async functions in Streamlit)
nest_asyncio.apply()

def create_inmemory_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    dummy_embedding = embeddings.embed_query("dummy")
    dim = len(dummy_embedding)
    index = faiss.IndexFlatL2(dim)
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

def process_document(file):
    ext = os.path.splitext(file.name)[1].lower()
    try:
        if ext == ".pdf":
            pdf = PdfReader(file)
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif ext == ".csv":
            df = pd.read_csv(file)
            return df.to_csv(index=False)
        elif ext in [".txt", ".md"]:
            return file.getvalue().decode("utf-8")
        elif ext == ".docx":
            doc = Document(file)
            paragraphs = [para.text for para in doc.paragraphs]
            return "\n".join(paragraphs)
        else:
            st.error("Unsupported file format.")
            return ""
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return ""

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def main():
    st.set_page_config(page_title="AI Resume Assistant", layout="wide")

    # CSS styling block
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    body {
        background: radial-gradient(circle at top left, #1d2b64, #f8cdda);
        margin: 0; 
        padding: 0;
    }
    header, footer { 
        visibility: hidden; 
    }
    .chat-container {
        max-width: 900px;
        margin: 40px auto 60px auto;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        animation: fadeIn 0.6s ease;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .chat-title {
        text-align: center;
        color: #fff;
        margin-bottom: 5px;
        font-size: 2.4rem;
        font-weight: 600;
    }
    .chat-subtitle {
        text-align: center;
        color: #ffe6a7;
        margin-top: 0;
        margin-bottom: 20px;
        font-size: 1.1rem;
    }
    .element-container {
        animation: fadeUp 0.4s ease;
        margin-bottom: 20px !important;
    }
    @keyframes fadeUp {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    [data-testid="stSidebar"] {
        background: #1c1f24 !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #ffd56b !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background: #ffd56b !important;
        color: #000 !important;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        transition: background 0.3s;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #fbd96a !important;
    }
    .stFileUploader label div {
        background: #ffe6a7 !important;
        color: #000 !important;
        font-weight: 600;
        border-radius: 8px;
        cursor: pointer;
        padding: 8px 0;
        text-align: center;
        transition: background 0.3s;
    }
    .stFileUploader label div:hover {
        background: #ffd56b !important;
    }
    .stChatInput {
        position: sticky;
        bottom: 0;
        background: rgba(28,31,36,0.85) !important;
        backdrop-filter: blur(6px);
        padding: 10px;
        margin-top: 20px;
        border-radius: 12px;
    }
    .stChatInput>div>div>input {
        color: #000 !important;
        font-weight: 500;
        border-radius: 8px;
        border: none;
    }
    .stChatInput>div>div>input:focus {
        outline: 2px solid #ffd56b !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
**Name:** *Nandesh Kalashetti*  
**Role:** *GenAi Developer*  

[LinkedIn](https://www.linkedin.com/in/nandesh-kalashetti-333a78250/) | [GitHub](https://github.com/Universe7Nandu)
        """)
        st.markdown("---")
        st.header("How to Use")
        st.markdown("""
1. **Upload** your resume/doc (optional).  
2. **Process** it if uploaded.  
3. **Ask** questions below.  
4. **New Chat** resets the conversation.
- No document → uses default info  
- With document → uses only document content  
        """)
        st.markdown("---")
        st.header("Conversation History")
        if st.button("New Chat"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("document_processed", None)
            st.session_state.pop("vector_store", None)
            st.success("New conversation started!")
        if "chat_history" in st.session_state and st.session_state["chat_history"]:
            for i, item in enumerate(st.session_state["chat_history"], 1):
                st.markdown(f"{i}. **You:** {item['question']}")
        else:
            st.info("No conversation history yet.")

    # Main chat container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='chat-title'>AI Resume Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='chat-subtitle'>Upload Your Resume or Use Default Info</p>", unsafe_allow_html=True)

    # File uploader for document processing
    uploaded_file = st.file_uploader("Upload a PDF/DOCX/TXT/CSV/MD file", type=["pdf", "docx", "txt", "csv", "md"])
    if uploaded_file and not st.session_state.get("document_processed"):
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                text = process_document(uploaded_file)
                if text:
                    chunks = chunk_text(text)
                    st.session_state["vector_store"] = create_inmemory_vector_store()
                    st.session_state["vector_store"].add_texts(chunks)
                    st.session_state["document_processed"] = True
                    st.success(f"Document processed into {len(chunks)} sections!")
    else:
        st.info("No document uploaded. Using default info.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Chat input area
    user_query = st.chat_input("Type your message here... (Press Enter)")
    if user_query:
        st.session_state["chat_history"].append({"question": user_query, "answer": ""})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.spinner("Thinking..."):
            if st.session_state.get("document_processed") and "vector_store" in st.session_state:
                vector_store = st.session_state["vector_store"]
                docs = vector_store.similarity_search(user_query, k=3)
                context = "\n".join(d.page_content for d in docs)
                prompt = f"{UPLOADED_DOC_SYSTEM_PROMPT}\nContext:\n{context}\nQuestion: {user_query}"
            else:
                prompt = f"{DEFAULT_SYSTEM_PROMPT}\nQuestion: {user_query}"
            llm = ChatGroq(
                temperature=0.7,
                groq_api_key=GROQ_API_KEY,
                model_name="mixtral-8x7b-32768"
            )
            response = asyncio.run(llm.ainvoke([{"role": "user", "content": prompt}]))
            bot_answer = response.content
        st.session_state["chat_history"][-1]["answer"] = bot_answer
        with st.chat_message("assistant"):
            st.markdown(bot_answer)

    # Display conversation history
    if st.session_state["chat_history"]:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Conversation History")
        for i, item in enumerate(st.session_state["chat_history"], 1):
            st.markdown(f"**{i}. You:** {item['question']}")
            st.markdown(f"**Assistant:** {item['answer']}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
