import streamlit as st
import os
import tempfile

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import PromptTemplate

# ---------------- Page Config ----------------
st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📄 PDF Summarizer with Language Selection")

# ---------------- Sidebar ----------------
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

# ---------------- Language Selection ----------------
language = st.selectbox(
    "🌍 Choose summary language",
    [
        "English",
        "Hindi",
        "Bengali",
        "Tamil",
        "Telugu",
        "French",
        "German",
        "Spanish"
    ]
)

# ---------------- NVIDIA API (STREAMLIT CLOUD SAFE) ----------------
if "NVIDIA_API_KEY" not in st.secrets:
    st.error("❌ NVIDIA API key not found. Please add it in Streamlit Secrets.")
    st.stop()

os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]

# ---------------- File Upload ----------------
def upload_files():
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            for f in uploaded_files:
                file_path = os.path.join(temp_dir, f.name)
                with open(file_path, "wb") as file:
                    file.write(f.getvalue())

            loader = PyPDFDirectoryLoader(temp_dir)
            documents = loader.load()
            st.session_state.raw_documents = documents

upload_files()

# ---------------- Vector Embedding ----------------
def vector_embedding():
    if "raw_documents" not in st.session_state:
        st.error("❌ Please upload PDFs first")
        return

    embeddings = NVIDIAEmbeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(st.session_state.raw_documents)

    st.session_state.documents = split_docs
    st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)

# ---------------- Buttons ----------------
if st.button("📌 Create Vector Database"):
    vector_embedding()
    st.success("✅ Vector database created successfully")

if st.button("📝 Summarize PDF"):
    if not openai_api_key:
        st.error("❌ Please enter your OpenAI API key")
        st.stop()

    if "documents" not in st.session_state:
        st.error("❌ Please create vector DB first")
        st.stop()

    # ---------------- LLM ----------------
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0.3
    )

    # ---------------- Prompts ----------------
    map_prompt = PromptTemplate(
        input_variables=["text", "language"],
        template="""
Summarize the following content clearly and concisely in {language}.
Preserve important details.

{text}
"""
    )

    # ---------------- Load Summarization Chain ----------------
    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=map_prompt
    )

    # ---------------- Run Summarization ----------------
    with st.spinner("🔄 Generating summary..."):
        result = chain.invoke({
            "input_documents": st.session_state.documents,
            "language": language
        })

    st.success(f"✅ Summary generated in {language}")
    st.write(result["output_text"])
