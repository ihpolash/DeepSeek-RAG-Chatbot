import streamlit as st
import os
import torch
from dotenv import load_dotenv
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# Fix for torch classes not found error in Streamlit
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
except (AttributeError, TypeError):
    pass

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://http://44.222.117.180:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_MODEL = os.getenv("MODEL", "deepseek-r1:1.5b")  # Get default from env or fallback

# Available LLM models
AVAILABLE_MODELS = {
    "DeepSeek 1.5B": "deepseek-r1:1.5b",
    "Llama 3.2 1B": "llama3.2:1b",
    "Qwen 2.5 1.5B": "qwen2.5:1.5b"
}

EMBEDDINGS_MODEL = "nomic-embed-text:latest"

# Set up device for PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize reranker
reranker = None
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")

# Streamlit page config
st.set_page_config(page_title="Simple RAG Chatbot", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .stApp { background-color: #f5f7f9; }
        h1 { color: #2e6fac; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #edfaed; }
        .stButton>button { background-color: #4CAF50; color: white; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
if "max_contexts" not in st.session_state:
    st.session_state.max_contexts = 3
if "selected_model" not in st.session_state:
    # Set default model from env variable or fallback
    default_model_name = next((name for name, model in AVAILABLE_MODELS.items() 
                              if model == DEFAULT_MODEL), list(AVAILABLE_MODELS.keys())[0])
    st.session_state.selected_model = default_model_name

# Function to process uploaded PDF documents
def process_documents(uploaded_files):
    documents = []
    
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Process each file
    for file in uploaded_files:
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue
                
            documents.extend(loader.load())
            os.remove(file_path)  # Clean up temp file
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return False
    
    if not documents:
        st.error("No valid documents found.")
        return False
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL)
    st.session_state.vectorstore = FAISS.from_documents(texts, embeddings)
    return True

# Function to retrieve documents based on query
def retrieve_documents(query):
    if not st.session_state.vectorstore:
        return []
    
    # Retrieve relevant documents
    docs = st.session_state.vectorstore.similarity_search(query, k=st.session_state.max_contexts)
    
    # Rerank if reranker is available
    if reranker:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return ranked_docs
    
    return docs

# Main application layout
st.title("📚 Simple RAG Chatbot")
st.caption("Upload PDF documents and chat with them")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF or text documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            success = process_documents(uploaded_files)
            if success:
                st.session_state.documents_loaded = True
                st.success("Documents processed successfully!")
    
    st.markdown("---")
    st.header("⚙️ Settings")
    
    # Model selection dropdown
    st.subheader("🤖 LLM Model")
    selected_model_name = st.selectbox(
        "Choose an LLM model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model_name
    
    # Display model info
    selected_model_id = AVAILABLE_MODELS[selected_model_name]
    st.caption(f"Using model: `{selected_model_id}`")
    
    # Check if model needs to be pulled
    if st.button("Check/Pull Model"):
        with st.spinner(f"Checking if {selected_model_id} is available..."):
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                if selected_model_id in available_models:
                    st.success(f"✅ Model {selected_model_id} is available")
                else:
                    st.warning(f"⚠️ Model {selected_model_id} not found. Pulling now...")
                    pull_response = requests.post(
                        f"{OLLAMA_BASE_URL}/api/pull",
                        json={"name": selected_model_id}
                    )
                    if pull_response.status_code == 200:
                        st.success(f"✅ Successfully pulled {selected_model_id}")
                    else:
                        st.error(f"❌ Failed to pull model: {pull_response.text}")
            else:
                st.error("❌ Failed to connect to Ollama API")
    
    st.markdown("---")
    
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Check if documents are loaded
        if not st.session_state.documents_loaded:
            full_response = "Please upload and process documents first."
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            # Retrieve relevant context
            try:
                docs = retrieve_documents(prompt)
                context = "\n".join(
                    f"[Source {i+1}]: {doc.page_content}" 
                    for i, doc in enumerate(docs)
                )
                
                # Create system prompt with context
                system_prompt = f"""
                Use the following context from documents to answer the user's question:
                
                {context}
                
                Question: {prompt}
                Answer:"""
                
                # Get the selected model ID from the dropdown
                model_id = AVAILABLE_MODELS[st.session_state.selected_model]
                
                # Stream response from Ollama
                response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": model_id,
                        "prompt": system_prompt,
                        "stream": True,
                        "options": {
                            "temperature": st.session_state.temperature,
                            "num_ctx": 4096
                        }
                    },
                    stream=True
                )
                
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode())
                        token = data.get("response", "")
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                        
                        # Stop if we detect the end token
                        if data.get("done", False):
                            break
                            
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; font-size: 12px; color: gray;">
        Simple RAG Chatbot | Built with Streamlit and Ollama
    </div>
""", unsafe_allow_html=True) 