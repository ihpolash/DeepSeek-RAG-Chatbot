#!/usr/bin/env python3
"""
Simple Terminal RAG Tester
This script allows you to test RAG capabilities from the terminal.
"""

import os
import sys
import argparse
import requests
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import torch

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
EMBEDDINGS_MODEL = "nomic-embed-text:latest"

# Available LLM models
AVAILABLE_MODELS = {
    "deepseek": "deepseek-r1:1.5b",
    "llama3": "llama3.2:1b",
    "qwen": "qwen2.5:1.5b"
}

def print_colored(text, color="white"):
    """Print colored text to terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def load_documents(file_paths):
    """Load documents from file paths."""
    documents = []
    
    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                print_colored(f"File not found: {file_path}", "red")
                continue
                
            if file_path.endswith(".pdf"):
                print_colored(f"Loading PDF: {file_path}", "cyan")
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                print_colored(f"Loading text file: {file_path}", "cyan")
                loader = TextLoader(file_path)
            else:
                print_colored(f"Unsupported file type: {file_path}", "yellow")
                continue
                
            documents.extend(loader.load())
            print_colored(f"✓ Loaded {file_path}", "green")
        except Exception as e:
            print_colored(f"Error processing {file_path}: {str(e)}", "red")
    
    if not documents:
        print_colored("No valid documents loaded!", "red")
        return None
    
    return documents

def process_documents(documents):
    """Process documents and create vector store."""
    print_colored("\nProcessing documents...", "blue")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print_colored(f"✓ Split into {len(texts)} chunks", "green")
    
    # Create vector store
    try:
        print_colored("Creating vector embeddings (this may take a moment)...", "blue")
        embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL)
        vectorstore = FAISS.from_documents(texts, embeddings)
        print_colored("✓ Vector store created successfully", "green")
        return vectorstore
    except Exception as e:
        print_colored(f"Error creating vector store: {str(e)}", "red")
        return None

def retrieve_documents(vectorstore, query, max_docs=3):
    """Retrieve relevant documents for a query."""
    try:
        docs = vectorstore.similarity_search(query, k=max_docs)
        return docs
    except Exception as e:
        print_colored(f"Error retrieving documents: {str(e)}", "red")
        return []

def query_llm(model_id, prompt):
    """Query the LLM with a prompt."""
    try:
        # First, try with streaming disabled
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model_id,
                "prompt": prompt,
                "stream": False,  # Explicitly disable streaming
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 4096
                }
            },
            timeout=60  # Increase timeout for larger responses
        )
        
        if response.status_code == 200:
            try:
                # Parse the JSON response
                response_json = response.json()
                return response_json.get("response", "")
            except json.JSONDecodeError:
                # If JSON parsing fails, try handling as a stream
                print_colored("Attempting to handle response as stream...", "yellow")
                full_response = ""
                
                # Try again with streaming enabled
                stream_response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": model_id,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.3,
                            "num_ctx": 4096
                        }
                    },
                    stream=True,
                    timeout=60
                )
                
                for line in stream_response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode())
                            token = data.get("response", "")
                            full_response += token
                            # Print a progress indicator
                            print(".", end="", flush=True)
                        except json.JSONDecodeError:
                            continue
                
                print()  # New line after progress dots
                return full_response
                
        else:
            print_colored(f"Error: API returned status code {response.status_code}", "red")
            return f"Error: {response.text}"
    except Exception as e:
        print_colored(f"Error querying LLM: {str(e)}", "red")
        return f"Error: {str(e)}"

def list_available_models():
    """List available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            available = [model["name"] for model in response.json().get("models", [])]
            print_colored("\nAvailable models:", "blue")
            for name, model_id in AVAILABLE_MODELS.items():
                status = "✓" if model_id in available else "✗"
                status_color = "green" if model_id in available else "red"
                print(f"  {name} ({model_id}): ", end="")
                print_colored(status, status_color)
            return True
        else:
            print_colored(f"Error connecting to Ollama API: {response.status_code}", "red")
            return False
    except Exception as e:
        print_colored(f"Error connecting to Ollama: {str(e)}", "red")
        return False

def main():
    parser = argparse.ArgumentParser(description='Terminal RAG Tester')
    parser.add_argument('--files', '-f', nargs='+', help='Paths to PDF or TXT files')
    parser.add_argument('--model', '-m', choices=list(AVAILABLE_MODELS.keys()), default='deepseek', 
                        help='LLM model to use')
    parser.add_argument('--list-models', '-l', action='store_true', help='List available models')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--ollama-url', help='URL of the Ollama API server (e.g., http://your-ec2-ip:11434)')
    
    args = parser.parse_args()
    
    # Update Ollama URL if provided
    if args.ollama_url:
        global OLLAMA_BASE_URL, OLLAMA_API_URL
        OLLAMA_BASE_URL = args.ollama_url
        OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
        print_colored(f"Using Ollama API at: {OLLAMA_BASE_URL}", "cyan")
    
    # Check if Ollama is running
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
        print_colored("✓ Connected to Ollama", "green")
    except Exception:
        print_colored(f"Error: Cannot connect to Ollama at {OLLAMA_BASE_URL}", "red")
        print_colored("Make sure Ollama is running and the URL is correct", "yellow")
        sys.exit(1)
    
    # List models if requested
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    # Load documents if provided
    vectorstore = None
    if args.files:
        documents = load_documents(args.files)
        if documents:
            vectorstore = process_documents(documents)
    
    # Interactive mode
    if args.interactive or not args.files:
        if not args.files:
            print_colored("\nNo files specified. Starting in interactive mode.", "yellow")
        
        while True:
            if not vectorstore:
                print_colored("\nNo documents loaded. You need to load documents first.", "yellow")
                files_input = input("Enter paths to documents (space-separated, or 'q' to quit): ")
                
                if files_input.lower() == 'q':
                    break
                    
                file_paths = files_input.split()
                documents = load_documents(file_paths)
                if documents:
                    vectorstore = process_documents(documents)
                continue
            
            model_choice = args.model
            model_id = AVAILABLE_MODELS[model_choice]
            
            print_colored(f"\nUsing model: {model_id}", "blue")
            print_colored("Enter your question (or 'q' to quit, 'c' to change model, 'n' to load new documents):", "blue")
            query = input("> ")
            
            if query.lower() == 'q':
                break
            elif query.lower() == 'c':
                print_colored("\nAvailable models:", "blue")
                for i, (name, model) in enumerate(AVAILABLE_MODELS.items()):
                    print(f"{i+1}. {name} ({model})")
                try:
                    choice = int(input("Select model (number): ")) - 1
                    if 0 <= choice < len(AVAILABLE_MODELS):
                        model_choice = list(AVAILABLE_MODELS.keys())[choice]
                    else:
                        print_colored("Invalid choice. Using current model.", "yellow")
                except ValueError:
                    print_colored("Invalid input. Using current model.", "yellow")
                continue
            elif query.lower() == 'n':
                vectorstore = None
                continue
            
            # Retrieve relevant documents
            print_colored("\nRetrieving relevant documents...", "blue")
            docs = retrieve_documents(vectorstore, query, max_docs=3)
            
            if not docs:
                print_colored("No relevant documents found!", "yellow")
                continue
                
            # Create context from documents
            context = "\n".join(
                f"[Document {i+1}]: {doc.page_content}" 
                for i, doc in enumerate(docs)
            )
            
            # Create prompt
            prompt = f"""Use the following context from documents to answer the question:
            
            {context}
            
            Question: {query}
            Answer:"""
            
            # Query LLM
            print_colored("\nQuerying LLM...", "blue")
            response = query_llm(model_id, prompt)
            
            # Display response
            print_colored("\n=== Answer ===", "green")
            print(response)
            print_colored("=============", "green")
    
    # Non-interactive mode with files and query
    elif vectorstore:
        model_id = AVAILABLE_MODELS[args.model]
        print_colored(f"\nUsing model: {model_id}", "blue")
        print_colored("Enter your question:", "blue")
        query = input("> ")
        
        # Retrieve relevant documents
        print_colored("\nRetrieving relevant documents...", "blue")
        docs = retrieve_documents(vectorstore, query, max_docs=3)
        
        if not docs:
            print_colored("No relevant documents found!", "yellow")
            sys.exit(1)
            
        # Create context from documents
        context = "\n".join(
            f"[Document {i+1}]: {doc.page_content}" 
            for i, doc in enumerate(docs)
        )
        
        # Create prompt
        prompt = f"""Use the following context from documents to answer the question:
        
        {context}
        
        Question: {query}
        Answer:"""
        
        # Query LLM
        print_colored("\nQuerying LLM...", "blue")
        response = query_llm(model_id, prompt)
        
        # Display response
        print_colored("\n=== Answer ===", "green")
        print(response)
        print_colored("=============", "green")

if __name__ == "__main__":
    main() 