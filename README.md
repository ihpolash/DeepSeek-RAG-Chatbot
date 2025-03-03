# DeepSeek RAG Chatbot

A simple RAG (Retrieval Augmented Generation) chatbot that allows you to query your documents using local LLMs.

## Features

- Upload and process PDF, DOCX, and TXT files
- Query documents using various LLM models
- Hybrid retrieval with vector search and reranking

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ihpolash/DeepSeek-RAG-Chatbot.git
   cd DeepSeek-RAG-Chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Pull the required models with Ollama:
   ```bash
   ollama pull deepseek-r1:1.5b
   ollama pull nomic-embed-text
   ```

## Usage

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run simple_rag_app.py
   ```

3. Open your browser at http://localhost:8501

4. Upload documents using the sidebar and start asking questions

## Terminal Version

For a simpler command-line interface:

```bash
python terminal_rag.py -i
```

Or to process specific files:

```bash
python terminal_rag.py -f document.pdf -m deepseek
```

See `terminal_rag_usage.md` for more details on the terminal version.

## Configuration

You can configure the application by creating a `.env` file based on the `.env.example` template:

```
OLLAMA_API_URL=http://localhost:11434
MODEL=deepseek-r1:1.5b
```
