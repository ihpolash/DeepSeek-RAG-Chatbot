# Terminal RAG Tester

This is a command-line tool for testing your RAG (Retrieval Augmented Generation) system with different models without the Streamlit interface.

## Setup

Make sure you have installed all the required dependencies:

```bash
pip install -r requirements.txt
```

And that Ollama is running on your system.

## Usage

### Basic Usage

```bash
python terminal_rag.py -f document1.pdf document2.txt -m deepseek
```

This will:
1. Load the specified documents
2. Process them into a vector store
3. Prompt you to enter a question
4. Answer your question using the DeepSeek model

### Interactive Mode

```bash
python terminal_rag.py -i
```

In interactive mode, you can:
- Load documents
- Ask multiple questions
- Switch between different models
- Load new documents without restarting

### Available Commands in Interactive Mode

When asked for a question, you can enter:
- `q` to quit
- `c` to change the model
- `n` to load new documents

### Check Available Models

```bash
python terminal_rag.py -l
```

This will list all models defined in the script and check which ones are available in your Ollama installation.

### Command Line Options

```
-f, --files    Specify PDF or TXT files to process
-m, --model    Select the model to use (deepseek, llama3, qwen)
-l, --list-models  List available models
-i, --interactive  Run in interactive mode
```

## Examples

Process specific files with the Llama model:
```bash
python terminal_rag.py -f research.pdf notes.txt -m llama3
```

Run in fully interactive mode:
```bash
python terminal_rag.py -i
```

Just check which models are available:
```bash
python terminal_rag.py -l
```

## Troubleshooting

If you encounter any issues:

1. Make sure Ollama is running (`ollama serve`)
2. Check that your Ollama API URL is correct in the `.env` file
3. Verify you have the necessary models pulled in Ollama 