# RAG Workshop

This project demonstrates Retrieval Augmented Generation (RAG) using Google's Gemini API and ChromaDB for vector storage.

## Prerequisites

- Python 3.8+
- Google Gemini API Key
- UV package manager (or pip)

## Setup

1. Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_api_key_here
```

2. Install dependencies:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Run the CLI version (main.py)

This runs the RAG pipeline in the command line and shows the comparison between responses with and without RAG.

```bash
uv run python main.py
```

Or without uv:
```bash
python main.py
```

**What it does:**
- Initializes ChromaDB with sample documents about "41C) 'DFH1"
- Asks a predefined question
- Shows the response WITHOUT RAG (direct from model)
- Shows the response WITH RAG (using retrieved documents)
- Prints everything to the terminal

### Option 2: Run the Streamlit UI (app.py)

This provides an interactive web interface to compare RAG vs non-RAG responses.

```bash
uv run streamlit run app.py
```

Or without uv:
```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

**Features:**
- Side-by-side comparison of responses
- Interactive question input
- View stored documents in sidebar
- See retrieved context for each query
- Full RTL (right-to-left) support for Arabic

## Project Structure

```
rag-workshop/
 main.py              # CLI version - demonstrates RAG pipeline
 app.py               # Streamlit UI - interactive web interface
 .env                 # Environment variables (create this)
 chroma_db/           # ChromaDB persistent storage
 .streamlit/          # Streamlit configuration
   config.toml
 README.md            # This file
```

## How It Works

1. **Document Storage**: Sample documents are embedded using Gemini's embedding model and stored in ChromaDB
2. **Query Processing**: User questions are converted to embeddings
3. **Retrieval**: Most relevant documents are retrieved using vector similarity
4. **Generation**: Gemini generates answers using the retrieved context
5. **Comparison**: Shows the difference between answers with and without RAG

## Sample Questions

Try these questions in the Streamlit app:
- متى تأسست شركة النور ومن هو رئيسها التنفيذي؟
- كم أرباح شركة النور في الربع الثاني؟
- ما هو منتج النور برو؟
- أين يقع مقر شركة النور؟
- ماهو الاعلان الذي اعلنته شركة النور مؤخراً؟

## Technologies Used

- **Google Gemini API**: For embeddings and text generation
- **ChromaDB**: Vector database for document storage
- **Streamlit**: Web UI framework
- **Python**: Core programming language
