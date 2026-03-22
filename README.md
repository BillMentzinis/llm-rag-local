# LLM RAG Local

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A local Retrieval-Augmented Generation (RAG) chat application powered by **Llama 3.1 8B Instruct** with a Streamlit UI. Runs entirely on your machine — no cloud API keys required.

## Quick Start

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate llm-local

# 2. Launch the app
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. On first run it will download the embedding model (~90 MB).

> **Note:** You need a Hugging Face account with access to `meta-llama/Llama-3.1-8B-Instruct`. Log in with `huggingface-cli login` before starting.

---

## Features

- Chat interface with Llama 3.1 8B Instruct (4-bit quantized, ~4 GB VRAM)
- RAG support: upload documents and get cited, grounded answers
- Supported file types: PDF, TXT, MD, and common code files
- Document management: upload, index, delete, view stats
- Persistent vector storage with ChromaDB
- Configurable generation and retrieval parameters

## Architecture

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| LLM | Llama 3.1 8B Instruct (4-bit via bitsandbytes) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| PDF parsing | PyMuPDF |

## File Structure

```
LLM/
├── streamlit_app.py        # Main Streamlit UI
├── rag_pipeline.py         # RAG orchestration
├── document_processor.py   # File parsing and chunking
├── vector_store_manager.py # ChromaDB operations
├── config.py               # All configuration parameters
├── environment.yml         # Conda environment
└── chroma_db/              # Vector database (auto-created, gitignored)
```

## Usage

### Basic Chat (no documents)

1. Start the app
2. Disable "Enable RAG" in the sidebar
3. Chat normally using the model's base knowledge

### RAG-Enhanced Chat

1. **Upload documents** — click "Browse files" in the sidebar, select files, then "Process Documents"
2. **Enable RAG** — ensure "Enable RAG" is checked
3. **Ask questions** — the model retrieves relevant chunks and cites sources

### Adjusting Settings

| Setting | Description |
|---------|-------------|
| Context Chunks | Number of document chunks to retrieve (1–10) |
| Temperature | Response randomness (0.1 = focused, 2.0 = creative) |
| Max Tokens | Maximum response length |

## Configuration

All parameters are in `config.py`:

- **RAG**: `chunk_size` (512 tokens), `chunk_overlap` (50), `top_k` (3), `min_similarity` (0.5)
- **Generation**: `temperature` (0.7), `top_p` (0.9), `max_new_tokens` (512)

## Supported File Types

**Documents:** PDF, TXT, MD
**Code:** Python, JavaScript, Java, C/C++, C#, Go, Rust, Ruby

## Performance

- **VRAM:** ~4 GB (model at 4-bit + embeddings)
- **First run:** Slow (downloads embedding model)
- **Retrieval:** <100 ms
- **Generation:** ~20–30 tokens/second (GPU-dependent)

## Troubleshooting

**Model loads slowly** — expected on first run; subsequent loads are faster.

**Poor retrieval quality** — increase `top_k` in `config.py` or the sidebar, and ensure uploaded documents are relevant.

**Out of memory** — reduce `max_new_tokens`, close other GPU apps.

**Document processing fails** — check file format is supported and size is <10 MB.
