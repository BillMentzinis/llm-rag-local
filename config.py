"""
Configuration file for RAG-enabled Llama 3.1 8B Streamlit application.
Centralizes all configuration parameters for easy tuning.
"""

import os
import torch

# Model Configuration
MODEL_CONFIG = {
    "name": "meta-llama/Llama-3.1-8B-Instruct",
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
    },
    "device_map": "auto",
    "trust_remote_code": True,
}

# Available LLMs for the model selector
_QUANT = MODEL_CONFIG["quantization"]
AVAILABLE_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "display_name": "Llama 3.1 8B (default, ~4GB)",
        "quantization": _QUANT,
        "device_map": "auto",
        "trust_remote_code": True,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "display_name": "Llama 3.2 3B (~2GB)",
        "quantization": _QUANT,
        "device_map": "auto",
        "trust_remote_code": True,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "display_name": "Qwen 2.5 7B (~4GB)",
        "quantization": _QUANT,
        "device_map": "auto",
        "trust_remote_code": True,
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "display_name": "Phi-3.5 Mini 3.8B (~2GB)",
        "quantization": _QUANT,
        "device_map": "auto",
        "trust_remote_code": True,
    },
    "microsoft/phi-4": {
        "display_name": "Phi-4 14B (~8GB, large GPU)",
        "quantization": _QUANT,
        "device_map": "auto",
        "trust_remote_code": True,
    },
}

# RAG Configuration
RAG_CONFIG = {
    # all-MiniLM-L6-v2 only reads the first 256 tokens of its input, so a chunk
    # (overlap included) must stay under that or its tail is never searchable
    "chunk_size": 200,  # Tokens per chunk, overlap included (estimated as 4 chars/token)
    "chunk_overlap": 30,  # Overlap between chunks
    "embedding_model": "all-MiniLM-L6-v2",  # Fast, 384-dim embeddings
    "top_k": 5,  # Number of chunks to retrieve
    "context_max_tokens": 2048,  # Max tokens for RAG context
    "min_similarity": 0.3,  # Minimum cosine similarity for a chunk to be used (0-1)
}

# Generation Configuration
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

# RAG-specific generation (more factual, less creative)
RAG_GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.6,  # Lower for factual responses
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

# Supported File Types
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF Document",
    ".txt": "Text File",
    ".md": "Markdown",
    ".py": "Python",
    ".js": "JavaScript",
    ".java": "Java",
    ".cpp": "C++",
    ".c": "C",
    ".h": "C Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".rb": "Ruby",
    ".go": "Go",
    ".rs": "Rust",
}

# File Processing Configuration
FILE_CONFIG = {
    "max_file_size_mb": 10,
    "encoding": "utf-8",
    "encoding_fallback": "latin-1",
}

# Path Configuration
PATHS = {
    "chroma_db": os.path.join(os.path.dirname(__file__), "chroma_db"),
    "chats": os.path.join(os.path.dirname(__file__), "chats"),
}

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant. You have access to relevant information from uploaded documents:

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---

Use this information to answer the question. If the context is relevant, cite the source in your response (e.g., "According to document.pdf..."). If the context doesn't contain relevant information, you may use your general knowledge but indicate this clearly.

Question: {query}"""

# System Prompt for Non-RAG Mode
SYSTEM_PROMPT = """You are a helpful AI assistant. Provide clear, accurate, and concise responses to user questions."""

# Streamlit UI Configuration
UI_CONFIG = {
    "page_title": "Local LLM with RAG",
    "page_icon": "🦙",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Token Budget Configuration (for context window management)
TOKEN_BUDGET = {
    "response": 1024,  # Reserved for model generation
    "history": 512,  # Reserved for conversation history
    "rag_context": 2048,  # Reserved for RAG context
    "system_prompt": 256,  # Reserved for system prompt
}
