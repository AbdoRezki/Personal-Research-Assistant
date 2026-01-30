# RAG Research Assistant - FREE LOCAL VERSION 🆓

A completely FREE Python-based RAG system that runs locally using Ollama

## What Changed?

✅ **Removed**: Anthropic API (paid service)  
✅ **Added**: Ollama (100% free local LLMs)  
✅ **Result**: Unlimited queries, zero cost, complete privacy!

## Prerequisites

### 1. Install Ollama

Ollama lets you run powerful LLMs locally on your computer.

**Installation:**
- **Mac**: Download from https://ollama.ai or `brew install ollama`
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows**: Download from https://ollama.ai

**Start Ollama:**
```bash
ollama serve
```
(Leave this running in a terminal)

### 2. Download a Model

Choose a model based on your hardware:

```bash
# Recommended: Fast and capable (4GB RAM)
ollama pull llama3.2

# Alternative options:
ollama pull mistral        # 7GB - Good balance
ollama pull gemma2:2b      # 2GB - Lightweight
ollama pull llama3.1:8b    # 8GB - More capable
ollama pull qwen2.5:7b     # 7GB - Great for technical content
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Check Your Setup
```bash
python rag_assistant.py check
```
This will show you:
- ✅ If Ollama is running
- 📦 Which models you have installed

### 2. Add Your Papers
```bash
python rag_assistant.py add paper1.pdf paper2.pdf
```

### 3. Ask Questions
```bash
# Single query
python rag_assistant.py query "What are the main findings?"

# Interactive chat
python rag_assistant.py interactive
```

## Usage Examples

### Check Installation
```bash
python rag_assistant.py check
```

### Add Papers
```bash
# Add single paper
python rag_assistant.py add my_paper.pdf

# Add multiple papers
python rag_assistant.py add paper1.pdf paper2.pdf paper3.pdf

# Add all PDFs in a folder
python rag_assistant.py add ./papers/*.pdf
```

### Query Your Papers
```bash
# Basic query (uses llama3.2 by default)
python rag_assistant.py query "What datasets were used?"

# Use a different model
python rag_assistant.py query "What are the results?" --model mistral

# Get more context
python rag_assistant.py query "Your question" --top-k 10
```

### Interactive Mode
```bash
# Start chat with default model
python rag_assistant.py interactive

# Use a specific model
python rag_assistant.py interactive --model gemma2:2b
```

## Model Recommendations

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **llama3.2** | 3GB | Fast | Good | General use (recommended) |
| **gemma2:2b** | 2GB | Very Fast | Decent | Low-end hardware |
| **mistral** | 7GB | Medium | Great | Detailed analysis |
| **llama3.1:8b** | 8GB | Slow | Excellent | Complex reasoning |
| **qwen2.5:7b** | 7GB | Medium | Great | Technical papers |

**Hardware Requirements:**
- **Minimum**: 8GB RAM (use gemma2:2b)
- **Recommended**: 16GB RAM (use llama3.2 or mistral)
- **Optimal**: 32GB+ RAM (use llama3.1:8b)

## Complete Workflow Example

```bash
# 1. Check setup
python rag_assistant.py check

# 2. Download a model (if needed)
ollama pull llama3.2

# 3. Add your research papers
python rag_assistant.py add ./research_papers/*.pdf

# 4. Start asking questions
python rag_assistant.py interactive

# Example questions:
# - "What are the main contributions of these papers?"
# - "How do the methodologies differ?"
# - "What datasets were used for evaluation?"
# - "What are the key limitations identified?"
```

## Performance Comparison

### Ollama (Local - FREE) vs Anthropic API (Paid)

| Aspect | Ollama (Free) | Anthropic API |
|--------|---------------|---------------|
| **Cost** | $0 forever | $0.01-$0.05/query |
| **Privacy** | 100% local | Data sent to cloud |
| **Speed** | 5-30 sec/query* | 1-3 sec/query |
| **Quality** | Very good | Excellent |
| **Setup** | 5 minutes | 1 minute |
| **Internet** | Not required | Required |

*Speed depends on your hardware and model size

