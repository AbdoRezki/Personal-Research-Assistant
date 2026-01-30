#!/usr/bin/env python3
"""
RAG Research Assistant for Academic Papers (FREE LOCAL VERSION)
Uses Ollama for free local LLM inference - no API costs!
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import requests

# Document processing
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# For better text chunking
import re


class DocumentProcessor:
    """Handles PDF extraction and text chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for delimiter in ['. ', '.\n', '! ', '? ']:
                    last_delim = text.rfind(delimiter, start, end)
                    if last_delim != -1:
                        end = last_delim + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = {
                    'text': chunk_text,
                    'metadata': metadata or {},
                    'start_char': start,
                    'end_char': end
                }
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a PDF file into chunks with metadata."""
        text = self.extract_text_from_pdf(pdf_path)
        
        metadata = {
            'source': pdf_path,
            'filename': os.path.basename(pdf_path)
        }
        
        chunks = self.chunk_text(text, metadata)
        print(f"Processed {pdf_path}: {len(chunks)} chunks created")
        
        return chunks


class VectorStore:
    """Manages document embeddings and similarity search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with a sentence transformer model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks and create embeddings."""
        if not chunks:
            return
        
        print(f"Creating embeddings for {len(chunks)} chunks...")
        texts = [chunk['text'] for chunk in chunks]
        new_embeddings = self.model.encode(texts, show_progress_bar=True)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.chunks.extend(chunks)
        print(f"Total chunks in store: {len(self.chunks)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for most relevant chunks."""
        if not self.chunks:
            return []
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.chunks[i], float(similarities[i]))
            for i in top_indices
        ]
        
        return results
    
    def save(self, directory: str):
        """Save the vector store to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save chunks
        with open(os.path.join(directory, 'chunks.json'), 'w') as f:
            json.dump(self.chunks, f, indent=2)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(os.path.join(directory, 'embeddings.npy'), self.embeddings)
        
        print(f"Vector store saved to {directory}")
    
    def load(self, directory: str):
        """Load the vector store from disk."""
        chunks_path = os.path.join(directory, 'chunks.json')
        embeddings_path = os.path.join(directory, 'embeddings.npy')
        
        if os.path.exists(chunks_path):
            with open(chunks_path, 'r') as f:
                self.chunks = json.load(f)
        
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        
        print(f"Loaded {len(self.chunks)} chunks from {directory}")


class OllamaClient:
    """Client for interacting with Ollama local LLM."""
    
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/generate"
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    def generate(self, prompt: str, system: str = None) -> str:
        """Generate response from Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Ollama returned status {response.status_code}"
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be too large or slow."
        except Exception as e:
            return f"Error: {str(e)}"


class RAGAssistant:
    """Main RAG assistant that combines retrieval and generation."""
    
    def __init__(self, vector_store: VectorStore, model: str = "llama3.2"):
        self.vector_store = vector_store
        self.llm = OllamaClient(model=model)
        
        # Check if Ollama is running
        if not self.llm.is_available():
            print("\n⚠️  WARNING: Ollama is not running!")
            print("Please start Ollama first:")
            print("  1. Install from: https://ollama.ai")
            print("  2. Run: ollama serve")
            print("  3. Pull a model: ollama pull llama3.2")
            print()
    
    def query(self, question: str, top_k: int = 5, show_sources: bool = True) -> str:
        """Query the research assistant."""
        # Check Ollama availability
        if not self.llm.is_available():
            return "Error: Ollama is not running. Please start it with 'ollama serve'"
        
        # Retrieve relevant chunks
        results = self.vector_store.search(question, top_k=top_k)
        
        if not results:
            return "No documents found in the knowledge base. Please add some papers first."
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(f"[Document {i}]\n{chunk['text']}\n")
            sources.append({
                'filename': chunk['metadata'].get('filename', 'Unknown'),
                'score': score
            })
        
        context = "\n".join(context_parts)
        
        # Create prompt for LLM
        system_prompt = """You are a helpful research assistant. Answer questions based on the provided academic paper excerpts. 

If the answer cannot be found in the excerpts, say so. When referencing information, mention which document number it came from.

Be concise but comprehensive in your answers."""
        
        user_prompt = f"""Question: {question}

Relevant excerpts from papers:
{context}

Please provide a comprehensive answer based on these excerpts."""
        
        # Query local LLM
        print("Generating answer with local LLM...")
        answer = self.llm.generate(user_prompt, system=system_prompt)
        
        # Add sources if requested
        if show_sources and answer:
            answer += "\n\n--- Sources ---\n"
            for i, source in enumerate(sources, 1):
                answer += f"[{i}] {source['filename']} (relevance: {source['score']:.3f})\n"
        
        return answer


def main():
    parser = argparse.ArgumentParser(
        description='RAG Research Assistant for Academic Papers (FREE LOCAL VERSION)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Check Ollama command
    check_parser = subparsers.add_parser('check', help='Check Ollama installation and models')
    
    # Add documents command
    add_parser = subparsers.add_parser('add', help='Add PDF documents to the knowledge base')
    add_parser.add_argument('files', nargs='+', help='PDF files to add')
    add_parser.add_argument('--store-dir', default='./vector_store', 
                           help='Directory to save vector store')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the research assistant')
    query_parser.add_argument('question', help='Your research question')
    query_parser.add_argument('--store-dir', default='./vector_store',
                             help='Directory with vector store')
    query_parser.add_argument('--top-k', type=int, default=5,
                             help='Number of relevant chunks to retrieve')
    query_parser.add_argument('--model', default='llama3.2',
                             help='Ollama model to use (e.g., llama3.2, mistral, gemma2)')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', 
                                              help='Start interactive Q&A session')
    interactive_parser.add_argument('--store-dir', default='./vector_store',
                                   help='Directory with vector store')
    interactive_parser.add_argument('--model', default='llama3.2',
                                   help='Ollama model to use')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        # Check Ollama installation
        print("Checking Ollama installation...\n")
        
        client = OllamaClient()
        
        if client.is_available():
            print("✅ Ollama is running!")
            
            models = client.list_models()
            if models:
                print(f"\n📦 Available models ({len(models)}):")
                for model in models:
                    print(f"  - {model}")
                print("\nYou're ready to use the RAG assistant!")
            else:
                print("\n⚠️  No models installed.")
                print("Install a model with:")
                print("  ollama pull llama3.2")
        else:
            print("❌ Ollama is not running.")
            print("\nTo install and start Ollama:")
            print("  1. Download from: https://ollama.ai")
            print("  2. Install and run: ollama serve")
            print("  3. Pull a model: ollama pull llama3.2")
    
    elif args.command == 'add':
        # Process and add documents
        processor = DocumentProcessor()
        vector_store = VectorStore()
        
        # Load existing store if it exists
        if os.path.exists(args.store_dir):
            vector_store.load(args.store_dir)
        
        # Process each PDF
        for pdf_file in args.files:
            if not os.path.exists(pdf_file):
                print(f"Warning: {pdf_file} not found, skipping...")
                continue
            
            chunks = processor.process_pdf(pdf_file)
            vector_store.add_documents(chunks)
        
        # Save the updated store
        vector_store.save(args.store_dir)
        print(f"\n✅ Successfully added {len(args.files)} document(s)")
    
    elif args.command == 'query':
        # Load vector store
        vector_store = VectorStore()
        
        if not os.path.exists(args.store_dir):
            print(f"Error: Vector store not found at {args.store_dir}")
            print("Please add some documents first using: python rag_assistant_free.py add <pdf_files>")
            return
        
        vector_store.load(args.store_dir)
        
        # Create assistant
        assistant = RAGAssistant(vector_store, model=args.model)
        
        # Query
        print(f"\n❓ Question: {args.question}\n")
        answer = assistant.query(args.question, top_k=args.top_k)
        print(answer)
    
    elif args.command == 'interactive':
        # Load vector store
        vector_store = VectorStore()
        
        if not os.path.exists(args.store_dir):
            print(f"Error: Vector store not found at {args.store_dir}")
            print("Please add some documents first using: python rag_assistant_free.py add <pdf_files>")
            return
        
        vector_store.load(args.store_dir)
        
        # Create assistant
        assistant = RAGAssistant(vector_store, model=args.model)
        
        print("\n" + "=" * 70)
        print("🆓 RAG Research Assistant - Interactive Mode (FREE LOCAL)")
        print(f"Using model: {args.model}")
        print("=" * 70)
        print("Ask questions about your academic papers. Type 'quit' to exit.\n")
        
        while True:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🤔 Thinking...\n")
            answer = assistant.query(question)
            print(answer)
            print("\n" + "="*70 + "\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()