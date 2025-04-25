"""
RAG Pipeline with ColBERT Retrieval Model

This script implements a complete Retrieval-Augmented Generation (RAG) pipeline 
using ColBERT as the retrieval model, with sample documents and generation capabilities.
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from pycolbert.modeling.colbert import ColBERT
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import faiss
from tqdm import tqdm
import argparse

def load_sample_documents():
    """Load sample documents from CNN/DailyMail dataset"""
    print("Loading sample documents...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:50]")
    print(f"Loaded {len(dataset)} documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
    )
    
    documents = []
    for doc in dataset:
        chunks = text_splitter.split_text(doc['article'])
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{doc['id']}_{i}", 
                "text": chunk,
                "source": "cnn_dailymail"
            })
    
    print(f"Created {len(documents)} document chunks")
    return documents

def setup_colbert_model():
    """Set up and load the ColBERT model"""
    print("Setting up ColBERT model...")
    model_path = "colbert-ir/colbertv2.0"  # Pre-trained ColBERT model
    
    colbert_model = ColBERT.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set model to evaluation mode
    colbert_model.eval()
    if torch.cuda.is_available():
        colbert_model = colbert_model.cuda()
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
        
    return colbert_model, tokenizer

def get_colbert_embeddings(texts, model, tokenizer, batch_size=8):
    """Create embeddings using ColBERT"""
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                                      max_length=512, return_tensors="pt")
            
            if torch.cuda.is_available():
                encoded_inputs = {k: v.cuda() for k, v in encoded_inputs.items()}
                
            # Get ColBERT embeddings
            outputs = model.encode_passage(encoded_inputs)  # [batch_size, seq_len, dim]
            
            # For simplicity, average token embeddings
            # Note: A full ColBERT implementation would keep token-level representations
            doc_embeddings = outputs.mean(dim=1)  # [batch_size, dim]
            
            embeddings.append(doc_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)

def build_vector_index(documents, colbert_model, tokenizer):
    """Build FAISS vector index from document embeddings"""
    print("Creating document embeddings...")
    doc_texts = [doc["text"] for doc in documents]
    doc_embeddings = get_colbert_embeddings(doc_texts, colbert_model, tokenizer)
    
    print("Building FAISS index...")
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(doc_embeddings)
    print(f"Created FAISS index with {index.ntotal} vectors of dimension {dimension}")
    
    return index, doc_embeddings

def retrieve_documents(query, index, documents, colbert_model, tokenizer, top_k=5):
    """Retrieve most relevant documents for a query"""
    # Get query embedding
    query_embedding = get_colbert_embeddings([query], colbert_model, tokenizer)
    
    # Search in the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve documents
    retrieved_docs = []
    for i, idx in enumerate(indices[0]):
        retrieved_docs.append({
            "id": documents[idx]["id"],
            "text": documents[idx]["text"],
            "score": float(distances[0][i])
        })
    
    return retrieved_docs

def setup_llm():
    """Set up language model for generation"""
    # NOTE: You need to set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        
        # Create prompt template
        rag_prompt_template = """
        You are an assistant that answers questions based on provided context information.

        Context information:
        {context}

        Question: {question}

        Provide a helpful, accurate, and concise answer based on the context information provided.
        If the context doesn't contain the necessary information, state that you don't have enough information.
        """
        
        rag_prompt = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
        rag_chain = LLMChain(llm=llm, prompt=rag_prompt)
        
        return rag_chain
    except:
        print("Warning: OpenAI API key not set. LLM generation will not be available.")
        return None

def rag_pipeline(query, index, documents, colbert_model, tokenizer, rag_chain=None, top_k=3):
    """Complete RAG pipeline"""
    # Retrieve documents
    retrieved_docs = retrieve_documents(
        query, index, documents, colbert_model, tokenizer, top_k=top_k
    )
    
    # Format context
    context_str = "\n\n".join([f"Document {i+1}: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
    
    result = {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "context": context_str
    }
    
    # Generate response if LLM is available
    if rag_chain:
        try:
            response = rag_chain.run(context=context_str, question=query)
            result["response"] = response
        except Exception as e:
            print(f"Error generating response: {e}")
    
    return result

def main():
    """Main function to run the RAG pipeline"""
    parser = argparse.ArgumentParser(description='RAG Pipeline with ColBERT')
    parser.add_argument('--query', type=str, default="What are the main topics in these documents?",
                        help='Query to run through the RAG pipeline')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of documents to retrieve')
    parser.add_argument('--use_llm', action='store_true',
                        help='Use LLM for generation (requires OpenAI API key)')
    args = parser.parse_args()
    
    # Load documents
    documents = load_sample_documents()
    
    # Setup ColBERT
    colbert_model, tokenizer = setup_colbert_model()
    
    # Build index
    index, _ = build_vector_index(documents, colbert_model, tokenizer)
    
    # Setup LLM if requested
    rag_chain = None
    if args.use_llm:
        rag_chain = setup_llm()
    
    # Run query
    print(f"\nQuery: {args.query}")
    result = rag_pipeline(args.query, index, documents, colbert_model, tokenizer, rag_chain, top_k=args.top_k)
    
    # Show results
    print("\nRetrieved Documents:")
    for i, doc in enumerate(result['retrieved_documents']):
        print(f"\nDocument {i+1} (Score: {doc['score']:.4f}):")
        print(doc['text'][:200] + "...")
    
    if 'response' in result:
        print(f"\nGenerated Response:\n{result['response']}")
    else:
        print("\nNo response generated. To generate responses, set --use_llm flag and configure OpenAI API key.")

if __name__ == "__main__":
    main() 