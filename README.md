# RAG Pipeline with ColBERT

This repository contains an implementation of a Retrieval-Augmented Generation (RAG) pipeline using ColBERT as the retrieval model.

## Overview

The implementation includes:

1. Document loading and chunking
2. ColBERT model setup for dense retrieval
3. Document embedding generation
4. FAISS index creation for efficient similarity search
5. Query processing and document retrieval
6. LLM integration for answer generation

## What is ColBERT?

ColBERT (Contextualized Late Interaction over BERT) is a state-of-the-art neural retrieval model that efficiently leverages the expressive power of BERT-based language models. It uses a unique late interaction architecture that provides both high-quality retrieval and computational efficiency.

Key advantages of ColBERT:
- Preserves token-level representations rather than collapsing to a single vector
- Enables more fine-grained matching between query and documents
- Better context understanding through late interaction scoring

## Installation

```bash
# Install required packages
pip install transformers datasets torch pycolbert langchain langchain_openai faiss-cpu
```

## Usage

### Python Script

The main implementation is in `colbert_rag.py`. You can run it directly:

```bash
# Basic usage
python colbert_rag.py

# With a custom query
python colbert_rag.py --query "What political events are mentioned in these documents?"

# Customize the number of retrieved documents
python colbert_rag.py --top_k 5

# Enable LLM generation (requires OpenAI API key)
python colbert_rag.py --use_llm
```

To use the LLM generation feature, you need to set your OpenAI API key:

```python
# In the code or as an environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## How It Works

1. **Document Loading**: Sample documents are loaded from the CNN/DailyMail dataset and split into chunks for better retrieval
2. **Embedding Generation**: ColBERT generates contextualized embeddings for each document chunk
3. **Indexing**: Document embeddings are indexed using FAISS for efficient retrieval
4. **Retrieval**: When a query is received, ColBERT generates a query embedding and retrieves the most similar documents
5. **Generation**: Retrieved documents are used as context for an LLM to generate a relevant answer

## Customization

### Using Different Documents

Modify the `load_sample_documents` function to load your own documents:

```python
def load_custom_documents(file_path):
    # Load documents from your custom source
    # Process and chunk them
    # Return in the same format as the sample function
    documents = [...]
    return documents
```

### Using a Local LLM

You can replace the OpenAI LLM with a local model:

```python
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, pipeline

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
local_tokenizer = AutoTokenizer.from_pretrained(model_id)
local_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)

pipe = pipeline(
    "text-generation",
    model=local_model,
    tokenizer=local_tokenizer,
    max_new_tokens=256,
    temperature=0.3
)

local_llm = HuggingFacePipeline(pipeline=pipe)
```

## Advanced Implementation

Note that this implementation simplifies ColBERT by averaging token embeddings into a single vector for each document. A full-fledged ColBERT implementation would:

1. Preserve token-level embeddings for both documents and queries
2. Use MaxSim matching between query and document tokens
3. Implement the full ColBERT scoring mechanism with late interaction

## References

- [ColBERT Paper](https://arxiv.org/abs/2004.12832) - Original research paper
- [pycolbert](https://github.com/stanford-futuredata/pycolbert) - Python implementation of ColBERT 