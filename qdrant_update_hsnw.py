import itertools
import time
from statistics import mean, quantiles
from qdrant_client.models import HnswConfigDiff, SearchParams
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(url="http://localhost:6333")

vector_store_fillings = QdrantVectorStore(
    client=qdrant_client,
    collection_name="fillings",
    embeddings=embedding_model,
    content_payload_key="content",    # for content lookup
)

vector_store_earnings = QdrantVectorStore(
    client=qdrant_client,
    collection_name="earnings",
    embeddings=embedding_model,
    content_payload_key="content",
)

vector_store_news = QdrantVectorStore(
    client=qdrant_client,
    collection_name="news",
    embeddings=embedding_model,
    content_payload_key="content",
)

# Your pre-loaded LangChain QdrantVectorStore instances
stores = [
    vector_store_fillings,
    vector_store_earnings,
    vector_store_news,
]


# Reusable updater
def update_hnsw(store, m, ef_construct):
    """
    Update only the HNSW graph parameters on an existing Qdrant collection.
    Other collection properties (vector size, distance metric) remain unchanged.
    """
    store.client.update_collection(
        collection_name=store.collection_name,
        hnsw_config=HnswConfigDiff(
            m=m,
            ef_construct=ef_construct
        )
    )

# 1) Apply the best settings per store:
update_hnsw(vector_store_fillings, m=8,  ef_construct=100)
update_hnsw(vector_store_earnings, m=16, ef_construct=400)
update_hnsw(vector_store_news,     m=16, ef_construct=100)

# 2) When you search, always set the matching ef:
def search_with_ef(store, prompt, ef, k=5):
    vec = store.embeddings.embed_query(prompt)
    return store.client.search(
        collection_name=store.collection_name,
        query_vector=vec,
        limit=k,
        search_params=SearchParams(hnsw_ef=ef)
    )

# Examples:
results_fillings = search_with_ef(vector_store_fillings, 
                                  "Latest AAPL risks", ef=400)
results_earnings = search_with_ef(vector_store_earnings, 
                                  "Q1 2025 highlights", ef=400)
results_news     = search_with_ef(vector_store_news,     
                                  "Apple stock news", ef=200)
