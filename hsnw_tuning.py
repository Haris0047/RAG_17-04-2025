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

# # 1️⃣ Define the HNSW grid and EF values to test
# m_values          = [8, 16, 32]
# ef_construct_vals = [100, 200, 400]
# ef_search_vals    = [100, 200, 400]

# # 2️⃣ A small set of representative queries
# test_queries = [
#     "Summarize Apple Q1 2025 earnings highlights",
#     "What risks did Apple mention in their latest filing?",
#     "Key news about AAPL from the past week",
#     # …add more as needed…
# ]

def update_hnsw(store, m, efc):
    """Update an existing collection's HNSW parameters in-place."""
    client = store.client
    client.update_collection(
        collection_name=store.collection_name,
        hnsw_config=HnswConfigDiff(m=m, ef_construct=efc)
    )

# def time_search(store, ef):
#     """
#     Run each test query through the raw QdrantClient.search
#     with a given hnsw_ef, return (p50, p95) in seconds.
#     """
#     client = store.client
#     name   = store.collection_name
#     latencies = []
#     for q in test_queries:
#         vec = store.embeddings.embed_query(q)
#         start = time.perf_counter()
#         client.search(
#             collection_name=name,
#             query_vector=vec,
#             limit=5,
#             search_params=SearchParams(hnsw_ef=ef)
#         )
#         latencies.append(time.perf_counter() - start)
#     latencies.sort()
#     return (
#         latencies[len(latencies)//2],
#         quantiles(latencies, n=100)[94]
#     )

# # 3️⃣ Run the grid search
# results = []

# for store in stores:
#     print(f"\nTuning «{store.collection_name}»")
#     for m, efc in itertools.product(m_values, ef_construct_vals):
#         update_hnsw(store, m, efc)
#         # give Qdrant a moment to apply new config
#         time.sleep(1)

#         for ef in ef_search_vals:
#             p50, p95 = time_search(store, ef)
#             results.append({
#                 "store": store.collection_name,
#                 "m": m,
#                 "ef_construct": efc,
#                 "ef": ef,
#                 "p50_s": p50,
#                 "p95_s": p95
#             })
#             print(f" m={m}, efc={efc}, ef={ef} → p50={p50:.3f}s, p95={p95:.3f}s")

# # 4️⃣ Summarize top performers per store
# from collections import defaultdict

# # Initialize an empty dict (no default), so we can test membership
# best_by_store = {}

# for r in results:
#     store = r["store"]
#     # If we've never seen this store, or found a better p95_s, update:
#     if (store not in best_by_store 
#         or r["p95_s"] < best_by_store[store]["p95_s"]):
#         best_by_store[store] = r

# print("\n→ Best settings per collection (by p95 latency):")
# for store, cfg in best_by_store.items():
#     print(
#         f" • {store}: m={cfg['m']}, "
#         f"ef_construct={cfg['ef_construct']}, ef={cfg['ef']}, "
#         f"p95={cfg['p95_s']:.3f}s"
#     )
# update_hnsw(store, m, efc)