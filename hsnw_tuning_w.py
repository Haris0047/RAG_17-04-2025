import itertools
import time
from statistics import quantiles

import weaviate                     # weaviate-client v4
from dotenv import load_dotenv
import os

from weaviate.collections.classes.config import Reconfigure
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings

# 1. Load API key and embeddings
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 2. Connect to local Weaviate
client = weaviate.connect_to_local()  # returns a WeaviateClient :contentReference[oaicite:6]{index=6}

# 3. Instantiate LangChain stores (for semantic search)
vector_store_fillings = WeaviateVectorStore(
    client=client,
    index_name="Fillings",
    text_key="content",
    embedding=embeddings,
    attributes=["ticker", "date", "filling_type"]
)
vector_store_earnings = WeaviateVectorStore(
    client=client,
    index_name="Earnings",
    text_key="content",
    embedding=embeddings,
    attributes=["ticker", "date", "quarter"]
)
vector_store_news = WeaviateVectorStore(
    client=client,
    index_name="News",
    text_key="content",
    embedding=embeddings,
    attributes=["ticker", "date", "url", "title", "publisher", "site"]
)

classes = ["Fillings", "Earnings", "News"]
stores = [vector_store_fillings, vector_store_earnings, vector_store_news]

# 4. Parameter grids
max_conn_values = [8, 16, 32]       # Immutable—shown for reference only
efc_values      = [100, 200, 400]   # Immutable—shown for reference only
ef_search_vals  = [100, 200, 400]   # Query-time `ef` (mutable per-search)

test_queries = [
    "Summarize Apple Q1 2025 earnings highlights",
    "What risks did Apple mention in their latest filing?",
    "Key news about AAPL from the past week",
]

def update_dynamic_hnsw(class_name, dynamic_ef_min, dynamic_ef_max, dynamic_ef_factor):
    """
    Update only the mutable HNSW settings on an existing collection.
    """
    # Build the mutable config object
    hnsw_update = Reconfigure.VectorIndex.hnsw(
        dynamic_ef_min=dynamic_ef_min,
        dynamic_ef_max=dynamic_ef_max,
        dynamic_ef_factor=dynamic_ef_factor,
        # You can also pass ef=..., flat_search_cutoff=..., vector_cache_max_objects=...
    )
    # Retrieve the collection and apply the update
    collection = client.collections.get(class_name)         # v4 collections API :contentReference[oaicite:7]{index=7}
    collection.config.update(vector_index_config=hnsw_update)  # apply mutable config :contentReference[oaicite:8]{index=8}

def time_search(store, ef):
    """
    Run each test query and measure median (p50) and 95th‐pct latency.
    """
    latencies = []
    for q in test_queries:
        start = time.perf_counter()
        _ = store.similarity_search(q, k=5, search_kwargs={"ef": ef})
        latencies.append(time.perf_counter() - start)
    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p95 = quantiles(latencies, n=100)[94]
    return p50, p95

# 5. Benchmark loop
results = []
for cls, store in zip(classes, stores):
    print(f"\nTuning «{cls}»")
    # Only dynamic_ef parameters are mutable—graph `M` and build `efConstruction` cannot be changed on-the-fly :contentReference[oaicite:9]{index=9}
    for dynamic_min, dynamic_max, dynamic_factor in [(100, 400, 8), (200, 800, 16), (50, 200, 4)]:
        update_dynamic_hnsw(cls, dynamic_min, dynamic_max, dynamic_factor)
        time.sleep(1)  # allow config propagation
        for ef in ef_search_vals:
            p50, p95 = time_search(store, ef)
            results.append({
                "class": cls,
                "dynamicEfMin": dynamic_min,
                "dynamicEfMax": dynamic_max,
                "dynamicEfFactor": dynamic_factor,
                "ef": ef,
                "p50_s": p50,
                "p95_s": p95
            })
            print(
                f" dynamicEfMin={dynamic_min}, dynamicEfMax={dynamic_max}, "
                f"dynamicEfFactor={dynamic_factor}, ef={ef} → "
                f"p50={p50:.3f}s, p95={p95:.3f}s"
            )

# 6. Summarize best settings per class by p95 latency
best_by_class = {}
for row in results:
    cls = row["class"]
    if cls not in best_by_class or row["p95_s"] < best_by_class[cls]["p95_s"]:
        best_by_class[cls] = row

print("\n→ Best dynamic‐HNSW settings per class (by p95 latency):")
for cls, cfg in best_by_class.items():
    print(
        f" • {cls}: dynamicEfMin={cfg['dynamicEfMin']}, "
        f"dynamicEfMax={cfg['dynamicEfMax']}, dynamicEfFactor={cfg['dynamicEfFactor']}, "
        f"ef={cfg['ef']}, p95={cfg['p95_s']:.3f}s"
    )

client.close()
