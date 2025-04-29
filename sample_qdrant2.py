# from qdrant_client import QdrantClient

# client = QdrantClient(url="http://localhost:6333")

# # Mark "date" as a datetime payload for efficient range filtering
# client.create_payload_index(
#     collection_name="test_collection",
#     field_name="date",
#     field_schema="datetime",    # or use PayloadIndexParams(type="datetime")
#     wait=True
# )






# from qdrant_client import QdrantClient
# from qdrant_client.models import Filter, FieldCondition
# from qdrant_client.http.models import DatetimeRange  # for datetime filtering

# client = QdrantClient(url="http://localhost:6333")

# # 1️⃣ Build your datetime range filter:
# date_filter = Filter(
#     must=[
#         FieldCondition(
#             key="date",
#             range=DatetimeRange(
#                 gte="2025-01-01T00:00:00Z",
#                 lte="2025-02-01T23:59:59Z"
#             )
#         )
#     ]
# )

# # 2️⃣ Call `search` with the correct argument names:
# results = client.search(
#     collection_name="test_collection",
#     query_vector=[0.1, 0.2, 0.3, 0.4],
#     query_filter=date_filter,   # ← Not `filter`
#     limit=10,                   # ← Not `top`
#     with_payload=True,
#     with_vectors=False,
# )

# # 3️⃣ Inspect results:
# for pt in results:
#     print(pt.id, pt.payload["date"], pt.payload["content"])


#!/usr/bin/env python3
import os
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from langchain.embeddings import OpenAIEmbeddings

# ── Configuration ───────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Collections to search
COLLECTIONS = {
    "filings": "sec_filings_chunks",
    "earnings": "earnings_chunks",
    "news":    "news_articles",
}

# Number of results to return per collection
TOP_K = 5

# ── Setup Qdrant & Embeddings ───────────────────────────────────────────────────
qdrant = QdrantClient(url=QDRANT_URL)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ── Helper: embed + search a single collection ─────────────────────────────────
def search_collection(collection_name: str, query: str, k: int = TOP_K, date_from=None, date_to=None):
    # 1️⃣ Embed the query text
    query_vector = embeddings.embed_query(query)
    
    # 2️⃣ (Optional) Build a date filter if provided
    qdrant_filter = None
    if date_from or date_to:
        conds = []
        if date_from:
            conds.append(FieldCondition(
                key="date",
                range=Range(gte=date_from)
            ))
        if date_to:
            conds.append(FieldCondition(
                key="date",
                range=Range(lte=date_to)
            ))
        qdrant_filter = Filter(must=conds)

    # 3️⃣ Execute the search
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    return hits

# ── Main CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic search in Qdrant with OpenAI embeddings")
    parser.add_argument("query", help="Text query to search for")
    parser.add_argument("--date-from", help="ISO date lower bound (YYYY-MM-DD or ISO‐8601)")
    parser.add_argument("--date-to",   help="ISO date upper bound (YYYY-MM-DD or ISO‐8601)")
    parser.add_argument("--collections", nargs="+", choices=COLLECTIONS.keys(),
                        default=list(COLLECTIONS.keys()),
                        help="Which collections to query (default: all)")
    args = parser.parse_args()

    # Normalize date strings (keep as ISO‐8601)
    date_from = args.date_from
    date_to   = args.date_to

    print(f"\n🔍 Searching for: “{args.query}”")
    if date_from or date_to:
        print(f"   Date range: {date_from or '-∞'} → {date_to or '∞'}\n")

    # Loop over requested collections
    for key in args.collections:
        coll = COLLECTIONS[key]
        print(f"▶️  Collection: {key} ({coll})")
        try:
            results = search_collection(coll, args.query, TOP_K, date_from, date_to)
            if not results:
                print("   (no results)\n")
                continue

            for pt in results:
                dt = pt.payload.get("date")
                # If stored as timestamp, convert; if as ISO string, leave it
                ts = ""
                try:
                    # If float
                    dt = float(dt)
                    ts = datetime.utcfromtimestamp(dt).isoformat() + "Z"
                except Exception:
                    ts = dt
                print(f" • ID: {pt.id}, Score: {pt.score:.4f}, Date: {ts}")
                print(f"   → {pt.payload.get('content')[:200]}…\n")
        except Exception as e:
            print(f"   ERROR: {e}\n")
