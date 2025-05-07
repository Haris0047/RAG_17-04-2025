from qdrant_client import QdrantClient

client = QdrantClient(url="http://74.208.122.216:6333")

try:
    # Any quick read-only call works—e.g. list collections
    client.get_collections()
    print("✅ Connected to Qdrant")
except Exception as e:
    print("❌ Connection failed:", e)



# 74.208.122.216:6333