from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://dce970ea-40b1-4132-84c5-56f51a0365c8.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="apikey",
)

print(qdrant_client.get_collections())