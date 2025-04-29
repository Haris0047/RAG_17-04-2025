from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

# ── Configuration ───────────────────────────────────────────────────────────────
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "news"       # change if needed
FIELDS_TO_INDEX  = [
    ("ticker",        PayloadSchemaType.KEYWORD),
    # ("quarter",       PayloadSchemaType.KEYWORD),
    # ("filling_type",  PayloadSchemaType.KEYWORD),
]

# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL)

    for field_name, schema_type in FIELDS_TO_INDEX:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_type,
                wait=True
            )
            print(f"✅ Indexed field '{field_name}' as {schema_type.name}")
        except Exception as e:
            # If already indexed, Qdrant will return an error—ignore it
            msg = str(e).lower()
            if "already exists" in msg or "acknowledged" in msg:
                print(f"ℹ️  Field '{field_name}' is already indexed.")
            else:
                raise
