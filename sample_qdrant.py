from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
import time

# Connect to your Qdrant server
qdrant = QdrantClient(url="http://localhost:6333")

# List of all your collections
collections = ["fillings", "earnings", "news"]

def fix_dates_in_collection(collection_name):
    print(f"🛠️ Fixing collection: {collection_name}")
    updated_count = 0
    next_offset = None

    while True:
        points, next_offset = qdrant.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
            offset=next_offset,
            limit=100,  # batch size
        )

        if not points:
            break

        updates = []

        for point in points:
            payload = point.payload
            if payload and "date" in payload:
                try:
                    date_field = payload["date"]

                    # ✅ If 'date' is still a string like "2025-01-30"
                    if isinstance(date_field, str):
                        dt = datetime.strptime(date_field, "%Y-%m-%d")  # parse 'YYYY-MM-DD'
                        timestamp_value = dt.timestamp()

                        # Prepare updated payload
                        new_payload = payload.copy()
                        new_payload["date"] = timestamp_value

                        updates.append(
                            PointStruct(
                                id=point.id,
                                payload=new_payload,
                                vector=point.vector,
                            )
                        )
                except Exception as e:
                    print(f"⚠️ Error processing point {point.id}: {e}")

        # Only upsert if there are any updates
        if updates:
            qdrant.upsert(
                collection_name=collection_name,
                points=updates
            )
            updated_count += len(updates)
            print(f"✅ Updated {len(updates)} points...")

        if next_offset is None:
            break

    print(f"🎯 Finished fixing {updated_count} points in '{collection_name}'\n")


if __name__ == "__main__":
    start_time = time.time()
    for collection in collections:
        fix_dates_in_collection(collection)
    end_time = time.time()
    print(f"✅ All done! Took {round(end_time - start_time, 2)} seconds.")
