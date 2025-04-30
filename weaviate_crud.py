#!/usr/bin/env python3
import sys
import json
import weaviate
from pprint import pprint

# ─── Configuration ─────────────────────────────────────────────────────────────

# Change this if your Weaviate is on a different host/port
WEAVIATE_URL = "http://localhost:8080"

# ─── Helper: Connect ───────────────────────────────────────────────────────────

def get_client():
    """Connect to local Weaviate (Docker) and verify readiness."""
    client = weaviate.connect_to_local()
    if not client.is_ready():
        print("❌ Cannot connect to Weaviate at", WEAVIATE_URL, file=sys.stderr)
        sys.exit(1)
    return client

# ─── CRUD Operations ───────────────────────────────────────────────────────────

def list_collections(client):
    """Return a list of collection (class) names."""
    return client.collections.list_all(simple=True)

def describe_collection(client, name):
    """Fetch the full schema/config of a given collection."""
    coll = client.collections.get(name)
    return coll.config.get()

def create_collection(client, name, schema: dict = None):
    """
    Create a collection.
    - If `schema` is None, we create an empty collection with auto-schema.
    - Else, `schema` must be a dict matching Weaviate's class schema JSON.
    """
    if schema:
        # expects {"class": name, "properties": [...], ...}
        client.collections.create_from_dict(schema)
    else:
        client.collections.create(name)
    print(f"✅ Created collection: {name}")

def delete_collection(client, name):
    """Drop a collection and all its objects."""
    client.collections.delete(name)
    print(f"🗑️  Deleted collection: {name}")

def flush_collection(client, name):
    """
    Delete **all** objects in a collection but keep its schema.
    """
    client.collections.get(name).data.delete_all()
    print(f"🚿 Flushed all objects in: {name}")

# ─── Interactive CLI ──────────────────────────────────────────────────────────

def main():
    client = get_client()

    menu = {
        "1": ("List collections", lambda: print(list_collections(client))),
        "2": ("Describe a collection", 
              lambda: pprint(describe_collection(client, input("Name: ")))),
        "3": ("Create a collection (auto-schema)", 
              lambda: create_collection(client, input("Name: "))),
        "4": ("Create with custom schema JSON",
              lambda: create_collection(
                  client, 
                  json.loads(input("Full JSON schema dict:\n")))
              ),
        "5": ("Delete a collection", 
              lambda: delete_collection(client, input("Name: "))),
        "6": ("Flush a collection (delete all objects)", 
              lambda: flush_collection(client, input("Name: "))),
        "q": ("Quit", sys.exit)
    }

    while True:
        print("\nWeaviate Manager — Select an option:")
        for key, (desc, _) in menu.items():
            print(f"  [{key}] {desc}")
        choice = input("> ").strip()
        action = menu.get(choice)
        if not action:
            print("❓ Invalid choice, try again.")
            continue
        # run the handler
        action[1]()

if __name__ == "__main__":
    main()
