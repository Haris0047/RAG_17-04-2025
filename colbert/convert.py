import os

# Create the folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Define your documents here
documents = [
    "Green tea is rich in antioxidants and may improve brain function.",
    "It can help with fat loss, protect against cancer, and lower the risk of heart disease.",
    "Green tea contains catechins, which are natural antioxidants.",
    "Drinking green tea regularly may reduce the risk of cardiovascular diseases.",
    "The caffeine and L-theanine in green tea can enhance brain function."
]

# Write to TSV (doc_id \t text)
with open("colbert/data/docs.tsv", "w", encoding="utf-8") as f:
    for idx, text in enumerate(documents):
        f.write(f"{idx}\t{text}\n")

print("✅ data/docs.tsv created with", len(documents), "documents.")
