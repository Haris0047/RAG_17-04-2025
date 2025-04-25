from colbert.infra import Run
from colbert.search import Searcher

index_name = "my_index"
query = "What are the health benefits of green tea?"

with Run().context():
    searcher = Searcher(index=index_name, checkpoint="colbert-ir/colbertv2.0")
    results = searcher.search(query, k=5)

    print("\nTop Results:\n")
    for i, (doc_id, score) in enumerate(results[0]):
        print(f"[{i+1}] Score: {score:.2f}")
        print(searcher.collection[doc_id])
        print()
