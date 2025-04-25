from transformers import pipeline
from colbert import Searcher
from colbert.infra import Run, RunConfig  # ✅ Added RunConfig

def rag_with_colbert(query, top_k=5):
    with Run().context(RunConfig(nranks=1)):  # ✅ Pass RunConfig explicitly
        searcher = Searcher(index="my_index", checkpoint="colbert-ir/colbertv2.0")
        results = searcher.search(query, k=top_k)
        docs = [searcher.collection[doc_id] for doc_id, _ in results[0]]

    context = "\n".join(docs)

    rag_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = f"Given the context below, answer the question:\n\nContext:\n{context}\n\nQuestion:\n{query}"

    response = rag_generator(prompt, max_new_tokens=150)[0]['generated_text']
    return response


if __name__ == "__main__":
    question = "What are the health benefits of green tea?"
    answer = rag_with_colbert(question)
    print("\nAnswer:", answer)
