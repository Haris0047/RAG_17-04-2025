from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
import os

if __name__ == '__main__':
    # Your document collection and index name
    collection_path = "data/docs.tsv"  # <-- your corpus
    index_name = "my_index"
    checkpoint_path = "colbert-ir/colbertv2.0"  # or your downloaded checkpoint
    experiment_path = "experiments"  # optional: where to store index files

    os.makedirs(experiment_path, exist_ok=True)

    # Start indexing
    with Run().context(RunConfig(nranks=1, experiment="custom_experiment")):
        config = ColBERTConfig(
            nbits=2,                    # You can increase this if you want less compression
            root=experiment_path,       # Where all indexing artifacts go
            doc_maxlen=300,             # Max tokens per doc
        )

        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection_path)
