from source.models.Embedding.siglip2 import SigLIP2
from source.models.Embedding.clip import CLIPS
from source.retrieval.retrieval import SemanticRetriever

# Load model
model = SigLIP2()

# Load retriever
retriever = SemanticRetriever(
    embedding_path="data/vector/L21_V001.npy",
    metadata_path="data/metadata/L21_V001.json",
    embedding_model=model,
)

# Search
results = retriever.search(
    query="Chăn nuôi cá",
    top_k=5,
)

# Print
for i, r in enumerate(results):
    print(
        f"{i+1}. score={r['score']:.4f}"
        f" frame={r['frame_idx']}"
        f" image={r['image_uri']}"
    )

# Visualize
retriever.visualize(results)