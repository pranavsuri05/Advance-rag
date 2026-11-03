# Vector Databases and Embeddings

An embedding is a dense vector of floating-point numbers that represents the meaning of a piece of text. Texts with similar meaning are mapped to nearby points in the vector space, which makes semantic search possible even when the query and the document share no words.

## Embedding models
The sentence-transformers model all-MiniLM-L6-v2 produces 384-dimensional embeddings and runs comfortably on a CPU. It was trained on over one billion sentence pairs and is a popular baseline for English semantic search. Larger models give better accuracy but cost more memory and latency. The same embedding model must be used for indexing and for querying; mixing models produces meaningless similarity scores.

## Similarity measures
Cosine similarity compares the angle between two vectors and ignores their length. If every vector is normalised to unit length, cosine similarity equals the inner product (dot product). That is why many systems normalise embeddings and use an inner-product index.

## FAISS
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. The simplest index, IndexFlatIP, performs exact brute-force inner-product search and is perfectly adequate for up to a few hundred thousand vectors. For millions of vectors, approximate indexes such as IVF (inverted file) or HNSW (graph-based) trade a small amount of recall for a large speed-up.

## Vector databases
A vector database adds persistence, metadata filtering, replication and access control on top of nearest-neighbour search. A library like FAISS stores only vectors and integer ids, so the application must keep its own mapping from ids to chunk text and metadata.

## Limitations of dense retrieval
Dense retrieval can miss exact identifiers, rare technical terms, error codes and names, because the embedding blurs them into general meaning. This weakness is the main motivation for combining it with keyword search.
