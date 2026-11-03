# Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is an architecture that gives a large language model access to external knowledge at answer time. Instead of relying only on facts memorised in model weights, a RAG system first retrieves relevant passages from a document collection and then asks the model to answer using those passages.

## Why RAG exists
Language models have a training cutoff, cannot see private documents, and sometimes produce fluent but false statements known as hallucinations. RAG addresses all three problems: knowledge can be updated by changing the document index rather than retraining the model, private data stays in the company's own store, and answers can be traced back to citations.

## The basic pipeline
A minimal RAG pipeline has four steps. First, documents are split into chunks and converted into embeddings during an offline indexing phase. Second, at query time the user question is embedded and the most similar chunks are retrieved. Third, the chunks are placed into a prompt as evidence. Fourth, the model generates an answer grounded in that evidence.

## Naive versus advanced RAG
Naive RAG performs a single vector search and sends the top results to the model. It fails when the question is vague, when the right words differ from the document wording, or when the retrieved chunks are only loosely related. Advanced RAG adds techniques before retrieval (query rewriting and decomposition), during retrieval (hybrid search and Reciprocal Rank Fusion), after retrieval (cross-encoder reranking and context grading) and during generation (grounded prompting, citations and abstention).

## Typical failure modes
Common failures include retrieving the wrong chunk, splitting an answer across chunk boundaries, stuffing too much irrelevant context into the prompt, and the model ignoring the evidence. Each advanced technique in this knowledge base targets one of these failures.
