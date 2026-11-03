# Sample transcript (ILLUSTRATIVE - MOCK LLM)

> **This is NOT a real Gemini run.** No `GEMINI_API_KEY` was available when this file was generated. Embeddings, FAISS, BM25, RRF and the cross-encoder are the **real local components** over the bundled corpus; every LLM response (rewrite, grading, answer) is scripted by `transcripts/generate_mock_transcript.py`, so answer text is placeholder and says so. Re-generate with a real key by chatting in the app.

## 1. Normal factual question
**User:** What is the default value of k in Reciprocal Rank Fusion?

- Rewritten query: `What is the default value of k in Reciprocal Rank Fusion?`
- Subqueries: none
- Vector hits: 20 · BM25 hits: 20 · RRF candidates: 18 · reranker: cross_encoder
- Attempt 0: queries=['What is the default value of k in Reciprocal Rank Fusion?'] → grade sufficient (0.90); top chunks: ['04-hybrid-search-and-rrf_chunk_001', '14-semantic-search-and-retrieval-metrics_chunk_001', '04-hybrid-search-and-rrf_chunk_000']
- Corrective retry occurred: **False**
- Outcome: **answered**

Stages:
  - Query transformation: success - query already clear; unchanged
  - Vector search (FAISS): success - 20 hits
  - BM25 keyword search: success - 20 hits
  - Reciprocal Rank Fusion: success - 18 fused candidates (k=60)
  - Cross-encoder reranking: success - 5 chunks kept
  - Context grading: success - initial: sufficient (score 0.90)
  - Corrective retry: skipped - initial context sufficient
  - Answer generation: success - answered from evidence
  - Citation validation: success - 2 valid, 0 removed, 0 repaired

**Assistant (mock text):** (Mock answer text) The sources state the relevant facts. [Source: 04_hybrid_search_and_rrf.md, Chunk: 04-hybrid-search-and-rrf_chunk_001] [Source: 14_semantic_search_and_retrieval_metrics.md, Chunk: 14-semantic-search-and-retrieval-metrics_chunk_001]

Citations: `04_hybrid_search_and_rrf.md` / `04-hybrid-search-and-rrf_chunk_001`, `14_semantic_search_and_retrieval_metrics.md` / `14-semantic-search-and-retrieval-metrics_chunk_001`

## 2. Multi-part question with decomposition / hybrid retrieval
**User:** Explain how hybrid retrieval and reranking work together, and why both are needed.

- Rewritten query: `Explain how hybrid retrieval and reranking work together, and why both are needed.`
- Subqueries: ['How does hybrid retrieval combine vector and keyword search?', 'How does cross-encoder reranking improve relevance?', 'What is the difference between fusion and reranking?']
- Vector hits: 27 · BM25 hits: 36 · RRF candidates: 18 · reranker: cross_encoder
- Attempt 0: queries=['Explain how hybrid retrieval and reranking work together, and why both are needed.', 'How does hybrid retrieval combine vector and keyword search?', 'How does cross-encoder reranking improve relevance?', 'What is the difference between fusion and reranking?'] → grade sufficient (0.90); top chunks: ['06-query-rewriting-and-decomposition_chunk_001', '05-cross-encoder-reranking_chunk_000', '05-cross-encoder-reranking_chunk_002']
- Corrective retry occurred: **False**
- Outcome: **answered**

Stages:
  - Query transformation: success - rewritten; 3 subqueries
  - Vector search (FAISS): success - 27 hits
  - BM25 keyword search: success - 36 hits
  - Reciprocal Rank Fusion: success - 18 fused candidates (k=60)
  - Cross-encoder reranking: success - 5 chunks kept
  - Context grading: success - initial: sufficient (score 0.90)
  - Corrective retry: skipped - initial context sufficient
  - Answer generation: success - answered from evidence
  - Citation validation: success - 2 valid, 0 removed, 0 repaired

**Assistant (mock text):** (Mock answer text) The sources state the relevant facts. [Source: 06_query_rewriting_and_decomposition.md, Chunk: 06-query-rewriting-and-decomposition_chunk_001] [Source: 05_cross_encoder_reranking.md, Chunk: 05-cross-encoder-reranking_chunk_000]

Citations: `06_query_rewriting_and_decomposition.md` / `06-query-rewriting-and-decomposition_chunk_001`, `05_cross_encoder_reranking.md` / `05-cross-encoder-reranking_chunk_000`

## 3. Corrective retrieval triggered
**User:** How do I make the first pass better?

- Rewritten query: `How do I make the first pass better?`
- Subqueries: none
- Vector hits: 20 · BM25 hits: 20 · RRF candidates: 18 · reranker: cross_encoder
- Attempt 0: queries=['How do I make the first pass better?'] → grade insufficient (0.20); top chunks: ['05-cross-encoder-reranking_chunk_002', '08-prompt-engineering-and-grounded-generation_chunk_003', '05-cross-encoder-reranking_chunk_001']
- Attempt 1: queries=['query rewriting and decomposition before retrieval'] → grade sufficient (0.90); top chunks: ['06-query-rewriting-and-decomposition_chunk_000', '01-intro-to-rag_chunk_002', '06-query-rewriting-and-decomposition_chunk_001']
- Corrective retry occurred: **True**
- Outcome: **answered**

Stages:
  - Query transformation: success - query already clear; unchanged
  - Vector search (FAISS): success - 20 hits
  - BM25 keyword search: success - 20 hits
  - Reciprocal Rank Fusion: success - 18 fused candidates (k=60)
  - Cross-encoder reranking: success - 5 chunks kept
  - Context grading: success - initial: insufficient (score 0.20)
  - Corrective retry: success - retry query: 'query rewriting and decomposition before retrieval'; final grade: sufficient (0.90)
  - Answer generation: success - answered from evidence
  - Citation validation: success - 2 valid, 0 removed, 0 repaired

**Assistant (mock text):** (Mock answer text) The sources state the relevant facts. [Source: 06_query_rewriting_and_decomposition.md, Chunk: 06-query-rewriting-and-decomposition_chunk_000] [Source: 01_intro_to_rag.md, Chunk: 01-intro-to-rag_chunk_002]

Citations: `06_query_rewriting_and_decomposition.md` / `06-query-rewriting-and-decomposition_chunk_000`, `01_intro_to_rag.md` / `01-intro-to-rag_chunk_002`

## 4. Unanswerable question (abstention)
**User:** What is the monthly price of a managed Pinecone subscription?

- Rewritten query: `What is the monthly price of a managed Pinecone subscription?`
- Subqueries: none
- Vector hits: 20 · BM25 hits: 20 · RRF candidates: 18 · reranker: cross_encoder
- Attempt 0: queries=['What is the monthly price of a managed Pinecone subscription?'] → grade insufficient (0.05); top chunks: ['10-hallucination-detection-and-abstention_chunk_003', '02-vector-databases-and-embeddings_chunk_000', '04-hybrid-search-and-rrf_chunk_001']
- Attempt 1: queries=['query rewriting and decomposition before retrieval'] → grade insufficient (0.05); top chunks: ['06-query-rewriting-and-decomposition_chunk_000', '01-intro-to-rag_chunk_002', '06-query-rewriting-and-decomposition_chunk_001']
- Corrective retry occurred: **True**
- Outcome: **abstained**

Stages:
  - Query transformation: success - query already clear; unchanged
  - Vector search (FAISS): success - 20 hits
  - BM25 keyword search: success - 20 hits
  - Reciprocal Rank Fusion: success - 18 fused candidates (k=60)
  - Cross-encoder reranking: success - 5 chunks kept
  - Context grading: success - initial: insufficient (score 0.05)
  - Corrective retry: success - retry query: 'query rewriting and decomposition before retrieval'; final grade: insufficient (0.05)
  - Answer generation: skipped - context insufficient after retries -> abstention message
  - Citation validation: skipped - abstained: no citations expected

**Assistant (mock text):** The knowledge base does not contain enough information to answer this: it covers RAG techniques, not this topic.

Citations: none

## 5. Follow-up using conversation memory
**User:** and why is that needed?

- Rewritten query: `Why is cross-encoder reranking needed after hybrid retrieval?`
- Subqueries: none
- Vector hits: 20 · BM25 hits: 20 · RRF candidates: 18 · reranker: cross_encoder
- Attempt 0: queries=['Why is cross-encoder reranking needed after hybrid retrieval?'] → grade sufficient (0.90); top chunks: ['06-query-rewriting-and-decomposition_chunk_001', '05-cross-encoder-reranking_chunk_001', '05-cross-encoder-reranking_chunk_002']
- Corrective retry occurred: **False**
- Outcome: **answered**

Stages:
  - Query transformation: success - rewritten; 0 subqueries
  - Vector search (FAISS): success - 20 hits
  - BM25 keyword search: success - 20 hits
  - Reciprocal Rank Fusion: success - 18 fused candidates (k=60)
  - Cross-encoder reranking: success - 5 chunks kept
  - Context grading: success - initial: sufficient (score 0.90)
  - Corrective retry: skipped - initial context sufficient
  - Answer generation: success - answered from evidence
  - Citation validation: success - 2 valid, 0 removed, 0 repaired

**Assistant (mock text):** (Mock answer text) The sources state the relevant facts. [Source: 06_query_rewriting_and_decomposition.md, Chunk: 06-query-rewriting-and-decomposition_chunk_001] [Source: 05_cross_encoder_reranking.md, Chunk: 05-cross-encoder-reranking_chunk_001]

Citations: `06_query_rewriting_and_decomposition.md` / `06-query-rewriting-and-decomposition_chunk_001`, `05_cross_encoder_reranking.md` / `05-cross-encoder-reranking_chunk_001`
