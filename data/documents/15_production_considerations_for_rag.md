# Production Considerations for RAG Applications

Moving a RAG prototype to production introduces concerns beyond answer quality.

## Latency and cost
Each advanced technique adds work. Query transformation, context grading and corrective query generation each add a language model call, and reranking adds local compute. Teams manage this by skipping transformation when the query is already clear, capping corrective retries, retrieving a bounded number of candidates and caching loaded models.

## Index freshness
Documents change. A production system needs incremental ingestion, deduplication by content hash, deletion that removes chunks from every index consistently, and the ability to rebuild an index. The vector index, keyword index and metadata store must always agree.

## Rate limits and resilience
Hosted model APIs impose quotas. Applications should handle HTTP 429 gracefully, apply bounded exponential backoff for transient failures, set timeouts and degrade clearly instead of crashing. The free tier of a model API usually allows only a small number of requests per minute.

## Security and privacy
Never commit API keys; load them from environment variables. Apply access control so users only retrieve documents they may see. Treat retrieved text as untrusted, since documents can contain prompt-injection instructions.

## Monitoring and evaluation
Run the gold-dataset evaluation on every significant change to catch regressions, and monitor live metrics such as failure rate, latency and token usage. Collect user feedback on answers to extend the dataset.

## Scaling
Exact FAISS search is fine for small collections. Larger deployments move to approximate indexes or a managed vector database and may run rerankers on GPUs.
