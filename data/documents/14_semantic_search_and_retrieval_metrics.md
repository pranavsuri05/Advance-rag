# Semantic Search and Retrieval Metrics

Semantic search retrieves documents by meaning rather than exact words. Evaluating it requires ranking metrics computed against known relevant documents.

## Precision@K
Precision@K is the fraction of the top K retrieved items that are relevant. If 2 of the top 5 chunks are relevant, Precision@5 is 0.4. It rewards a clean result list.

## Recall@K
Recall@K is the fraction of all relevant items that appear in the top K. If three documents are relevant and two appear in the top 5, Recall@5 is about 0.67. It shows whether the retriever finds the evidence at all.

## Hit Rate@K
Hit Rate@K, also called success@K, is 1 if at least one relevant item appears in the top K and 0 otherwise, averaged over queries. It is a lenient measure of whether retrieval worked.

## Mean Reciprocal Rank
For each query, the reciprocal rank is 1 divided by the position of the first relevant result, or 0 if none appears. MRR is the mean across queries. A first relevant result at rank 1 scores 1.0, at rank 2 scores 0.5, and at rank 4 scores 0.25.

## nDCG
Normalised Discounted Cumulative Gain handles graded relevance and discounts lower positions logarithmically. It is popular in search evaluation but needs graded labels.

## Practical advice
Report several metrics at the K the generator actually uses. Compare retrieval variants, such as vector-only versus hybrid versus hybrid plus reranking, on the same gold set. Document-level relevance labels are cheaper to create than chunk-level labels and are usually sufficient for a prototype.
