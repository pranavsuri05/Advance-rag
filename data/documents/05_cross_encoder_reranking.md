# Cross-Encoder Reranking

Reranking is a second-stage step that reorders a shortlist of retrieved candidates using a more accurate but more expensive model.

## Bi-encoders versus cross-encoders
A bi-encoder, used for vector search, embeds the query and each document independently. Document vectors can therefore be precomputed and searched quickly, but the model never sees query and document together. A cross-encoder feeds the query and one document into the transformer as a single concatenated input, so attention can compare every query word with every document word. The output is one relevance score for that pair.

## Cost and trade-off
Because a cross-encoder must run once per query-document pair and nothing can be precomputed, it is too slow to scan a whole corpus. The standard design is two-stage: retrieve roughly 15 to 20 candidates cheaply with hybrid search, then rerank only those with the cross-encoder and pass the best 4 or 5 chunks to the language model.

## Model used in this project
The model cross-encoder/ms-marco-MiniLM-L-6-v2 was trained on the MS MARCO passage ranking dataset. It has 6 transformer layers, runs locally on a CPU and outputs raw logits where higher means more relevant. The scores are only meaningful for comparing candidates for the same query, not across different queries.

## Benefits
Reranking raises precision at the top of the list, reduces the amount of irrelevant text sent to the generator and lowers token cost. Studies on MS MARCO show large improvements in MRR when a cross-encoder reorders BM25 or dense results.

## Pitfalls
If the first-stage retriever never returns the relevant chunk, no reranker can fix that, so candidate recall matters. Very long passages are truncated by the model's maximum input length of 512 tokens.
