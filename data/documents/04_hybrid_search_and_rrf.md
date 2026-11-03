# Hybrid Search and Reciprocal Rank Fusion

Hybrid search runs a dense retriever (vector similarity) and a sparse retriever (BM25) on the same query and merges their results. Each method covers the blind spots of the other, which typically improves recall.

## The score-scale problem
Vector similarity scores (for example cosine values between 0 and 1) and BM25 scores (unbounded, depending on corpus statistics) are not comparable. Adding them directly requires careful normalisation and tuning that breaks when the corpus changes.

## Reciprocal Rank Fusion
Reciprocal Rank Fusion (RRF) avoids the problem by using only ranks. For each document d, the fused score is the sum over all ranked lists of 1 / (k + rank(d)), where rank is the 1-based position in that list and k is a smoothing constant. The commonly used default is k = 60, proposed in the original RRF paper by Cormack, Clarke and Buettcher.

## Worked example
Suppose a chunk is ranked 1st by vector search and 3rd by BM25. Its RRF score is 1/(60+1) + 1/(60+3), about 0.0164 + 0.0159 = 0.0323. A chunk ranked 2nd by vector search only gets 1/62, about 0.0161. The chunk found by both retrievers wins, which is the desired behaviour.

## Properties
A document that appears in only one list simply receives one term in the sum. The constant k dampens the influence of top ranks: a larger k makes the fusion flatter. RRF needs no training data, is robust to outliers and is trivial to implement.

## Fusion versus reranking
Fusion merges multiple candidate lists cheaply using only rank positions; it never reads the text. Reranking, by contrast, reads the query and each candidate together to compute a new relevance score. Fusion improves recall, while reranking improves precision, so production pipelines use both.
