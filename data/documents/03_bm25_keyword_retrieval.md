# BM25 and Keyword Retrieval

BM25 (Best Matching 25) is a ranking function used by search engines to score documents against a query using word statistics. It is the successor of TF-IDF and remains a very strong baseline for retrieval.

## How BM25 scores documents
For every query term, BM25 multiplies two ingredients. The inverse document frequency (IDF) gives rare terms more weight than common ones, so "FAISS" matters more than "the". The term-frequency component rewards documents that mention the term more often, but with saturation: the tenth occurrence adds much less than the second. A length normalisation factor stops long documents from winning just because they contain more words.

## Parameters
BM25 has two main parameters. The saturation parameter k1, typically between 1.2 and 2.0, controls how quickly extra occurrences stop helping. The length-normalisation parameter b, usually 0.75, controls how strongly document length is penalised.

## Strengths
BM25 needs no training, is fast, is fully explainable because the score can be traced to matching words, and excels at exact matches such as product names, acronyms, function names and error codes. In the rank-bm25 Python library, the BM25Okapi class builds an index from tokenised documents and returns a score for every document.

## Weaknesses
BM25 is lexical: it cannot match synonyms or paraphrases. A query for "automobile repair" will not retrieve a passage that only says "car maintenance". It also depends on consistent tokenisation, so the same tokenizer must be used for the corpus and the query.

## Role in modern RAG
Because dense retrieval is strong on meaning and BM25 is strong on exact words, production systems usually run both and merge the results, an approach called hybrid search.
