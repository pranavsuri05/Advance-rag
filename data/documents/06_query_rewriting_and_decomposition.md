# Query Rewriting and Decomposition

User questions are often poorly suited for retrieval. They may be vague, depend on earlier conversation turns, or ask several things at once. Query transformation uses a language model to turn the raw question into better retrieval queries before any search happens.

## Query rewriting
Rewriting resolves ambiguity and context dependence. In a conversation, the follow-up "And how does it handle ties?" cannot be searched on its own. Given the history, the model rewrites it into a standalone question such as "How does Reciprocal Rank Fusion handle ties?". Rewriting can also replace casual wording with the technical vocabulary used in documents.

## Query decomposition
Decomposition splits a multi-part question into independent subqueries. The question "Explain how hybrid retrieval and reranking work together, and why both are needed" contains several information needs: how hybrid retrieval merges vector and keyword search, how cross-encoder reranking improves relevance, and the difference between fusion and reranking. Each subquery is retrieved separately and the results are merged, so every part of the question receives evidence.

## When not to transform
Blindly rewriting every query is harmful: it adds latency, costs tokens and can drift from the user's intent. A good transformer returns clear, self-contained questions unchanged and only rewrites when needed. It must never add facts that the user did not mention.

## Related techniques
HyDE (Hypothetical Document Embeddings) asks the model to write a fake answer and embeds that. Step-back prompting asks a more general question first. Multi-query retrieval generates several paraphrases. Decomposition differs because the subqueries are about different aspects, not paraphrases.

## Safety
Transformation output should be validated, for example by requiring JSON with expected fields, and the system should fall back to the original query when parsing fails.
