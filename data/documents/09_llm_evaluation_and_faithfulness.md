# LLM Evaluation and Faithfulness

Evaluating a RAG system means measuring retrieval quality and generation quality separately, because a wrong answer can come from either stage.

## Gold datasets
A gold dataset contains questions with reference answers, the documents expected to contain the evidence, expected keywords, and a flag saying whether the question is answerable. It should include direct factual questions, multi-hop questions, multi-part questions and unanswerable questions.

## Answer correctness
Simple automatic measures include normalised exact match and token-level F1, which is the harmonic mean of token precision and recall against the reference answer. They are transparent but penalise correct paraphrases. An LLM-as-a-judge can compare meaning but introduces its own bias and cost, so heuristic and judge scores should be reported separately and never confused with objective truth.

## Faithfulness
Faithfulness measures whether every claim in the answer is supported by the retrieved context, regardless of whether the claim is true in the real world. It is typically computed by having a judge model extract claims and check each against the context; the score is the fraction of supported claims. High faithfulness with low correctness points to a retrieval problem, while low faithfulness points to a generation problem.

## Citation validity
Citation validity checks that every cited chunk id appears among the retrieved chunks. It is a deterministic check that needs no judge.

## Abstention accuracy
For unanswerable questions, the system should decline. Abstention accuracy is the fraction of unanswerable questions where the system abstained, and the complementary false-abstention rate measures answerable questions that were wrongly refused.

## Judge reliability
Judges can be inconsistent, so report when a judge was unavailable instead of inventing a score.
