# Corrective RAG and Context Grading

Corrective RAG (CRAG) adds a feedback loop after retrieval. Instead of trusting whatever the retriever returns, the system evaluates the retrieved context and takes corrective action when it is inadequate.

## Context grading
A grader, usually a language model, receives the question and the retrieved passages and decides whether they are sufficient to answer. The grader must judge evidence sufficiency, not keyword overlap: a passage can mention every query term without stating the answer. A useful grader returns a structured result containing a relevant flag, a confidence score, a description of the missing information and a short explanation.

## The corrective loop
If the grade is sufficient, the pipeline proceeds directly to generation and no retry occurs. If the grade is insufficient, the system uses the grader's description of the missing information to generate a better retrieval query, runs hybrid retrieval and reranking again, and grades the new context. The number of retries is capped, with one retry being a sensible default, so the loop can never run forever and cost stays bounded.

## What happens after failure
If the context is still insufficient after the maximum number of retries, the system abstains: it tells the user that the knowledge base lacks the information instead of guessing. The original CRAG paper also proposes falling back to web search, but a closed-domain assistant should usually abstain.

## Why a plain second retrieval is not enough
Running retrieval twice with the same query returns the same results. The corrective step is valuable only because the grader diagnoses what is missing and the query generator uses that diagnosis to change the query vocabulary or focus.

## Observability
Each attempt should be recorded: the query, the retrieved context, the grade and the reason. This history makes failures debuggable.
