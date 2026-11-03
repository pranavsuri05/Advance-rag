# Hallucination Detection and Abstention

A hallucination is generated content that is not supported by the source material or by reality. In RAG, the most important kind is unfaithful generation: statements that go beyond the retrieved context.

## Causes
Hallucinations occur when retrieval returns irrelevant or incomplete chunks and the model fills the gaps from memory, when the prompt does not forbid outside knowledge, or when the question has a false premise that the model accepts.

## Layered defences
No single technique is sufficient. A retrieval score threshold alone is unreliable, because similarity scores do not tell whether the passage actually answers the question. A robust system layers several defences: context grading before generation, a grounding prompt with explicit abstention instructions, citation validation after generation, and optional faithfulness checks by a judge model.

## Abstention
Abstention means the system declines to answer when evidence is insufficient. A good abstention message is honest and specific: it says the knowledge base does not contain enough information, mentions what is missing, and does not provide a guess. Over-abstention is also a failure, so abstention accuracy and false-abstention rate should both be tracked.

## Detecting hallucinations
Post-hoc detection methods include claim-level verification against the context, natural language inference models that classify a claim as entailed or not, and self-consistency checks that sample several answers and compare them. These add cost and are usually applied in evaluation rather than on every request.

## Unanswerable question example
If the knowledge base describes FAISS and BM25 but the user asks for the price of a managed vector database subscription, the correct behaviour is to abstain, even though the topic sounds related.
