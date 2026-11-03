# Prompt Engineering and Grounded Generation

Grounded generation constrains the language model to answer from supplied evidence rather than from its parametric memory.

## Elements of a grounding prompt
An effective grounding prompt states the role, instructs the model to use only the provided passages, and forbids filling gaps with outside knowledge. It specifies the output structure and requires a citation after every factual claim. It also tells the model what to do when the evidence is insufficient: say so clearly instead of guessing.

## Citations
A citation format such as [Source: filename.md, Chunk: chunk_id] lets readers verify claims. Each passage in the prompt carries a header with its source and chunk id so the model can copy them verbatim. Citations must be validated after generation by checking that every cited chunk id exists among the retrieved chunks; invalid citations are removed or corrected, because models can fabricate plausible-looking references.

## Evidence versus inference
Good answers distinguish between what the sources directly state and what the model reasonably infers from them. Phrases like "The sources state" and "It can reasonably be inferred" make this explicit and help reviewers judge risk.

## Structured outputs
Asking for JSON with an answer field and an abstain flag makes the response machine-checkable. The application must still validate the JSON and handle malformed output with a safe fallback.

## Temperature and context size
Low temperature, such as 0.0 to 0.2, reduces creative drift in factual tasks. Passing only the top four or five reranked chunks keeps the prompt focused; too much context dilutes attention and increases the chance that the model uses an irrelevant passage.

## Conversation history
History helps the model understand follow-up questions, but it must not be treated as evidence. Only retrieved passages count as sources.
