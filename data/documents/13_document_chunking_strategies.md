# Document Chunking Strategies

Chunking splits documents into pieces small enough to embed and retrieve precisely. Chunk quality strongly influences RAG accuracy.

## Fixed-size chunking
The simplest approach cuts text every N characters or tokens. This project defaults to about 800 characters with an overlap of 120 characters. It is fast and predictable but may cut sentences or ideas in half.

## Overlap
Overlap repeats the end of one chunk at the start of the next. If an answer straddles a boundary, the overlap increases the chance that at least one chunk contains it intact. Too much overlap inflates the index and produces near-duplicate results.

## Boundary-aware chunking
Recursive splitters try paragraph breaks first, then sentence boundaries, then spaces, falling back to a hard cut only when necessary. This keeps ideas together. Markdown-aware chunking also tracks the nearest heading so each chunk carries section metadata.

## Semantic chunking
Semantic chunkers embed sentences and start a new chunk where similarity between neighbouring sentences drops. They can produce coherent chunks but cost extra embedding computation at indexing time.

## Choosing a size
Small chunks give precise retrieval but lose context; large chunks keep context but dilute the embedding and waste prompt tokens. Sizes between 500 and 1,000 characters work well for prose. The best value should be chosen by measuring retrieval metrics on a gold dataset.

## Metadata
Each chunk should store its document id, source filename, page number for PDFs, chunk index and section so that citations can point back to the exact location.
