# LLM Observability and Tracing

Observability is the ability to understand what an LLM application did and why. Because model calls are slow, costly and non-deterministic, every call should be recorded.

## What to log per call
A useful trace record contains a unique call id, a timestamp, the model name, the call type, the full prompt, the full response, input and output token counts, latency in milliseconds, status and any error. Adding the session id links calls to a conversation.

## Token accounting
Providers such as Gemini return usage metadata with prompt and candidate token counts. These provider-reported numbers are exact. If usage metadata is missing, an estimate (about four characters per token) can be used, but it must be labelled as estimated and never reported as exact.

## Centralised client
All calls should flow through one client wrapper so that tracing, retry and error handling are consistent. Scattered direct API calls lead to blind spots.

## Reliability patterns
Transient errors such as network failures and 5xx responses deserve bounded retries with exponential backoff, for example waits of 1, 2 and 4 seconds with a maximum number of attempts. Quota errors (HTTP 429) should produce a clear message rather than endless retries. Each request needs a timeout.

## Dashboards
Useful dashboard views include total calls, success and failure counts, token totals, average latency, calls grouped by type, and a recent-call table with expandable prompts and responses.

## Privacy
Prompts can contain user data and documents, so logs should be stored locally with restricted access, and API keys must be scrubbed so that credentials never appear in trace files.
