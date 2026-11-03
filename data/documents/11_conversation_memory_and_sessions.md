# Conversation Memory and Session Management

Conversation memory lets an assistant handle follow-up questions such as "Why is that needed?" by remembering what was discussed.

## Session identity
Each conversation belongs to a session with a unique id, often a UUID. Isolating sessions prevents one user's history from leaking into another's. Streamlit applications keep the active session id in st.session_state, which lives per browser tab and survives reruns of the script but not a new tab.

## What to store
A transcript entry should include a timestamp, the role (user or assistant), the message content, the sources cited and metadata such as the rewritten query and timings. Persisting sessions to disk, for example as JSON files, lets the history survive a server restart.

## Bounded windows
Sending the entire history to the model wastes tokens and can exceed context limits. Common strategies are a sliding window of the last N messages, summarising older turns, or retrieving relevant past turns. This project uses a sliding window of the most recent six messages for follow-up query rewriting.

## Memory versus evidence
History is used to understand what the user means, not as a source of facts. Answers must still be grounded in retrieved documents, otherwise an earlier mistake could be repeated as truth.

## Reset and privacy
Users need a clear way to reset a conversation. Clearing a session deletes its messages and context history. Because stored conversations can contain sensitive text, they should be excluded from version control.
