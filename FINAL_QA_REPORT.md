# FINAL QA REPORT – Advanced RAG Studio (pre-release audit)

**Scope:** engineering QA of this project after the initial v1.0 commit; the fixes and tests described here are part of the final v1.0 release commit.
**Status keys:** **PASS** = verified by an executed automated test with evidence · **PARTIAL** = verified in part, gap stated · **FAIL** = defect remains · **NOT TESTED** = not exercised.
**Honesty note:** this report does **not** claim 100 % completion. Items under §13 and §14 are not verified. No manual browser testing was performed by the auditor.

## 0. Headline numbers
| Item | Result |
|---|---|
| Full suite | `python -m pytest -q` → **346 passed, 0 failed, 0 skipped** (v1.0 baseline: 102) |
| New tests added in this audit | **244** across 10 new test files (+2 helper modules, `tests/appharness.py`, `tests/pdfgen.py`) plus additions to 3 existing files |
| Defects found and fixed | **14** (see §12), each with a regression test that was shown to **fail without the fix** (mutation check, §12) |
| Live LLM API calls during this audit | **0** (evidence in §15) |
| Release | the fixes and tests in this report are included in the final `v1.0` release commit |

## 1. Architecture note and operation inventory
The project is **not a REST service**: there are no HTTP routes, no ASGI/Flask/FastAPI app, and no database. Functionality is exposed through (a) an internal Python service layer (`src/`), (b) six Streamlit pages that call it (`ui/`), and (c) two CLI scripts (`evaluation/`). Persistence is file-based: JSON/NPY/FAISS files under `data/indexes/`, one JSON file per session under `data/sessions/`, JSONL traces in `logs/`. **There are no database models.** Configuration is 13 environment variables read in `src/config.py`.

### 1a. Key service operations (curated)
| # | Operation (file) | Purpose | Inputs | Output | Error handling | Tests (before → after) |
|---|---|---|---|---|---|---|
| 1 | `load_file` (`src/ingestion/loader.py`) | Extract text from PDF/TXT/MD | path | `LoadedDocument(pages[page,text], size, sha256, type)` | `IngestionError` for unsupported/empty/blank/corrupt/missing; **now also unreadable/directory** | basic → real PDFs, truncated, blank, unicode, invalid UTF-8 |
| 2 | `chunk_document`, `split_text` (`chunker.py`) | Overlapping chunks with section + page metadata | doc, size, overlap | `list[Chunk]` (id, doc id, source, page, index, text, section) | `ValueError` on bad size/overlap | basic → invariants over the real 15-doc corpus |
| 3 | `KnowledgeBase.add_file / remove_document / rebuild / status` (`indexer.py`) | Own docs, chunks, embeddings, FAISS, BM25, persistence | file / doc id | dict / bool / stats | duplicate, id-collision, embedding failure → `IngestionError`; corrupt/missing index files → empty KB + `last_error`; nothing persisted on failure | basic → persistence, reload, recovery, no re-embed on rebuild |
| 4 | `SentenceTransformerEmbedder.encode` | Local MiniLM embeddings | `list[str]` | normalised float32 (n×384) | lazy load | **real model** tests added |
| 5 | `VectorStore.build/search/save` (`vector_store.py`) | FAISS inner-product (cosine) | vectors, ids | `(id, score)` list | length mismatch → `ValueError`; empty → `[]` | basic → reload equality; **OpenMP guard** |
| 6 | `BM25Retriever.build/search` | Sparse retrieval | chunks / query | `(id, score>0)` | empty/blank query → `[]` | basic → unicode |
| 7 | `rrf_fuse`, `HybridRetriever.search` (`hybrid_retriever.py`) | RRF over vector+BM25 (per query/subquery) | queries | `HybridResult(per_query, vector, bm25, fused)` | blank queries dropped; single-retriever results fuse correctly | basic → stability, one-retriever-empty, multi-query |
| 8 | `CrossEncoderReranker.rerank` | Cross-encoder rescoring (max over subqueries) | queries, candidates, top_n | `RerankResult(items, reranked, status, error)` | model load/predict failure → explicit `reranked=False` (never faked) | fake model → **real model** |
| 9 | `QueryTransformer.transform` | History-aware rewrite + optional decomposition | question, history | `TransformedQuery` | malformed JSON / wrong types → fallback; fatal LLM errors re-raised | basic → full pipeline with real retrieval |
| 10 | `ContextGrader.grade` | LLM evidence-sufficiency verdict | question, chunks | `ContextGrade(relevant, score, missing, explanation, graded)` | empty context → insufficient without an LLM call; malformed → `graded=False` | basic → 8 branch tests |
| 11 | `CorrectiveRAG.run/_refine` | Grade → refine → retry (bounded) | question, queries, retrieve_fn | `CorrectiveResult(attempts, final, retried, proceed)` | refine failure/malformed → stop and abstain; limit enforced | basic → real recovery, duplicates, empty 2nd retrieval |
| 12 | `Generator.generate/abstain` | Grounded answer / LLM-written abstention | question, chunks, history | `GenerationResult` | invalid JSON → text + flag; abstention strips citations | basic → full pipeline |
| 13 | `validate_citations` (`utils/citations.py`) | Check/repair citations vs retrieved chunks | answer, chunk index | `CitationReport` | unknown ids removed, wrong file/page repaired, **malformed markup removed** | basic → +5 |
| 14 | `RAGPipeline.ask / inspect` | Orchestrate all stages | question, session id | `RAGResponse` (answer, citations, all stage results, timings, call ids, stages) | empty query / no key / empty KB / LLM errors → `error` string, no exception; **empty answer → error** | 6 → 48 pipeline tests |
| 15 | `SessionManager.*` (`memory/session_manager.py`) | Per-session JSON persistence | session id, messages | dicts | id validation (path traversal); **corrupt/invalid file quarantined**; **window ≤ 0 → []** | 6 → 25 |
| 16 | `TracedLLM.generate` (`llm/provider.py`) | Single choke-point for every LLM call: retries, tokens, trace | prompt, call type | `LLMResult` | bounded backoff for 5xx/network only; 429/auth/404/400 never retried; key scrubbing | basic → provider matrix |
| 17 | `GroqProvider.complete`, `GeminiProvider.complete` | Provider calls (OpenAI-compatible HTTP / google-genai) | prompt, system, json_mode, caps | `RawCompletion(text, usage, diagnostics)` | status→kind matrix, empty/malformed responses, timeouts | Groq 20 → +47 for both |
| 18 | `build_llm` (`llm/factory.py`) | Provider selection from `LLM_PROVIDER` | Settings | `TracedLLM` or `None` | invalid provider → `LLMError`; missing key → `None` | covered |
| 19 | `Tracer.log/read`, `summarize` | JSONL traces + metrics | call records | records / dict | scrubs keys; **tolerates malformed/incomplete lines** | 8 → 18 |
| 20 | `run_offline_retrieval_eval` (`evaluation/offline.py`) | Retrieval metrics, **0 LLM calls** | KB, reranker, gold items | report dict | excluded items listed; empty index OK | added earlier, +8 |
| 21 | `Evaluator.run/evaluate_item`, `aggregate`, `validate_dataset`, `load_dataset` | End-to-end evaluation | dataset, pipeline | report dict | errors per item recorded; judge failure → `n/a`; **dataset shape/problems validated** | basic → CLI + UI paths |
| 22 | CLI `evaluation/run_evaluation.py`, `run_offline_eval.py` | Run evaluations | argv | stdout + saved JSON | no key → exit 2 | none → 5 tests |
| 23 | Streamlit pages ×6 (`ui/pages/*.py`) | Chat, Knowledge Base, Retrieval Inspector, Evaluation, Observability, Session Memory | widgets | UI | friendly errors, empty states | none → **45 AppTest tests** |
| 24 | `app.py` | Shell: services, theme selector, navigation | query param, radio | page | **config errors shown as a message** | none → 9 |
| 25 | `Settings` (`config.py`) | 13 env vars | env | dataclass | **clear `ValueError` for bad numbers** | 0 → 10 |

Full automated symbol list (173 public symbols in 38 modules): see **Appendix A**.

## 2. Document ingestion and extraction – **PASS** (with PARTIAL for real-world PDFs)
Tests: `tests/test_ingestion_real.py` (22), `tests/test_ingestion.py` (10), `tests/pdfgen.py` (genuine PDF builder).

| Requirement | Status | Evidence (test) |
|---|---|---|
| Genuine text PDF, text extraction matches source | PASS | `test_genuine_pdf_text_extraction_matches_source` (PDF built with real content streams + xref table; real `pypdf` extraction compared to the source strings) |
| Multi-PDF ingestion | PASS | `test_multi_pdf_ingestion_into_knowledge_base` |
| Multi-page + page-number preservation | PASS | `test_multipage_pdf_page_numbers_preserved_through_chunks` (unique token only on page 3 → only page-3 chunks) |
| Page references usable for citations | PASS | `test_pdf_chunks_support_page_citations`, `test_page_references_for_pdf_sources` (wrong filename/page repaired) |
| Metadata, document ids, chunk ids/indices | PASS | `test_metadata_and_sections`, `test_filename_edge_cases_*`, `test_doc_id_collision_*` |
| Chunk size/overlap/no loss over the real corpus | PASS | `test_real_corpus_chunking_invariants` (all 15 docs) |
| Empty, blank, corrupt, truncated PDFs | PASS | `test_blank_and_corrupt_and_truncated_pdfs` |
| Unsupported formats (6 cases) | PASS | `test_unsupported_formats` |
| Duplicates / re-ingestion / same name changed content | PASS | `test_reingestion_after_delete_*`, `test_duplicate_and_consistency` |
| Unicode & special characters | PASS | UTF-8 md/txt round-trip (CJK, emoji), Latin-1 PDF, invalid UTF-8 bytes |
| Very short documents; missing/malformed input | PASS | `test_very_short_documents`, `test_malformed_chunk_metadata_rejected`, directory-as-file (**bug 1**) |
| PDFs from real word processors / scanners / embedded fonts / images / CJK text in PDF / encrypted / very large | **NOT TESTED** | the test PDF is a hand-built but structurally genuine, base-14-font PDF; **real-world PDF diversity is PARTIAL** |

## 3. Embedding and indexing (real local models) – **PASS**
Tests: `tests/test_indexing_real.py` (15), real MiniLM + real FAISS + real BM25 on the 15-document corpus, `HF_HUB_OFFLINE=1`.
Model init & dimension 384 & unit norm & semantic ordering ✔ · empty / blank / single / multi inputs ✔ · FAISS persistence, reload (identical top-k and scores), faiss file readable ✔ · BM25 corpus file consistent ✔ · **FAISS ids == BM25 ids == chunk ids, same order, counts equal** ✔ · empty-index behaviour ✔ · deleted `faiss.index` recovers from embeddings ✔ · deleted/corrupt `embeddings.npy`, `index_meta.json`, `documents.json`, `chunks.json` → clear `last_error`, no crash, `rebuild()` recovers ✔ · rebuild re-embeds **0** unchanged chunks ✔ · duplicate chunk text across documents ✔ · mismatched vector/id lengths rejected ✔.
Not tested: concurrent writers, approximate FAISS indexes (not implemented), very large corpora – **NOT TESTED**.

## 4. RAG feature-by-feature (`tests/test_pipeline_real.py`, 48 tests: real embeddings/FAISS/BM25/cross-encoder, scripted LLM)
| Technique | Status | Evidence |
|---|---|---|
| A. Query transformation | **PASS** | clear query unchanged; multi-part decomposition drives retrieval for each subquery and pulls evidence from both expected documents; follow-up resolved from history and used for retrieval; 6 malformed rewrite shapes and 4 malformed decomposition shapes fall back; decomposition LLM error non-fatal. **Found/fixed bug 2** (string `subqueries` became single characters). |
| B. Hybrid retrieval | **PASS** | ranking stability, no duplicates, RRF with a single list, BM25-empty case, empty/None query without any LLM call |
| C. Real cross-encoder | **PASS** | reorders, descending scores, ranks `1..k`, pre-rank/RRF/chunk metadata preserved, empty candidates, >512-token and blank candidates, `top_n` > candidates |
| D. Grounded generation + citations | **PARTIAL** | PASS: supported answer with multiple citations mapped to chunks *and* documents; citation to a chunk that exists in the KB but **was not retrieved** is rejected; page references; missing citations flagged; malformed JSON; **empty answer = error (bug 4)**; **malformed citation markup removed (bug 3)**. **NOT TESTED / not implemented: detection of *unsupported claims* inside a correctly cited sentence** – that needs an LLM judge (no calls allowed) |
| E. Corrective RAG | **PASS** | see §6 |
| F. Abstention | **PASS** | see §7 |

## 5. Citation validation – **PASS**
The validator **rejects**, it does not merely format: unknown ids removed and reported (`test_invalid_citation_removed`), ids valid in the KB but not in the retrieved set removed, wrong filename/page repaired, malformed `[Source: …]` forms (`Chunk: id1, id2`, missing `Chunk:`, `[Source: x]`, leading space) removed, canonical labels stable when validated twice, abstentions cannot carry citations. **Not verified against a real model's citation habits beyond the two live runs recorded earlier.**

## 6. Corrective RAG branches – **PASS** (all 8 listed branches)
The mock grader judges "sufficient iff a retrieved passage contains `Cormack`", so recovery is caused by **genuinely different real retrieval results**.
| Branch | Test |
|---|---|
| 1. Sufficient first retrieval → no correction, no refine call | `test_E1_*` |
| 2. Insufficient → corrective query generated (grader's "missing information" present in the refine prompt) | `test_E2_E3_*` |
| 3. **Second retrieval recovers**: first top-5 lacks the evidence, retry query retrieves the chunk, re-grade sufficient, answer cites it, final context = second retrieval | `test_E2_E3_*` (asserts both attempts, grades, stages, generator prompt contains the evidence) |
| 4. Second retrieval still insufficient → abstention; answer generator never invoked; abstained turn persisted with no sources | `test_E4_*` |
| 5. Corrective-query generation fails (4 malformed shapes + LLM error) → no second retrieval, abstain | `test_E5_*`, `test_E5b_*` |
| 6. Retry limit 0/1/2 → exactly 1/2/3 attempts and refine calls | `test_E6_*` |
| 7. Empty second retrieval → insufficient **without** an LLM grading call | `test_E7_*` |
| 8. Identical first/second results terminate cleanly | `test_E8_*` |
Design observation (not a defect): `relevant` is authoritative and `score` is informational; a grader returning `relevant=true, score=0.1` proceeds to generation, whose own abstention rule is the second line of defence.

## 7. Abstention – **PASS**
Unrelated question; generator-level abstention when the grader said sufficient; partially-supported answer keeps only cited parts; inconsistent grader output (`relevant=false`, `score=0.95`) treated as insufficient; empty KB → error with **0** LLM calls; model-written abstention that contains fabricated citations is stripped; abstained turn persisted (`metadata.abstained=True`, `sources=[]`) including after a simulated restart (§8).

## 8. Memory and persistence – **PASS** (`tests/test_memory_persistence.py` 19 + `test_memory.py` 6 + pipeline/UI tests)
Creation, multi-turn, follow-up resolution (history in the rewrite prompt), restart simulation with a new `SessionManager`, unicode, empty sessions, window truncation (**bug 7: window 0/−1 leaked all history**), many isolated sessions, unknown id, 6 invalid ids (path traversal), reset, markdown export, **corrupt/invalid session JSON quarantined (bug 6)** with the Chat and Memory pages still rendering. Not tested: two processes writing the same session concurrently – **NOT TESTED**; a real Streamlit process restart – manual.

## 9. LLM providers and error handling – **PASS (mocked HTTP / fake SDK)**
`tests/test_providers_qa.py` (47) + `test_groq_provider.py` (20) + `test_llm_config.py` (10). Both providers satisfy the same interface (signature equality asserted). Status matrix 400/401/403/404/429/500/502/503/504 for both; Gemini "API key not valid" 400 → auth; network exceptions by **type** (**bug 8**); malformed JSON, empty `choices`, `null` content, non-JSON error bodies; missing/partial usage → labelled estimated; output-limit empty response hint; retry policy table for both providers (fatal kinds never retried; 5xx exactly `1+max_retries`; `max_retries=0` → one request; no loops); key redaction in errors, prompts, responses, diagnostics; factory and missing-key/invalid-provider behaviour.
**NOT TESTED:** real Groq/Gemini behaviour beyond the live runs recorded earlier (no calls made); Gemini live end-to-end was never completed in this project.

## 10. Evaluation and observability – **PASS**
Gold dataset loads and **validates against the real corpus** (18 questions, 3 unanswerable, 0 problems); malformed dataset shapes rejected with `ValueError` (**bug 9**); `validate_dataset` flags duplicate ids, empty questions, missing references and missing documents and is now used by the UI and CLI; metric edge cases; offline evaluation makes **zero** LLM calls (signature, patched `generate`/`complete`, socket guard, CLI run, and unchanged call-log bytes); invalid ground truth/empty retrieval/empty index; report round-trips; both CLIs (offline; live CLI with no key → exit 2; with mock pipeline, `--limit`, `--no-judge` honoured). Observability: success/failure, provider vs estimated tokens, latency, model, request id, finish reason, diagnostic headers, redaction, missing diagnostics, malformed/partial/non-object/incomplete JSONL lines (**bug 11**), 20 000-record log summarised in < 10 s.
**NOT TESTED:** a real LLM judge response (only the "judge unavailable" and "judge disabled" paths).

## 11. Streamlit automated tests – **PARTIAL** (`tests/test_streamlit_pages.py` 45, `test_utils_components.py` 17, config test)
Uses `streamlit.testing.v1.AppTest` with mocked services (hash embedder, fake cross-encoder, scripted LLM).
| Area | Status | Evidence |
|---|---|---|
| Chat | PASS | empty state, sample question → answer + citation expander + trace expander + persisted session; typed follow-up uses memory; abstention + corrective pills; missing key disables input; 5 LLM error kinds shown as friendly messages without tracebacks; empty KB; clear chat; corrupt session |
| Knowledge Base | PASS | listing/counts, search filter + no-match state, empty state, valid **PDF upload with page metadata**, corrupt/empty/blank uploads shown as errors with index untouched (**bug 10**: messages used to vanish), `.docx` blocked by the uploader, duplicate rejected, delete keeps FAISS/BM25 consistent, rebuild |
| Retrieval Inspector | PASS | empty state; all 8 stage sections with real scores; empty query and missing key errors; quota error; never generates an answer |
| Evaluation | PASS | offline run (0 LLM calls), end-to-end run with mock LLM, no-key disabled, invalid dataset error, dataset warnings |
| Observability | PASS | empty state, real call records, failures + diagnostics, tolerant of bad log lines |
| Session Memory | PASS | transcript, tabs/state dump, clear, new session, session isolation, corrupt session |
| Themes | PARTIAL | Auto = light base + `prefers-color-scheme: dark` media query; Dark/Light = explicit overrides with **no** media query; selection persists in state + URL across reruns; hostile `?theme=` value falls back to Auto; page CSS contains no hard-coded hex colours. **Actual rendered appearance, contrast, OS-preference switching: NOT TESTED automatically → manual** |
| Navigation | PARTIAL | the six page titles/URL paths are asserted via a spy on `st.navigation`; **actual clicking between pages is NOT TESTED** (AppTest cannot switch callable-based pages) → manual |
| Loading states / spinners, narrow viewport layout | **NOT TESTED** (CSS has a `max-width: 720px` rule; layout not checked) |
| HTML-injection safety of custom components | PASS | `test_components_escape_untrusted_html_everywhere` (tables, cards, evidence, charts, stages) |

## 12. Bugs discovered and fixed (14)
Each regression test was **mutation-verified**: the committed (pre-fix) file was temporarily restored and the test failed; fixes were then restored and the full suite re-run green.
| # | Defect | Where | Fix | Regression test |
|---|---|---|---|---|
| 1 | Directory / unreadable path crashed with a raw `IsADirectoryError` | `loader.load_file` | catch `OSError` → `IngestionError` | `test_missing_file_and_directory_and_unreadable_input` |
| 2 | A string `"subqueries"` was split into single-character queries; same for corrective `"alternatives"` | `query_transformer.transform`, `corrective_rag._refine` | require a list | `test_decomposition_failure_keeps_pipeline_running[...text]`, `test_non_list_alternatives_*` |
| 3 | Malformed citation markup (`Chunk: id1, id2`, `[Source: x]`) passed through **unvalidated** to the user (my first fix also double-processed valid labels; caught by the suite before finishing) | `utils/citations.validate_citations` | single-pass validation; leftover `[Source:` forms removed and reported | `test_malformed_citation_markup_never_survives_unvalidated`, `test_validator_*` |
| 4 | Empty model output produced a blank "successful" answer that was persisted | `rag/pipeline._run` | treated as an error, nothing persisted | `test_empty_llm_answer_is_an_error_not_a_blank_success` |
| 5 | **Process abort** (`OMP: Error #15`, exit 134) when FAISS and torch initialise their separate OpenMP runtimes in the wrong order; the full test suite crashed. The app worked only because of import-order luck | new `utils/omp_guard.py`, used by `retrieval/vector_store.py` | import scikit-learn then torch before FAISS (ordering found by isolated experiments: torch alone, or heavy torch ops first, still abort) | `tests/test_omp_guard.py` (subprocess) |
| 6 | A corrupt/invalid session file crashed the Chat/Memory pages | `memory/session_manager.load` | quarantine as `*.corrupt-<ts>`, start fresh | `test_corrupt_session_file_*`, 5 invalid shapes |
| 7 | `window(session, 0)` returned the **entire** history | `session_manager.window` | `n <= 0 → []` | `test_non_positive_window_sends_no_history` |
| 8 | Gemini network errors with empty/odd messages (e.g. `ReadTimeout("")`) classified `unknown`: not retried, unhelpful message | `gemini_client._classify` | classify by exception type | `test_gemini_network_exceptions_*` (6 types) |
| 9 | Non-list dataset JSON raised a raw `TypeError` (page crash path) | `evaluator.load_dataset` | `ValueError` with message | `test_load_dataset_rejects_malformed_files` |
| 10 | Upload errors/successes, delete and rebuild messages vanished instantly because `st.rerun()` followed them | `ui/pages/knowledge_base.py` | message queue in session state shown after rerun | `test_kb_upload_validation_errors_*` |
| 11 | One incomplete log line (`{"status": "failure"}`) raised `KeyError` in metrics and the Observability page | `tracer.read`, `metrics.summarize` | normalise/skip records | `test_tracer_tolerates_malformed_partial_and_incomplete_records` |
| 12 | ASCII-only tokenizer: `café`→`caf`, CJK dropped (BM25 blind to non-English text); `exact_match("日本語","中文")` scored **1.0** | `utils/text_utils`, `evaluation/scoring.normalize` | unicode-aware tokens/normalisation (offline eval numbers unchanged) | `test_tokenizer_and_bm25_handle_non_ascii_text`, `test_exact_match_and_f1_do_not_collapse_non_ascii_answers` |
| 13 | A non-integer or ≤0 numeric env value produced a raw traceback at startup | `config._int_env`, `app.py` | clear `ValueError`; app shows "Configuration error: …" | `test_invalid_numeric_env_values_give_clear_errors`, `test_app_shows_configuration_error_*` |
| 14 | Hygiene: 7 unused imports, dead `CitationReport.has_valid`, unused `pandas` requirement, undeclared (guard-required) `scikit-learn`, `validate_dataset` unused | various | removed / declared / wired into UI+CLI | static checks (§13a) |
Caveat on the mutation evidence for bug 9: restoring the pre-fix `evaluator.py` also removed `validate_dataset`, so that mutation showed a collection error rather than the single targeted failure.

## 13. Static and regression checks
**13a – PASS:** `compileall` clean · every one of the 38 modules imports in a **fresh interpreter** (no circular imports) · `pip check` clean · every third-party import declared in `requirements.txt` (and `pandas` removed as unused) · AST scan: no unused imports remain · dead-code scan clean · no key-like strings or absolute/OS-specific paths in app/docs (all path handling uses `pathlib`) · `.env.example`/README/`config.py` consistency tests (every env var documented, defaults equal) · app startup booted (§16).
**NOT TESTED / not configured:** type checking (no mypy config), linting (no ruff/flake8 config; none added to avoid new tooling), Windows/Linux path behaviour (only macOS arm64 run).
**Existing regression tests:** all 102 pre-existing tests still pass unchanged (one helper signature, `ScriptedProvider.complete`, was extended earlier).

## 14. Features still not tested, and items requiring manual browser verification
**Not tested (automated):** real LLM behaviour of Gemini end to end (never completed; 404/503/504 observed earlier) · real LLM judge output · detection of unsupported claims in cited sentences · real-world PDF variety (scanned, encrypted, embedded fonts, CJK in PDF, very large) · concurrent sessions/writers and multi-user Streamlit behaviour · type checks/lint · non-macOS platforms · log files of hundreds of MB.
**Manual browser checklist (not performed by the auditor):**
1. Sidebar navigation between all six pages; active-page highlight.
2. Appearance radio: **Auto** follows OS dark/light; **Dark** and **Light** override; contrast of text, inputs, tables, expanders, tabs, chat bubbles, charts; reload keeps the choice (`?theme=`).
3. Narrow window / mobile width: sidebar, cards, tables, stage list wrap sensibly.
4. Knowledge Base: drag-and-drop upload of a **real PDF**, progress bar, success/error messages now persisting, delete, rebuild.
5. Chat: spinner during a request, citation expanders showing the source text, trace expander, Clear chat, behaviour with a real key (Groq) and with the key removed.
6. Retrieval Inspector with a real key (**uses 3–6 LLM calls per press**).
7. Evaluation page buttons (offline = free; end-to-end uses LLM calls).
8. Observability charts with real data; Session Memory export download.

## 15. Zero live LLM API calls – **CONFIRMED**
* Rules observed: no Groq/Gemini call was made by the auditor; every test uses scripted providers, `httpx.MockTransport` or fake SDK clients.
* **Enforcement:** `tests/conftest.py` has an autouse fixture that blanks `GEMINI_API_KEY`/`GROQ_API_KEY` in the test environment and makes any non-loopback socket connect raise `AssertionError`; it also sets `HF_HUB_OFFLINE=1` so the real local models load only from disk. The whole suite passes under it, so no test attempted network access.
* **Evidence:** `logs/llm_calls.jsonl` (real calls only) holds 25 records, **70 416 bytes**, last modified `2026-11-03 23:44:04` local (last record `2026-11-03T18:14:04Z`), i.e. **before** this audit began (audit clock ≈ `2026-11-04 00:17`). The bytes were unchanged across the offline-eval CLI run, the full suite and application start-up.
* `.env`, API keys, provider and model were not read for display, modified or changed.

## 16. Application start-up
Start-up was verified locally: `streamlit run app.py --server.address localhost --server.port 8501` returned HTTP 200 and `ok` from `/_stcore/health`, and the LLM call log was unchanged afterwards (no LLM request is made at start-up). The application is a local app; it is not hosted or deployed anywhere.

---
## Appendix A – tests per file
| Test file | Tests |
|---|---|
| `tests/test_pipeline_real.py` | 48 |
| `tests/test_providers_qa.py` | 47 |
| `tests/test_streamlit_pages.py` | 45 |
| `tests/test_ingestion_real.py` | 22 |
| `tests/test_groq_provider.py` | 20 |
| `tests/test_memory_persistence.py` | 19 |
| `tests/test_utils_components.py` | 17 |
| `tests/test_indexing_real.py` | 15 |
| `tests/test_eval_obs_qa.py` | 14 |
| `tests/test_generation.py` | 11 |
| `tests/test_config_consistency.py` | 10 |
| `tests/test_ingestion.py` | 10 |
| `tests/test_llm_config.py` | 10 |
| `tests/test_corrective_rag.py` | 9 |
| `tests/test_evaluation.py` | 9 |
| `tests/test_observability.py` | 8 |
| `tests/test_retrieval.py` | 8 |
| `tests/test_memory.py` | 6 |
| `tests/test_pipeline_integration.py` | 6 |
| `tests/test_query_transformation.py` | 6 |
| `tests/test_reranking.py` | 4 |
| `tests/test_omp_guard.py` | 2 |
| **Total** | **346** |

## Appendix B – automated symbol inventory
| Location | Symbol | Purpose (docstring) | Name refs in tests |
|---|---|---|---|
| `app.py:15` | `get_services` | Shared, read-mostly services (models, indexes). Per-user state stays in st.sessi | 0 |
| `evaluation/run_evaluation.py:14` | `main` |  | 3 |
| `evaluation/run_offline_eval.py:14` | `main` |  | 3 |
| `src/bootstrap.py:17` | `Services` |  | 5 |
| `src/bootstrap.py:28` | `build_services` |  | 4 |
| `src/config.py:33` | `Settings` | Runtime settings. Override any field in tests by passing keyword arguments. | 22 |
| `src/config.py:70` | `Settings.has_api_key` | True when the SELECTED provider has a key configured. | 4 |
| `src/config.py:75` | `Settings.active_model` |  | 4 |
| `src/config.py:79` | `Settings.llm_log_path` |  | 2 |
| `src/config.py:82` | `Settings.ensure_dirs` |  | 3 |
| `src/evaluation/evaluator.py:21` | `GoldItem` |  | 18 |
| `src/evaluation/evaluator.py:33` | `load_dataset` |  | 6 |
| `src/evaluation/evaluator.py:40` | `validate_dataset` | Human-readable problems in a gold dataset (empty list = valid). `available_docs` | 3 |
| `src/evaluation/evaluator.py:61` | `Evaluator` |  | 3 |
| `src/evaluation/evaluator.py:62` | `Evaluator.__init__` |  | n/a |
| `src/evaluation/evaluator.py:65` | `Evaluator.run` |  | 75 |
| `src/evaluation/evaluator.py:75` | `Evaluator.evaluate_item` |  | 1 |
| `src/evaluation/evaluator.py:131` | `aggregate` |  | 11 |
| `src/evaluation/evaluator.py:156` | `save_results` |  | 4 |
| `src/evaluation/offline.py:28` | `run_offline_retrieval_eval` |  | 9 |
| `src/evaluation/offline.py:63` | `save_offline` |  | 4 |
| `src/evaluation/scoring.py:10` | `normalize` |  | 0 |
| `src/evaluation/scoring.py:14` | `exact_match` |  | 2 |
| `src/evaluation/scoring.py:18` | `token_f1` |  | 5 |
| `src/evaluation/scoring.py:29` | `keyword_recall` | Fraction of expected keywords (case-insensitive substring) present in the answer | 4 |
| `src/evaluation/scoring.py:37` | `precision_at_k` |  | 5 |
| `src/evaluation/scoring.py:42` | `recall_at_k` |  | 5 |
| `src/evaluation/scoring.py:46` | `hit_rate_at_k` |  | 6 |
| `src/evaluation/scoring.py:50` | `reciprocal_rank` |  | 3 |
| `src/evaluation/scoring.py:57` | `citation_validity` | Fraction of cited chunk ids that exist among retrieved chunks; None if nothing w | 6 |
| `src/evaluation/scoring.py:62` | `abstention_correct` |  | 7 |
| `src/evaluation/scoring.py:66` | `heuristic_correct` | Transparent rule: keyword recall >= 0.6 when keywords exist, otherwise token F1  | 5 |
| `src/evaluation/scoring.py:71` | `mean` |  | 2 |
| `src/ingestion/chunker.py:14` | `Chunk` |  | 27 |
| `src/ingestion/chunker.py:25` | `doc_id_for` |  | 4 |
| `src/ingestion/chunker.py:29` | `split_text` | Split into windows of ~`size` chars, preferring paragraph / sentence / word boun | 3 |
| `src/ingestion/chunker.py:58` | `chunk_document` |  | 8 |
| `src/ingestion/indexer.py:30` | `Embedder` |  | 0 |
| `src/ingestion/indexer.py:33` | `Embedder.encode` |  | 28 |
| `src/ingestion/indexer.py:36` | `SentenceTransformerEmbedder` | Local sentence-transformers embedder; the model is loaded lazily once. | 2 |
| `src/ingestion/indexer.py:39` | `SentenceTransformerEmbedder.__init__` |  | n/a |
| `src/ingestion/indexer.py:43` | `SentenceTransformerEmbedder.encode` |  | 28 |
| `src/ingestion/indexer.py:55` | `KnowledgeBase` |  | 29 |
| `src/ingestion/indexer.py:56` | `KnowledgeBase.__init__` |  | n/a |
| `src/ingestion/indexer.py:114` | `KnowledgeBase.ready` |  | 9 |
| `src/ingestion/indexer.py:117` | `KnowledgeBase.chunk_map` |  | 5 |
| `src/ingestion/indexer.py:120` | `KnowledgeBase.status` |  | 49 |
| `src/ingestion/indexer.py:126` | `KnowledgeBase.add_file` | Copy a file into the documents dir (if needed), chunk, embed and index it. | 16 |
| `src/ingestion/indexer.py:163` | `KnowledgeBase.remove_document` |  | 3 |
| `src/ingestion/indexer.py:175` | `KnowledgeBase.rebuild` | Re-read every file in the documents dir, re-chunk, and re-embed only text not se | 4 |
| `src/ingestion/loader.py:11` | `IngestionError` | Raised when a file cannot be turned into indexable text (message is user-safe). | 14 |
| `src/ingestion/loader.py:16` | `Page` |  | 9 |
| `src/ingestion/loader.py:22` | `LoadedDocument` |  | 0 |
| `src/ingestion/loader.py:30` | `file_hash` |  | 0 |
| `src/ingestion/loader.py:34` | `load_file` | Extract text from a file; raises IngestionError for unsupported/empty/corrupt in | 23 |
| `src/llm/factory.py:13` | `build_llm` | Return a traced client for the selected provider, or None when its API key is mi | 5 |
| `src/llm/gemini_client.py:12` | `GeminiProvider` |  | 8 |
| `src/llm/gemini_client.py:13` | `GeminiProvider.__init__` |  | n/a |
| `src/llm/gemini_client.py:41` | `GeminiProvider.complete` |  | 24 |
| `src/llm/gemini_client.py:92` | `build_llm` | Return a traced Gemini client, or None when no API key is configured. | 5 |
| `src/llm/groq_client.py:15` | `GroqProvider` |  | 8 |
| `src/llm/groq_client.py:16` | `GroqProvider.__init__` |  | n/a |
| `src/llm/groq_client.py:28` | `GroqProvider.complete` |  | 24 |
| `src/llm/provider.py:20` | `LLMError` | Provider failure with a coarse `kind` so the UI can show a helpful message. | 53 |
| `src/llm/provider.py:23` | `LLMError.__init__` |  | n/a |
| `src/llm/provider.py:31` | `LLMError.transient` |  | 2 |
| `src/llm/provider.py:35` | `LLMError.fatal` | Errors retrying or falling back cannot fix; the pipeline surfaces them to the us | 1 |
| `src/llm/provider.py:39` | `LLMError.user_message` |  | 5 |
| `src/llm/provider.py:51` | `RawCompletion` |  | 2 |
| `src/llm/provider.py:59` | `LLMResult` |  | 0 |
| `src/llm/provider.py:69` | `LLMProvider` |  | 5 |
| `src/llm/provider.py:73` | `LLMProvider.complete` | Single provider call. Raise LLMError on failure. | 24 |
| `src/llm/provider.py:78` | `TracedLLM` | Wraps a provider with bounded retries, token accounting and JSONL tracing. | 51 |
| `src/llm/provider.py:84` | `TracedLLM.__init__` |  | n/a |
| `src/llm/provider.py:93` | `TracedLLM.model` |  | 38 |
| `src/llm/provider.py:96` | `TracedLLM.generate` |  | 36 |
| `src/memory/session_manager.py:14` | `SessionManager` |  | 29 |
| `src/memory/session_manager.py:15` | `SessionManager.__init__` |  | n/a |
| `src/memory/session_manager.py:24` | `SessionManager.create` |  | 53 |
| `src/memory/session_manager.py:29` | `SessionManager.exists` |  | 7 |
| `src/memory/session_manager.py:35` | `SessionManager.load` | Load a session. A damaged/invalid file is quarantined (renamed *.corrupt) and re | 4 |
| `src/memory/session_manager.py:58` | `SessionManager.add_message` |  | 14 |
| `src/memory/session_manager.py:67` | `SessionManager.add_context` |  | 1 |
| `src/memory/session_manager.py:72` | `SessionManager.messages` |  | 26 |
| `src/memory/session_manager.py:75` | `SessionManager.window` | Bounded recent history (role + content only) for follow-up rewriting. | 6 |
| `src/memory/session_manager.py:81` | `SessionManager.reset` |  | 3 |
| `src/memory/session_manager.py:87` | `SessionManager.export_markdown` |  | 3 |
| `src/observability/diagnostics.py:12` | `safe_headers` | Allow-listed response headers only; credentials and cookies are never copied. | 3 |
| `src/observability/diagnostics.py:23` | `error_diagnostics` | Structured, size-bounded details from a google-genai APIError (status, message,  | 2 |
| `src/observability/diagnostics.py:48` | `response_diagnostics` | Diagnostics for a successful response: request-id style headers, finish reason,  | 1 |
| `src/observability/metrics.py:7` | `summarize` |  | 6 |
| `src/observability/tracer.py:14` | `Tracer` |  | 12 |
| `src/observability/tracer.py:15` | `Tracer.__init__` |  | n/a |
| `src/observability/tracer.py:21` | `Tracer.scrub` |  | 1 |
| `src/observability/tracer.py:31` | `Tracer.log` |  | 2 |
| `src/observability/tracer.py:77` | `Tracer.read` | All valid JSON-object records (malformed, partial or non-object lines are skippe | 23 |
| `src/rag/context_grader.py:12` | `ContextGrade` |  | 0 |
| `src/rag/context_grader.py:22` | `format_context` |  | 0 |
| `src/rag/context_grader.py:31` | `ContextGrader` |  | 6 |
| `src/rag/context_grader.py:32` | `ContextGrader.__init__` |  | n/a |
| `src/rag/context_grader.py:35` | `ContextGrader.grade` |  | 18 |
| `src/rag/corrective_rag.py:19` | `Attempt` |  | 0 |
| `src/rag/corrective_rag.py:27` | `CorrectiveResult` |  | 0 |
| `src/rag/corrective_rag.py:36` | `CorrectiveResult.retry_queries` |  | 2 |
| `src/rag/corrective_rag.py:40` | `CorrectiveRAG` |  | 5 |
| `src/rag/corrective_rag.py:41` | `CorrectiveRAG.__init__` |  | n/a |
| `src/rag/corrective_rag.py:44` | `CorrectiveRAG.run` |  | 75 |
| `src/rag/generator.py:14` | `CitationInfo` |  | 0 |
| `src/rag/generator.py:21` | `GenerationResult` |  | 0 |
| `src/rag/generator.py:36` | `Generator` |  | 3 |
| `src/rag/generator.py:37` | `Generator.__init__` |  | n/a |
| `src/rag/generator.py:40` | `Generator.generate` |  | 36 |
| `src/rag/generator.py:57` | `Generator.abstain` | LLM-written abstention message based on grader notes (no evidence passages are s | 21 |
| `src/rag/pipeline.py:20` | `StageRecord` |  | 0 |
| `src/rag/pipeline.py:27` | `RAGResponse` |  | 0 |
| `src/rag/pipeline.py:54` | `RAGPipeline` |  | 8 |
| `src/rag/pipeline.py:55` | `RAGPipeline.__init__` |  | n/a |
| `src/rag/pipeline.py:66` | `RAGPipeline.ask` | Run the full pipeline. `remember=False` skips memory writes (evaluation); `histo | 36 |
| `src/rag/pipeline.py:94` | `RAGPipeline.inspect` | Run retrieval stages (transform, hybrid, rerank, grade, corrective) without gene | 6 |
| `src/rag/query_transformer.py:11` | `TransformedQuery` |  | 0 |
| `src/rag/query_transformer.py:22` | `TransformedQuery.retrieval_queries` | Queries sent to the retriever: the (re)written query plus any subqueries. | 3 |
| `src/rag/query_transformer.py:31` | `QueryTransformer` |  | 4 |
| `src/rag/query_transformer.py:32` | `QueryTransformer.__init__` |  | n/a |
| `src/rag/query_transformer.py:35` | `QueryTransformer.transform` |  | 2 |
| `src/retrieval/bm25_retriever.py:10` | `BM25Retriever` |  | 0 |
| `src/retrieval/bm25_retriever.py:11` | `BM25Retriever.__init__` |  | n/a |
| `src/retrieval/bm25_retriever.py:15` | `BM25Retriever.build` |  | 58 |
| `src/retrieval/bm25_retriever.py:20` | `BM25Retriever.search` | Return up to k (chunk_id, bm25_score) pairs with score > 0, best first. | 37 |
| `src/retrieval/hybrid_retriever.py:10` | `RetrievedChunk` |  | 0 |
| `src/retrieval/hybrid_retriever.py:24` | `QueryRetrieval` |  | 0 |
| `src/retrieval/hybrid_retriever.py:30` | `HybridResult` |  | 0 |
| `src/retrieval/hybrid_retriever.py:38` | `rrf_fuse` | Reciprocal Rank Fusion: score(d) = sum_i 1 / (k + rank_i(d)), ranks are 1-based. | 6 |
| `src/retrieval/hybrid_retriever.py:57` | `HybridRetriever` |  | 20 |
| `src/retrieval/hybrid_retriever.py:58` | `HybridRetriever.__init__` |  | n/a |
| `src/retrieval/hybrid_retriever.py:61` | `HybridRetriever.search` |  | 37 |
| `src/retrieval/reranker.py:9` | `RerankResult` |  | 3 |
| `src/retrieval/reranker.py:16` | `CrossEncoderReranker` |  | 21 |
| `src/retrieval/reranker.py:17` | `CrossEncoderReranker.__init__` |  | n/a |
| `src/retrieval/reranker.py:31` | `CrossEncoderReranker.rerank` | Score every (query, chunk) pair; a chunk's score is its max over `queries` (orig | 13 |
| `src/retrieval/vector_store.py:14` | `VectorStore` |  | 5 |
| `src/retrieval/vector_store.py:15` | `VectorStore.__init__` |  | n/a |
| `src/retrieval/vector_store.py:19` | `VectorStore.build` |  | 58 |
| `src/retrieval/vector_store.py:27` | `VectorStore.search` | Return up to k (chunk_id, cosine_score) pairs, best first. | 37 |
| `src/retrieval/vector_store.py:35` | `VectorStore.save` |  | 0 |
| `src/utils/citations.py:15` | `Citation` |  | 2 |
| `src/utils/citations.py:20` | `Citation.label` |  | 6 |
| `src/utils/citations.py:26` | `CitationReport` |  | 0 |
| `src/utils/citations.py:33` | `validate_citations` | Check every citation against `chunk_index` (chunk_id -> {source, page}). | 8 |
| `src/utils/omp_guard.py:14` | `import_torch_before_faiss` |  | 3 |
| `src/utils/text_utils.py:12` | `tokenize` | Lowercase alphanumeric tokens (used by BM25 and scoring). | 5 |
| `src/utils/text_utils.py:17` | `tokenize_no_stop` |  | 0 |
| `src/utils/text_utils.py:21` | `slugify` |  | 0 |
| `src/utils/text_utils.py:25` | `estimate_tokens` | Rough token estimate (~4 chars/token). Always labelled 'estimated' wherever it i | 0 |
| `src/utils/text_utils.py:30` | `extract_json` | Parse a JSON object from model output, tolerating code fences and surrounding pr | 0 |
| `ui/components.py:10` | `esc` |  | 0 |
| `ui/components.py:14` | `metric_cards` | items: (label, value, sub) | 0 |
| `ui/components.py:21` | `pill` |  | 0 |
| `ui/components.py:25` | `empty_state` |  | 0 |
| `ui/components.py:29` | `html_table` |  | 0 |
| `ui/components.py:36` | `bar_chart` |  | 0 |
| `ui/components.py:47` | `line_chart` | Inline SVG line chart; strokes use CSS variables so it follows the theme. | 0 |
| `ui/components.py:65` | `stage_list` |  | 0 |
| `ui/components.py:72` | `evidence` |  | 9 |
| `ui/components.py:76` | `citation_panel` |  | 0 |
| `ui/components.py:86` | `chunk_rows` | Rows for retrieval tables from RetrievedChunk objects. | 0 |
| `ui/pages/chat.py:36` | `render` |  | 2 |
| `ui/pages/evaluation.py:25` | `render` |  | 2 |
| `ui/pages/knowledge_base.py:17` | `render` |  | 2 |
| `ui/pages/memory.py:8` | `render` |  | 2 |
| `ui/pages/observability.py:9` | `render` |  | 2 |
| `ui/pages/retrieval_inspector.py:19` | `render` |  | 2 |
| `ui/styles.py:97` | `build_css` |  | 2 |
| `ui/theme.py:21` | `token_css` | CSS variable definitions for the chosen mode. | 0 |

_41 modules, 173 public symbols. 'Name refs' counts textual references in `tests/` (a rough coverage proxy, not line coverage)._
