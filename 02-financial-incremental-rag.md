# Project 2: Financial Incremental RAG

Repo: https://github.com/vedantghodekar15/financial-incremental-rag

Status per the README: **MVP in development.** Say this openly; frame it as "core idea built, still iterating".

## 30-second pitch

A Financial Document Intelligence System that lets users query financial reports with RAG, with one key twist: **chunk-level incremental indexing**. When a document changes, only the modified sections are re-embedded and updated in the vector database, instead of rebuilding the whole index.

## The problem (product framing)

Financial documents are revised often: quarterly updates, restated figures, amended filings. A basic RAG system re-processes the entire document (or whole corpus) on every change. That wastes:

- **Time:** users wait for re-indexing.
- **Money/compute:** every chunk is re-embedded even if 95% did not change.
- **Freshness:** stale answers while the index rebuilds.

Incremental indexing fixes this by touching only what changed. This is the "why" you should lead with.

## Features (from the README)

- Upload financial documents: PDF, DOCX, TXT
- Semantic document chunking
- Financial Q&A using the Groq LLM
- ChromaDB vector storage
- Chunk-level incremental indexing
- Performance dashboard
- FastAPI backend, Streamlit frontend

## Tech stack

Python, FastAPI, Streamlit, Groq API, Sentence Transformers, ChromaDB, SQLite, LangChain, PyMuPDF.

## Repository layout (what each folder is for)

| Folder | Role |
|---|---|
| `app/` | Backend application code (FastAPI) |
| `frontend/` | Streamlit UI |
| `documents/` | Uploaded source documents |
| `chroma_storage/` | Persisted ChromaDB vector store |
| `manifest_db/` | SQLite store that tracks what is already indexed |
| `analytics_db/` | SQLite store for performance metrics behind the dashboard |

## How incremental indexing works (explain it like this)

1. A document is split into chunks.
2. Each chunk gets a fingerprint (for example a content hash).
3. The **manifest** records which chunks (and fingerprints) are already in the vector store.
4. On re-upload of an updated document, new fingerprints are compared with the manifest:
   - **Unchanged** chunks: skipped, no embedding cost.
   - **New or modified** chunks: embedded and inserted/updated in ChromaDB.
   - **Removed** chunks: deleted from ChromaDB and the manifest.
5. The performance dashboard records how much work was saved (chunks skipped vs re-embedded, time taken).

> Before the interview: open `app/` and confirm exactly how you fingerprint chunks and how you handle deletions. The README states the idea; know the code-level detail so you can answer follow-ups confidently.

## Why these tech choices

| Choice | Why |
|---|---|
| ChromaDB | Persistent vector store with easy add/update/delete by ID, which incremental updates need (FAISS in Project 1 has weaker support for updating individual vectors) |
| SQLite manifest | Lightweight, transactional record of what is indexed |
| Streamlit | Fast to build a data-app UI in Python, good for an MVP and a dashboard |
| PyMuPDF | Fast, reliable PDF text extraction |
| Groq | Low-latency LLM responses |

## Relationship to Project 1

Project 1 is the full-stack product (React, auth, summaries). Project 2 goes deeper on one hard problem: **keeping the index efficient and fresh as documents change**. Project 1 proved the RAG flow; Project 2 attacks the cost and freshness limits of that flow. This is a good "I iterate on what I learn" story.

## Key trade-offs

- **Chunk boundaries:** if an edit shifts text, chunk boundaries can move and many chunks look "changed". Stable, structure-aware chunking (by section/heading) reduces needless re-embedding.
- **Extra bookkeeping:** the manifest adds complexity and a new failure mode (manifest and vector store drifting out of sync). Updates should be done so both stay consistent.
- **Savings depend on the edit pattern:** big gains when few chunks change, small gains when a document is rewritten.

## Product-analyst angle: metrics

- **Efficiency:** % of chunks skipped per update; embedding calls saved; re-index time vs full rebuild.
- **Freshness:** time from upload of a revised document to answers reflecting it.
- **Quality guard:** answer accuracy before vs after incremental update must match a full rebuild (the key correctness check).
- **Cost:** embeddings and tokens per update.
- **Dashboard design question:** what does a user or admin actually need to see? Speed saved, documents indexed, last update time.

## Likely interview questions

1. **What is incremental indexing and why does it matter?** Re-embed only changed chunks; saves time and compute and keeps answers fresh.
2. **How do you detect a changed chunk?** Compare chunk fingerprints against the manifest (confirm your exact method in code).
3. **How do you know it is correct?** Compare answers and retrieval results against a full rebuild on the same documents.
4. **What would break it?** Chunk-boundary shifts, manifest/vector-store drift, deleted sections left behind.
5. **How would you prove business value?** Report % compute saved and update latency before vs after on a realistic revision scenario.
6. **Why is it still MVP?** Be straightforward: core pipeline and dashboard idea exist; next steps are evaluation, hardening, and more document types.
7. **What would you do next?** Section-aware chunking, automated regression tests against full rebuild, multi-document/versioning support.
