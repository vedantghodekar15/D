# Project 1: Financial Research Assistant

Repo: https://github.com/vedantghodekar15/financial-research-assistant

## 30-second pitch

A full-stack web app where a user signs up, uploads a financial PDF (annual report, financial statement), gets an AI-generated summary, and asks natural-language questions such as "What was total revenue in 2024?" or "How did operating expenses change year over year?". Answers come from the document itself using Retrieval-Augmented Generation (RAG), not from the model's memory.

## The problem (product framing)

Financial reports are long, dense and numeric. Analysts spend hours hunting for a few figures. The product goal: cut the time from "I have a 150-page PDF" to "I have the number I need, with the supporting excerpt" to a few seconds.

- **User:** analyst, student, or investor reading company filings.
- **Job to be done:** find and verify specific facts in a long document quickly.
- **Why RAG, not a plain chatbot:** a plain LLM can hallucinate numbers. RAG retrieves the relevant chunks first and makes the model answer from them, so answers are grounded and can show source excerpts.

## Features

- PDF upload and processing
- Automatic document summarisation
- Natural-language Q&A over uploaded documents
- Semantic search across documents
- User signup/login with JWT authentication
- Per-user document list, with metadata stored in SQLite

## Architecture

```
React + Vite (frontend)
   |  Axios, JWT in requests
   v
FastAPI (backend)
   |-- auth.py            JWT + bcrypt password hashing
   |-- database.py        SQLAlchemy models, SQLite (users, document metadata)
   |-- document_loader.py PDF text extraction (PyPDF)
   |-- chunking.py        splits text into chunks
   |-- vector_store.py    embeddings + FAISS index
   |-- rag_pipeline.py    retrieve -> build prompt -> LLM answer
   '-- summarizer.py      document summaries
```

**RAG flow in plain words**

1. Upload: PDF is read with PyPDF and split into chunks.
2. Each chunk is converted to an embedding (Sentence Transformers / HuggingFace) and stored in FAISS.
3. Ask: the question is embedded, FAISS returns the most similar chunks.
4. Those chunks are added to the prompt and sent to the LLM (Groq API, via LangChain).
5. The answer is returned to the UI along with relevant excerpts.

## Tech stack and why

| Choice | Why (say this in the interview) |
|---|---|
| FastAPI | Async, fast, auto-generated Swagger docs at `/docs`, clean request validation |
| React 19 + Vite | Fast dev loop, component-based UI, React Router for protected pages |
| FAISS | Fast local similarity search, no separate DB server needed for an MVP |
| Sentence Transformers | Free, local embeddings; no per-call embedding cost |
| Groq API | Very low-latency LLM inference, which matters for a chat-style UX |
| LangChain | Orchestrates retrieval + prompt + LLM so I did not hand-wire everything |
| JWT + bcrypt | Stateless auth; passwords never stored in plain text |
| SQLite + SQLAlchemy | Zero-setup persistence for users and document metadata |

## API endpoints

- `POST /signup`, `POST /login`
- `POST /upload`, `GET /documents`
- `POST /ask`, `POST /summarize`, `POST /search`
- All except signup/login require authentication.

## Key decisions and trade-offs

- **Chunking strategy:** chunk size is a trade-off. Small chunks give precise retrieval but lose context; large chunks keep context but add noise and cost tokens.
- **FAISS vs a hosted vector DB:** FAISS is simple and fast for an MVP, but it is in-process and not built for multi-user, frequently-updated corpora. That limitation is what led to Project 2 (incremental RAG).
- **Grounding:** answering only from retrieved text reduces hallucination, but if retrieval misses the right chunk, the answer is wrong or "not found". Retrieval quality is the real bottleneck.
- **Auth:** JWT with an expiry (30 min in the sample config) balances security and convenience.

## Limitations (be honest, interviewers like this)

- Tables and numbers inside PDFs can be extracted badly by plain text extraction.
- No formal evaluation set yet for answer accuracy.
- Updating a document means re-processing it (addressed in Project 2).
- SQLite and local FAISS would not scale to many concurrent users.

## Product-analyst angle: how I would measure it

- **Activation:** % of signups who upload a document and ask at least one question.
- **Engagement:** questions per document, return visits per week.
- **Quality:** answer accuracy on a labelled question set; thumbs up/down rate; "not found" rate.
- **Performance:** upload-to-ready time, answer latency (p50/p95).
- **Cost:** LLM tokens per question.
- **Funnel to watch:** signup -> upload -> first question -> second question. Drop-off after upload would point at processing time; drop-off after the first answer would point at answer quality.

## Likely interview questions

1. **Walk me through what happens when I ask a question.** Use the 5-step RAG flow above.
2. **Why RAG instead of fine-tuning?** Documents change and are private; RAG needs no retraining and can cite sources. Fine-tuning is costly and bakes in stale facts.
3. **How do you stop hallucinations?** Ground answers in retrieved chunks, instruct the model to say when the answer is not in the context, show source excerpts.
4. **How would you evaluate it?** Build a question/answer set from a few reports, measure answer correctness and retrieval hit rate.
5. **What would you build next?** Table-aware extraction, citations with page numbers, evaluation dashboard, multi-document comparison.
6. **How is user data separated?** Endpoints require JWT; documents are tied to the authenticated user.
7. **Biggest challenge?** Pick one you really faced (chunking quality, extracting numbers from PDFs, wiring auth between React and FastAPI, CORS) and tell it as problem -> what I tried -> result.
