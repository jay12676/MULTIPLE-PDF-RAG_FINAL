# DocuMind — Multi-PDF RAG System

Upload any number of PDFs from **any topic, any field, any size, any number of pages**. DocuMind breaks every PDF into chunks, stores them in a vector database, and lets you ask questions across all of them with exact citations. It also proactively finds contradictions, knowledge gaps, and patterns — without you having to ask.

> **Accepted input: PDF files only (`.pdf`)**
> Works for any topic — research, legal, finance, medical, engineering, education, business, government.
> Digital PDFs and scanned/image-based PDFs both supported.

---

## What Makes This Different From ChatPDF / NotebookLM / AskYourPDF

Every tool in the market does the same thing: upload → ask → get answer.

DocuMind adds **5 features that do not exist in any current tool**:

| Unique Feature | What it does |
|---|---|
| **Contradiction Detector** | Automatically finds where your PDFs disagree with each other |
| **Knowledge Gap Finder** | Tells you what topics your PDFs do NOT cover |
| **Smart Question Generator** | Suggests the best questions to ask your document set |
| **Document Evolution Timeline** | Shows how facts/numbers/opinions changed across documents over time |
| **Proactive Insight Engine** | Surfaces patterns across your PDFs without you asking anything |

---

## What Can You Use It For?

| Domain | Example PDFs | Example Question |
|---|---|---|
| Research & Academia | 50 research papers | "What methods were used across all studies?" |
| Legal | Contracts, agreements | "Which clauses conflict across these agreements?" |
| Finance | Annual reports 2018–2024 | "How did revenue strategy evolve over 6 years?" |
| Medical / Clinical | Clinical trial papers | "What adverse effects were reported across all trials?" |
| Engineering | Technical manuals | "What are the safety limits for component X?" |
| Education | Textbooks + notes | "Summarize all chapters that cover neural networks" |
| Business | Competitor reports | "Where do all reports agree on market trends?" |
| Government / Policy | Policy documents | "What changed between the 2020 and 2024 versions?" |

---

## Table of Contents

1. [How RAG Works in This System](#how-rag-works-chunking--storage)
2. [5 Unique Features](#5-unique-features)
3. [Full System Architecture](#full-system-architecture)
4. [How Streamlit Talks to FastAPI](#how-streamlit-talks-to-fastapi)
5. [Streamlit UI — Screens & Components](#streamlit-ui--screens--components)
6. [Complete Backend Workflow](#complete-backend-workflow)
   - [Phase 1: Ingestion — Parse & Chunk](#phase-1-ingestion--parse--chunk)
   - [Phase 2: Storage — Embed & Index](#phase-2-storage--embed--index)
   - [Phase 3: Retrieval — Hybrid Search](#phase-3-retrieval--hybrid-search)
   - [Phase 4: Reasoning — LangGraph](#phase-4-reasoning--langgraph)
   - [Phase 5: Response — Stream to UI](#phase-5-response--stream-to-ui)
   - [Phase 6: Unique Feature Pipelines](#phase-6-unique-feature-pipelines)
7. [Scalability Design](#scalability-design)
8. [Tech Stack](#tech-stack)
9. [Folder Structure](#folder-structure)
10. [API Endpoints](#api-endpoints)
11. [Setup & Run](#setup--run)

---

## How RAG Works — Chunking & Storage

This is the foundation of the entire system. Understanding this explains why it works at any scale.

```
YOU UPLOAD A PDF
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — PARSE                                                 │
│                                                                 │
│  PDF opened page by page (never fully loaded into RAM)          │
│  Each page → raw text extracted                                 │
│  Scanned page → Tesseract OCR converts image → text            │
│                                                                 │
│  Result: one long string of text per page                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — CHUNK (Break into pieces)                             │
│                                                                 │
│  Why chunk? The LLM has a limited context window.               │
│  We can't feed 500 pages at once. So we break the PDF           │
│  into small overlapping pieces called "chunks".                 │
│                                                                 │
│  4-Level Hierarchy:                                             │
│                                                                 │
│  Level 1 → 1 chunk = entire document summary                   │
│             Used for: "What is this document about?"           │
│                                                                 │
│  Level 2 → 1 chunk per section                                 │
│             (Introduction, Methods, Results, Conclusion...)     │
│             Used for: section-level questions                   │
│                                                                 │
│  Level 3 → paragraph chunks                                     │
│             512 tokens each, 64 token overlap                   │
│             ← this is the main retrieval unit                  │
│             Used for: most questions                            │
│                                                                 │
│  Level 4 → sentence chunks                                      │
│             Used for: exact citation pinpointing               │
│                                                                 │
│  Each chunk carries:                                            │
│    doc_id, page_number, section, level, text                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — EMBED (Convert text to numbers)                       │
│                                                                 │
│  Each chunk → sentence-transformers model → 384-dim vector      │
│                                                                 │
│  "Drug X causes headache" → [0.23, -0.11, 0.87, ...]           │
│  "Medication Y leads to nausea" → [0.21, -0.09, 0.84, ...]     │
│                                                                 │
│  These two are CLOSE in vector space = semantically similar     │
│  even though they share no exact words.                         │
│                                                                 │
│  This is why RAG finds relevant content even when the           │
│  user's question uses different words than the document.        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4 — STORE (Persist everything)                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FAISS Vector Store                                      │   │
│  │  Stores: the 384-dim vectors (embeddings)                │   │
│  │  Used for: similarity search — "find chunks closest      │   │
│  │            to the user's question in meaning"            │   │
│  │  Sharded per project, persisted to disk                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SQLite / PostgreSQL                                     │   │
│  │  Stores: doc metadata + chunk text + page numbers        │   │
│  │  Used for: "which document, which page did this          │   │
│  │            chunk come from?" → citation generation       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  BM25 Index                                              │   │
│  │  Stores: keyword frequencies per chunk                   │   │
│  │  Used for: exact keyword search (names, codes, IDs)      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

WHEN USER ASKS A QUESTION:
  Question → embed → search FAISS → find top-20 similar chunks
           → retrieve chunk text from SQLite
           → send chunks + question to GROQ LLM
           → LLM answers using only those chunks
           → cites doc name + page number
```

**Key point:** The LLM never reads your full PDFs. It only reads the 20 most relevant chunks retrieved for each question. This is why it's fast and accurate regardless of how many pages or PDFs you have.

---

## 5 Unique Features

### Feature 1 — Contradiction Detector

Automatically scans all uploaded PDFs and surfaces places where they disagree.

```
Triggered: after all PDFs are indexed OR on-demand via button

Pipeline:
  For each major topic found across docs:
    Pull top chunks from all documents about that topic
    Send to GROQ: "Do any of these chunks contradict each other?"
    If YES → store as contradiction with both sides cited

Output in Streamlit:

  CONTRADICTIONS FOUND IN YOUR DOCUMENTS
  ┌────────────────────────────────────────────────────────┐
  │ Topic: Market growth rate                              │
  │                                                        │
  │ report_2023.pdf (Page 4)  → "Market grew 18% in 2022" │
  │ analysis_Q4.pdf (Page 11) → "Market grew 6% in 2022"  │
  │                                                        │
  │ Likely reason: Different data sources used             │
  │ Confidence: High                                       │
  │ [View Both Passages]                                   │
  └────────────────────────────────────────────────────────┘
```

---

### Feature 2 — Knowledge Gap Finder

After ingestion, tells you what topics your documents are missing.

```
Triggered: automatically after all PDFs indexed

Pipeline:
  Extract all topics covered across all docs (using LLM)
  For each domain (auto-detected from content):
    Compare covered topics vs expected topics for that domain
    Flag what's missing

Output in Streamlit:

  KNOWLEDGE GAPS IN YOUR DOCUMENT SET
  ┌────────────────────────────────────────────────────────┐
  │ Your 12 PDFs cover well:                               │
  │   ✓ Revenue & financial performance                    │
  │   ✓ Product strategy                                   │
  │   ✓ Team & hiring                                      │
  │                                                        │
  │ Not covered in any document:                           │
  │   ✗ Regulatory / compliance risk                       │
  │   ✗ Supply chain dependencies                          │
  │   ✗ International market exposure                      │
  │                                                        │
  │ Suggested: Upload documents that cover these topics    │
  └────────────────────────────────────────────────────────┘
```

---

### Feature 3 — Smart Question Generator

After upload, generates the best questions the user can ask — before they ask anything.

```
Triggered: automatically after indexing completes

Pipeline:
  Sample chunks from all docs at Level 1 and Level 2
  Send to GROQ: "Generate 10 high-value questions a user
                 should ask about these documents"
  Categorize: comparison, factual, gap, trend questions

Output in Streamlit (shown when chat is empty):

  QUESTIONS YOUR DOCUMENTS CAN ANSWER
  ┌────────────────────────────────────────────────────────┐
  │ Comparison                                             │
  │  [How do methodologies differ across the 5 studies? →]│
  │  [Which document has the most recent data?           →]│
  │                                                        │
  │ Key Insights                                           │
  │  [What are the main risk factors across all docs?    →]│
  │  [Where do the authors most strongly agree?          →]│
  │                                                        │
  │ Not answerable (gaps in your docs)                     │
  │  [Effects beyond 2 years — not covered in your PDFs]  │
  └────────────────────────────────────────────────────────┘
```

---

### Feature 4 — Document Evolution Timeline

If PDFs span multiple years, builds a visual timeline of how something changed.

```
Triggered: user asks "how did X evolve?" OR clicks Timeline tab

Pipeline:
  Detect dates across all documents (published year, report year)
  For the queried topic: retrieve relevant chunks per document
  Sort by date → send to GROQ: "Summarize how this topic
                                changed across these dates"

Output in Streamlit:

  EVOLUTION: "Remote Work Policy"
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  2019 ──●── "Full-time office mandatory"               │
  │              policy_2019.pdf · Page 3                  │
  │          │                                             │
  │  2020 ──●── "WFH allowed temporarily (COVID)"          │
  │              policy_2020.pdf · Page 1                  │
  │          │                                             │
  │  2022 ──●── "Permanent hybrid — 3 days office"         │
  │              policy_2022.pdf · Page 5                  │
  │          │                                             │
  │  2024 ──●── "Full return to office required"           │
  │              policy_2024.pdf · Page 2                  │
  │                                                        │
  │  KEY SHIFT: Complete reversal of 2020 policy by 2024   │
  └────────────────────────────────────────────────────────┘
```

---

### Feature 5 — Proactive Insight Engine

Automatically surfaces patterns across all PDFs — no question needed.

```
Triggered: runs once after all PDFs are indexed

Pipeline:
  Sample chunks across all documents
  Run LangGraph pattern-finding loop:
    Node 1: Extract key claims from each document
    Node 2: Find recurring themes across docs
    Node 3: Find anomalies (things mentioned in only 1 doc)
    Node 4: Find trends (things increasing/decreasing over docs)
    Node 5: Format as insight cards

Output in Streamlit (shown in dashboard on first load):

  INSIGHTS FROM YOUR 15 DOCUMENTS
  ┌────────────────────────────────────────────────────────┐
  │ Pattern found in 12/15 docs                            │
  │ "AI investment" mentioned as increasing in all         │
  │  competitor reports — but no specific budget cited     │
  │                                                        │
  │ Trend across years                                     │
  │ Supply chain risk mentioned heavily in 2022–2023 docs, │
  │ absent from 2024 docs — issue likely resolved          │
  │                                                        │
  │ Anomaly                                                │
  │ Only report_outlier.pdf claims market share declined — │
  │ all other 14 docs say it grew                          │
  └────────────────────────────────────────────────────────┘
```

---

## Full System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND  (port 8501)                       │
│                                                                          │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────┐   │
│  │  SIDEBAR       │  │  CHAT INTERFACE  │  │  UNIQUE FEATURES PANEL │   │
│  │                │  │                  │  │                        │   │
│  │ • Project mgr  │  │ • st.chat_input  │  │ • Contradictions       │   │
│  │ • PDF uploader │  │ • Streaming resp │  │ • Knowledge Gaps       │   │
│  │ • Upload status│  │ • Citations      │  │ • Smart Questions      │   │
│  │ • Doc list     │  │ • Follow-ups     │  │ • Timeline             │   │
│  │ • Mode selector│  │ • Chat history   │  │ • Proactive Insights   │   │
│  └───────┬────────┘  └────────┬─────────┘  └──────────┬─────────────┘   │
└──────────┼────────────────────┼──────────────────────┼──────────────────┘
           │  multipart POST    │  POST + SSE stream    │  GET
           ▼                    ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND  (port 8000)                      │
│                                                                          │
│  POST /upload        POST /query          GET /insights                  │
│  GET /jobs/{id}      GET /contradictions  GET /gaps                      │
│  GET /documents      GET /questions       GET /timeline                  │
│  DELETE /documents   GET /sessions        GET /audit                     │
│                                                                          │
│  ┌──────────────────────┐   ┌────────────────────────────────────────┐  │
│  │   INGESTION QUEUE    │   │   QUERY + UNIQUE FEATURES ENGINE       │  │
│  │   Celery + Redis     │   │   LangGraph graphs for:                │  │
│  │   (background jobs)  │   │   • Multi-hop Q&A                      │  │
│  └──────────┬───────────┘   │   • Contradiction detection            │  │
│             │               │   • Gap finding                         │  │
└─────────────┼───────────────│   • Insight generation                  │  │
              │               └──────────────┬─────────────────────────┘  │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER (3 stores)                            │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────────┐  ┌─────────────────┐   │
│  │  FAISS           │  │  SQLite/PostgreSQL    │  │  Redis          │   │
│  │  Vector Store    │  │  Relational Store     │  │  Cache          │   │
│  │                  │  │                       │  │                 │   │
│  │ chunk embeddings │  │ documents table       │  │ query cache     │   │
│  │ 384-dim vectors  │  │ chunks table          │  │ session memory  │   │
│  │ HNSW ANN index   │  │ contradictions table  │  │ job status      │   │
│  │ sharded by proj  │  │ insights table        │  │                 │   │
│  └──────────────────┘  │ audit_log table       │  └─────────────────┘   │
│                         └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How Streamlit Talks to FastAPI

```
┌─────────────────────────────────────────────────────────────────────┐
│  ACTION                  │  STREAMLIT              │  FASTAPI        │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  User uploads PDFs       │  st.file_uploader()     │  POST /upload   │
│                          │  → multipart POST        │  → Celery jobs  │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Progress bar            │  polls GET /jobs/{id}   │  0–100% status  │
│                          │  every 1s via st.empty() │  from Celery   │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  User asks question      │  st.chat_input()        │  POST /query    │
│                          │  → stream=True           │  → SSE tokens  │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Contradictions tab      │  GET /contradictions    │  returns list   │
│                          │  on tab open             │  from DB        │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Knowledge gaps tab      │  GET /gaps              │  returns list   │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Smart questions         │  GET /questions         │  returns list   │
│  (shown on empty chat)   │  after indexing done     │  of questions  │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Timeline query          │  GET /timeline?topic=X  │  returns        │
│                          │                          │  dated events  │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Insights dashboard      │  GET /insights          │  returns        │
│                          │  on project load         │  insight cards │
└──────────────────────────┴─────────────────────────┴─────────────────┘
```

---

## Streamlit UI — Screens & Components

```
┌────────────────────────┬────────────────────────────────────────────────┐
│  SIDEBAR               │  TABS                                          │
│                        │  [Chat][Contradictions][Gaps][Timeline][Insights│
│  Project Selector      │  ────────────────────────────────────────────  │
│  [dropdown]            │                                                │
│  [+ New Project]       │  TAB 1 — CHAT                                  │
│                        │  ┌──────────────────────────────────────────┐  │
│  Upload PDFs           │  │  (when chat empty — Smart Questions)     │  │
│  [drag & drop]         │  │  Questions your docs can answer:         │  │
│                        │  │  [How do methodologies differ?        →] │  │
│  report.pdf      ✓     │  │  [What are the main risk factors?     →] │  │
│  manual.pdf      ✓     │  │  [Where do authors agree most?        →] │  │
│  thesis.pdf     ⟳68%   │  │──────────────────────────────────────────│  │
│                        │  │  user: Compare findings across all docs  │  │
│  Indexed Docs          │  │──────────────────────────────────────────│  │
│  report   42 pgs       │  │  assistant: [streams live...]            │  │
│  manual  210 pgs       │  └──────────────────────────────────────────┘  │
│  [Delete]              │  [Ask anything...                        ] [→] │
│                        │                                                │
│  Mode: [General ▼]     │  After answer:                                 │
│  Top-K: [20]           │  Citations · Confidence · Follow-ups           │
└────────────────────────┴────────────────────────────────────────────────┘

TAB 2 — CONTRADICTIONS
  ┌────────────────────────────────────────────────────────────────────┐
  │  3 contradictions found across your 12 PDFs          [Re-scan]    │
  │                                                                    │
  │  ▼ Market growth rate                                              │
  │    report_2023.pdf p.4  → "grew 18%"                               │
  │    analysis.pdf    p.11 → "grew 6%"                                │
  │    Likely reason: Different data sources                           │
  │                                                                    │
  │  ▼ Project timeline                                                │
  │    plan_v1.pdf p.2  → "Launch Q1 2025"                             │
  │    plan_v2.pdf p.1  → "Launch Q3 2025"                             │
  └────────────────────────────────────────────────────────────────────┘

TAB 3 — KNOWLEDGE GAPS
  ┌────────────────────────────────────────────────────────────────────┐
  │  Topics your 12 PDFs DO cover:  ✓ Revenue  ✓ Strategy  ✓ Team     │
  │  Topics NOT covered:  ✗ Compliance  ✗ Supply chain  ✗ ESG         │
  └────────────────────────────────────────────────────────────────────┘

TAB 4 — TIMELINE
  ┌────────────────────────────────────────────────────────────────────┐
  │  Topic: [type a topic...]  [Build Timeline]                        │
  │  2019 ──●── finding from oldest doc                                │
  │  2021 ──●── how it changed                                         │
  │  2024 ──●── current state                                          │
  └────────────────────────────────────────────────────────────────────┘

TAB 5 — INSIGHTS
  ┌────────────────────────────────────────────────────────────────────┐
  │  Auto-generated after upload — no question needed                  │
  │  Pattern · Trend · Anomaly cards                                   │
  └────────────────────────────────────────────────────────────────────┘
```

---

## Complete Backend Workflow

### Phase 1: Ingestion — Parse & Chunk

**Trigger:** Streamlit sends `POST /upload` → FastAPI queues one Celery job per PDF.

```
PDF received by FastAPI
        │
        ├─ Validate: must be application/pdf
        ├─ Validate: can be opened by PyMuPDF (not corrupted)
        ├─ Assign document_id (UUID)
        ├─ Save to data/uploads/{doc_id}.pdf
        └─ Queue Celery task → return job_id instantly

Celery Worker picks up the job:

  Step 1.1 — Parse (page by page, never full file in RAM)
    Each page:
      Has selectable text? → PyMuPDF extracts directly
      Image/scanned page?  → Tesseract OCR → text
      Table detected?      → rows converted to text
      Track: page_number, section_heading, font_size

  Step 1.2 — Metadata
    Title + authors from first page
    Store: filename, page_count, upload_date, doc_type

  Step 1.3 — Chunk (4 levels)
    Level 1: 1 summary chunk per document
    Level 2: 1 chunk per section
    Level 3: paragraph chunks — 512 tokens, 64 overlap  ← main unit
    Level 4: sentence chunks  ← for exact citations
    Each chunk stores: doc_id, page_no, level, section, text
```

---

### Phase 2: Storage — Embed & Index

```
Chunks from Phase 1
        │
        ▼
Embed (sentence-transformers/all-MiniLM-L6-v2)
  Batch: 256 chunks per forward pass
  Output: 384-dim float32 vector per chunk
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
FAISS Vector Store                BM25 Keyword Index
  Stores vectors                    Stores token frequencies
  HNSW ANN index                    For exact name/code search
  Sharded per project               Pickled per project
  Written to disk every batch
        │
        ▼
SQLite / PostgreSQL
  documents: doc_id, title, pages, project_id, upload_date
  chunks:    chunk_id, doc_id, page_no, level, section, text
  (chunk text needed to show citations and pass to LLM)
```

---

### Phase 3: Retrieval — Hybrid Search

```
User Question
        │
        ▼
Embed question → 384-dim vector
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
FAISS: top-50 by cosine similarity    BM25: top-50 by keyword score
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
        Reciprocal Rank Fusion → top-100 combined
                       │
                       ▼
        Cross-encoder re-ranking → top-20 final chunks
                       │
                       ▼
        Fetch full chunk text from SQLite
        Attach: doc_title, page_no, section
```

---

### Phase 4: Reasoning — LangGraph

```
LangGraph StateGraph
─────────────────────────────────────────────────────

  START → Query Decomposer (splits into sub-questions)
       → Retriever nodes (parallel, one per sub-question)
       → Evidence Aggregator (merge, deduplicate, flag conflicts)
       → Reasoning Node (GROQ LLM synthesizes answer)
       → Sufficient? YES → Formatter → END
                    NO  → refine queries, loop back (max 3x)
```

---

### Phase 5: Response — Stream to UI

```
GROQ streams tokens → FastAPI SSE → Streamlit st.write_stream()
After stream ends:
  citations    → st.expander() per source
  confidence   → st.progress()
  follow_ups   → st.button() per suggestion
  audit entry  → written to DB silently
```

---

### Phase 6: Unique Feature Pipelines

All 5 unique features run as **separate LangGraph pipelines** triggered after indexing:

```
After all PDFs indexed:
        │
        ├─ LangGraph: Contradiction Pipeline
        │    Sample overlapping topic chunks from all docs
        │    GROQ: "Do any chunks say conflicting things?"
        │    Store results → contradictions table in DB
        │    Available via GET /contradictions
        │
        ├─ LangGraph: Knowledge Gap Pipeline
        │    Extract all topics covered (Level 1 + 2 chunks)
        │    GROQ: "What important topics are missing?"
        │    Store results → gaps table in DB
        │    Available via GET /gaps
        │
        ├─ LangGraph: Smart Questions Pipeline
        │    Sample representative chunks from all docs
        │    GROQ: "Generate 10 high-value questions for this doc set"
        │    Store results → questions table in DB
        │    Available via GET /questions
        │    Shown in Streamlit when chat is empty
        │
        ├─ LangGraph: Insights Pipeline
        │    Extract claims → find patterns → find anomalies → find trends
        │    Store results → insights table in DB
        │    Available via GET /insights
        │    Shown on project dashboard load
        │
        └─ Timeline Pipeline (on-demand)
             User types topic + clicks "Build Timeline"
             Find all dated chunks about that topic across docs
             Sort by date → GROQ narrates the evolution
             Available via GET /timeline?topic=...
```

---

## Scalability Design

| Challenge | Solution |
|---|---|
| 1000+ page PDF | PyMuPDF streams one page at a time — RAM never spikes |
| Any number of PDFs | FAISS sharded per project — unused shards stay on disk |
| Slow embedding | Celery workers batch 256 chunks — add workers to scale |
| Repeated questions | Redis cache 1-hour TTL — skips LLM entirely |
| Unique feature pipelines | Run as background Celery jobs after indexing — non-blocking |
| Contradiction scan | Only runs once per upload batch, results cached in DB |
| Many concurrent users | FastAPI async + Celery queue — no blocking |
| FAISS gets large | Switch to IndexIVFFlat (nlist=1024) for sublinear search |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | **Streamlit** | **Chat, Contradictions, Gaps, Timeline, Insights UI** |
| LLM | GROQ API (llama3-70b) | Reasoning, contradiction detection, insight generation |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Chunk + query vectorization |
| Orchestration | LangChain + LangGraph | All reasoning pipelines (Q&A + 5 unique features) |
| Vector Store | FAISS | ANN search over chunk embeddings |
| Sparse Search | BM25 (rank_bm25) | Keyword search for exact terms |
| Re-ranking | cross-encoder (HuggingFace) | Precision re-ranking |
| PDF Parsing | PyMuPDF (fitz) | Page-by-page text extraction |
| OCR | Tesseract + pdf2image | Scanned page fallback |
| Backend | FastAPI | Async REST + SSE streaming |
| Task Queue | Celery + Redis | Background ingestion + feature pipelines |
| Database | SQLite (dev) / PostgreSQL (prod) | Chunks, docs, contradictions, gaps, insights |
| Cache | Redis | Query cache + session memory |
| HTTP Client | requests / httpx | Streamlit → FastAPI |

---

## Folder Structure

```
DocuMind/
│
├── frontend/
│   ├── app.py                        # Streamlit entry point
│   ├── pages/
│   │   ├── 1_Chat.py                 # Chat + smart questions
│   │   ├── 2_Contradictions.py       # Contradiction viewer
│   │   ├── 3_Knowledge_Gaps.py       # Gap finder viewer
│   │   ├── 4_Timeline.py             # Evolution timeline
│   │   └── 5_Insights.py             # Proactive insights dashboard
│   ├── components/
│   │   ├── sidebar.py                # Project selector + uploader
│   │   ├── citation_card.py          # st.expander citation
│   │   ├── confidence_meter.py       # st.progress bar
│   │   ├── contradiction_card.py     # Contradiction display component
│   │   ├── gap_card.py               # Knowledge gap display
│   │   ├── insight_card.py           # Insight card component
│   │   └── upload_tracker.py         # Per-file progress bar
│   └── api_client.py                 # All HTTP calls in one place
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/routes/
│   │   ├── upload.py                 # POST /upload — PDF only
│   │   ├── query.py                  # POST /query — streaming
│   │   ├── documents.py              # GET / DELETE /documents
│   │   ├── jobs.py                   # GET /jobs/{job_id}
│   │   ├── sessions.py               # Conversation history
│   │   ├── contradictions.py         # GET /contradictions
│   │   ├── gaps.py                   # GET /gaps
│   │   ├── questions.py              # GET /questions
│   │   ├── timeline.py               # GET /timeline
│   │   ├── insights.py               # GET /insights
│   │   └── audit.py                  # GET /audit
│   │
│   ├── ingestion/
│   │   ├── pdf_parser.py             # PyMuPDF + OCR, page streaming
│   │   ├── chunker.py                # 4-level hierarchical chunking
│   │   ├── metadata_extractor.py     # Title, pages, doc type
│   │   └── embedder.py               # Batch embedding
│   │
│   ├── storage/
│   │   ├── vector_store.py           # FAISS shard management
│   │   ├── bm25_store.py             # BM25 per project
│   │   ├── db.py                     # SQLAlchemy setup
│   │   └── models.py                 # Document, Chunk, Contradiction,
│   │                                 # Gap, Insight, AuditLog ORM models
│   ├── retrieval/
│   │   ├── hybrid_search.py          # FAISS + BM25 + RRF
│   │   ├── reranker.py               # Cross-encoder
│   │   └── context_assembler.py      # Final context builder
│   │
│   ├── reasoning/
│   │   ├── qa_graph.py               # Q&A LangGraph pipeline
│   │   ├── contradiction_graph.py    # Contradiction detection pipeline
│   │   ├── gap_graph.py              # Knowledge gap pipeline
│   │   ├── questions_graph.py        # Smart question generation pipeline
│   │   ├── insights_graph.py         # Proactive insights pipeline
│   │   ├── timeline_graph.py         # Evolution timeline pipeline
│   │   └── state.py                  # Shared typed state schema
│   │
│   ├── memory/
│   │   ├── session_manager.py        # Redis conversation history
│   │   └── follow_up_generator.py    # Follow-up suggestions
│   │
│   └── workers/
│       ├── celery_app.py
│       └── tasks.py                  # ingest_pdf + run_feature_pipelines
│
├── data/
│   ├── uploads/                      # Raw PDFs
│   ├── indexes/                      # FAISS + BM25 per project
│   └── db/                           # SQLite (dev)
│
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/upload` | Upload PDFs (PDF only, any count) |
| GET | `/jobs/{job_id}` | Upload + indexing progress |
| POST | `/query` | Ask a question (streaming SSE) |
| GET | `/documents` | List indexed PDFs in a project |
| DELETE | `/documents/{id}` | Remove a PDF |
| GET | `/contradictions` | Get detected contradictions |
| GET | `/gaps` | Get knowledge gaps |
| GET | `/questions` | Get smart questions for doc set |
| GET | `/timeline` | Get evolution timeline for a topic |
| GET | `/insights` | Get proactive insights |
| GET | `/sessions` | Conversation history |
| GET | `/audit` | Full query audit log |

---

## Setup & Run

---

### Option A — Run Locally (Recommended for Development)

#### Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Python | 3.10+ | [python.org](https://python.org) |
| Redis | 7+ | See below |
| Tesseract OCR | any | See below |
| GROQ API Key | — | [console.groq.com](https://console.groq.com) (free) |

**Install Redis:**
```bash
# Windows (using WSL or Chocolatey)
choco install redis-64
# OR use the Windows Redis port: https://github.com/microsoftarchive/redis/releases

# macOS
brew install redis

# Linux (Ubuntu/Debian)
sudo apt-get install redis-server
```

**Install Tesseract (for scanned PDFs):**
```bash
# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH after install

# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr
```

---

#### Step 1 — Clone & Create Virtual Environment
```bash
cd Desktop
# The project folder already exists at Multiple_Pdf_Upload_RAG
cd Multiple_Pdf_Upload_RAG

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

#### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```
> This installs: FastAPI, LangChain, LangGraph, GROQ, sentence-transformers, FAISS, BM25, Celery, Redis, Streamlit, PyMuPDF, SQLAlchemy, and all other dependencies.

#### Step 3 — Configure Environment
Open the `.env` file and make sure it has your GROQ API key:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
REDIS_URL=redis://localhost:6379
DATABASE_URL=sqlite:///./data/db/documind.db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=64
MAX_CHUNKS_RETRIEVED=20
UPLOAD_DIR=data/uploads
INDEX_DIR=data/indexes
DB_DIR=data/db
```

#### Step 4 — Create Data Directories
```bash
# Windows
mkdir data\uploads data\indexes data\db

# macOS / Linux
mkdir -p data/uploads data/indexes data/db
```

#### Step 5 — Start All 4 Services (4 separate terminals)

Open **4 terminal windows** in the project folder, run one command in each:

**Terminal 1 — Redis (message broker)**
```bash
redis-server
```
You should see: `Ready to accept connections`

**Terminal 2 — FastAPI Backend (port 8000)**
```bash
# Make sure venv is activated
uvicorn app.main:app --reload --port 8000
```
You should see: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 3 — Celery Worker (PDF background processing)**
```bash
# Make sure venv is activated
celery -A app.workers.celery_app worker --loglevel=info --pool=threads --concurrency=4
```
You should see: `celery@... ready`

**Terminal 4 — Streamlit Frontend (port 8501)**
```bash
# Make sure venv is activated
streamlit run frontend/app.py --server.port 8501
```
You should see: `You can now view your Streamlit app in your browser`

#### Step 6 — Open in Browser
```
http://localhost:8501
```

---

### Option B — Run with Docker (One Command)

#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

#### Run
```bash
# Build and start all 4 services (Redis, FastAPI, Celery, Streamlit)
docker compose up --build
```

First run downloads models (~500 MB) — takes 3-5 minutes.

#### Open in Browser
```
http://localhost:8501
```

#### Stop
```bash
docker compose down
```

#### Stop and delete all data
```bash
docker compose down -v
```

---

### Verify Everything Is Working

After starting, check these URLs in your browser:

| URL | Expected Result |
|---|---|
| `http://localhost:8000/health` | `{"status": "ok"}` |
| `http://localhost:8000/docs` | FastAPI Swagger UI (all endpoints listed) |
| `http://localhost:8501` | DocuMind Streamlit app |

The sidebar should show **● API Online** (green) — if it shows **● API Offline** (red), the FastAPI backend is not running.

---

### Quick Test (Verify the Full Pipeline Works)

1. Open `http://localhost:8501`
2. In the sidebar — **Project ID**: type `test-project`
3. Upload any PDF file using the file uploader
4. Watch the progress bar go 0% → 100% (takes 5-30s depending on PDF size)
5. Once indexed (green checkmark), type a question in the chat
6. You should get a streamed answer with citations (page numbers + text snippets)
7. Click **⚡ Contradictions** tab → **Scan for Contradictions**
8. Click **💡 Smart Questions** tab → **Generate Smart Questions**

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `API Offline` in sidebar | Start FastAPI: `uvicorn app.main:app --reload --port 8000` |
| `redis.exceptions.ConnectionError` | Start Redis: `redis-server` |
| PDF stuck at 0% | Check Celery terminal — worker must be running |
| `ModuleNotFoundError` | Activate venv and run `pip install -r requirements.txt` |
| Tesseract not found | Install Tesseract and add to system PATH |
| Port 8000 already in use | `uvicorn app.main:app --reload --port 8001` and update `BACKEND_URL` in `.env` |
| SQLite database error | Run `mkdir -p data/db` to create the directory |

---

## Why Any Topic, Any Pages, Any Number of PDFs

- **Any topic:** Uses general-purpose embeddings (sentence-transformers) — not domain-specific. Same system works for legal, medical, finance, engineering, or any other field.
- **Any number of pages:** PyMuPDF streams one page at a time. Peak RAM = 1 page. Always.
- **Any number of PDFs:** Each PDF is an independent Celery task. FAISS shards scale to millions of chunks. Query speed stays O(log n) regardless of how many docs are indexed.
