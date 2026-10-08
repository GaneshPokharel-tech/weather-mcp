# Multi MCP Agent

A production-oriented learning project that demonstrates how an AI agent can use multiple real tools through a **single Model Context Protocol (MCP) server**.

The application combines:

- Streamlit
- Google Gemini
- LangGraph
- LangChain MCP
- MCP Python SDK
- WeatherAPI
- Gmail API
- Google Calendar API
- Git
- YouTube
- Public URL fetching
- Advanced local PDF RAG
- Human approval for protected actions
- Persistent conversation memory

The goal of the project is to build a real multi-tool AI agent rather than a simple chatbot.

---

# Project Goal

The main goal is to understand and implement the complete agentic MCP workflow:

```text
User
 ↓
Streamlit UI
 ↓
LangGraph Agent
 ↓
Google Gemini
 ↓
Tool Decision
 ↓
MCP Adapter
 ↓
Single MCP Server
 ↓
External Service / Local Tool / RAG Engine
 ↓
Tool Result
 ↓
Gemini
 ↓
Final Response
```

The LLM does not directly communicate with external services.

Instead, the MCP server acts as the controlled interface between the AI agent and the available capabilities.

---

# Architecture

```text
┌──────────────────────────────────────┐
│                USER                  │
│        Streamlit Web Interface       │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│               app.py                 │
│                                      │
│ • Chat UI                            │
│ • Light / Dark mode                  │
│ • Chat history                       │
│ • PDF upload                         │
│ • Automatic PDF indexing             │
│ • Indexed document management        │
│ • Human approval UI                  │
│ • Streaming responses                │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│              agent.py                │
│                                      │
│ • Google Gemini                      │
│ • LangGraph                          │
│ • MCPAdapter                         │
│ • Tool selection                     │
│ • Human approval routing             │
│ • Conversation memory                │
│ • Retry / timeout handling           │
│ • Streaming                          │
└──────────────────┬───────────────────┘
                   │
                   │ MCP
                   ▼
┌──────────────────────────────────────┐
│           mcp_server.py              │
│                                      │
│ Weather                              │
│ Gmail                                │
│ Google Calendar                      │
│ Fetch                                │
│ Git                                  │
│ YouTube                              │
│ PDF RAG                              │
└───────┬─────────┬─────────┬──────────┘
        │         │         │
        ▼         ▼         ▼
   WeatherAPI   Google    Git / Web /
                APIs       YouTube
                              │
                              │ RAG tools
                              ▼
                  ┌─────────────────────┐
                  │       rag.py        │
                  │                     │
                  │ PDF extraction      │
                  │ Smart chunking      │
                  │ Local embeddings    │
                  │ Dense retrieval     │
                  │ BM25                │
                  │ RRF                 │
                  │ MMR                 │
                  │ SQLite index        │
                  └─────────────────────┘
```

---

# Main Files

## `app.py`

The Streamlit frontend of the application.

Responsibilities:

- renders the web interface
- manages light and dark mode
- displays conversations
- sends user messages to the agent
- displays streamed agent responses
- shows human approval requests
- manages chat threads
- uploads PDF documents
- automatically indexes uploaded PDFs
- displays indexed documents
- allows document removal
- shows connected capabilities and tool activity

In simple terms:

```text
app.py = User Interface
```

---

## `agent.py`

The main reasoning and orchestration layer.

Responsibilities:

- initializes Google Gemini
- connects to the MCP server through `MCPAdapter`
- discovers MCP tools
- binds tools to the language model
- builds the LangGraph workflow
- decides whether a tool should execute
- routes protected actions through human approval
- manages persistent conversation history
- manages thread IDs
- streams responses to Streamlit
- handles Gemini retries and timeouts
- injects the current date/time for Calendar requests
- injects the active conversation thread ID for RAG requests
- enforces tool-grounded behavior

In simple terms:

```text
agent.py = Brain / Orchestrator
```

---

## `mcp_server.py`

The single MCP server used by the entire project.

All tools are registered in this file.

It connects the agent to:

- WeatherAPI
- Gmail
- Google Calendar
- public websites
- the local Git repository
- YouTube
- the local RAG engine

In simple terms:

```text
mcp_server.py = Tool Server
```

The project intentionally uses **one MCP server** rather than creating a separate MCP server for every capability.

---

## `rag.py`

The internal PDF Retrieval-Augmented Generation engine.

It is not a separate MCP server.

The MCP RAG tools in `mcp_server.py` call functions inside `rag.py`.

Responsibilities:

- safely stores uploaded PDFs
- extracts PDF text
- removes repeated page noise
- performs page-aware smart chunking
- generates local embeddings
- stores document data in SQLite
- performs semantic retrieval
- performs BM25 keyword retrieval
- combines retrieval results using Reciprocal Rank Fusion
- reranks results using MMR
- returns document name and page information
- keeps documents isolated by conversation thread
- supports multiple PDFs

In simple terms:

```text
rag.py = PDF Knowledge + Retrieval Engine
```

---

## `__init__.py`

Marks `multi_mcp_agent` as a Python package.

This allows imports such as:

```python
from multi_mcp_agent.agent import AgentRuntime
from multi_mcp_agent.rag import search_documents
```

---

# Project Structure

```text
multi-mcp-agent/
│
├── .env
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
├── uv.lock
│
├── .rag/                              # ignored local RAG data
├── .langgraph_checkpoints.sqlite      # ignored conversation memory
├── .gmail_token.json                  # ignored Gmail OAuth token
├── .calendar_token.json               # ignored Calendar OAuth token
│
└── src/
    └── multi_mcp_agent/
        ├── __init__.py
        ├── agent.py
        ├── app.py
        ├── mcp_server.py
        └── rag.py
```

Runtime files containing private data are excluded from Git.

---

# Technologies Used

- Python 3.12
- Streamlit
- MCP Python SDK
- LangChain
- LangGraph
- Google Gemini
- Google Gmail API
- Google Calendar API
- WeatherAPI.com
- HTTPX
- Git
- yt-dlp
- PyPDF
- Sentence Transformers
- `intfloat/multilingual-e5-small`
- Rank BM25
- NumPy
- SQLite
- python-dotenv
- uv
- MCP Inspector

---

# MCP Tools

The unified MCP server currently exposes approximately **24 tools**.

## Weather

```text
get_current_weather
get_weather_forecast
```

### `get_current_weather`

Gets live weather information for a location.

Example request:

```text
What's the current weather in Kathmandu?
```

Flow:

```text
User
 ↓
Agent
 ↓
get_current_weather
 ↓
MCP Server
 ↓
WeatherAPI
 ↓
Agent
 ↓
Final weather response
```

### `get_weather_forecast`

Gets a short-term forecast.

Example:

```text
Will I need an umbrella tomorrow in Kathmandu?
```

---

# Gmail

```text
search_emails
get_email
create_draft
```

## `search_emails`

Searches the authenticated Gmail inbox.

Example:

```text
Show me my 3 most recent emails.
```

## `get_email`

Retrieves the full content of a selected email.

## `create_draft`

Creates an email draft.

It does **not send email**.

Example:

```text
Create a draft email to example@example.com.
```

---

# Google Calendar

```text
list_calendar_events
create_calendar_event
update_calendar_event
delete_calendar_event
```

The Calendar integration supports natural language dates.

Example:

```text
Create an event tomorrow at 5 PM called Project Review.
```

The agent receives the actual current date and timezone at runtime before resolving terms such as:

```text
today
tomorrow
yesterday
next Monday
next week
```

The default timezone is:

```text
Asia/Kathmandu
```

unless configured otherwise.

---

# Fetch

```text
fetch_url
```

Retrieves text from a public URL.

Example:

```text
Summarize https://example.com/article
```

The Fetch tool includes network safety checks and does not allow requests to private or internal network addresses.

---

# Git

The project includes Git tools for the current repository.

Read-only tools:

```text
git_status
git_log
git_diff
git_show
git_branch
```

Write / modifying tools:

```text
git_add
git_commit
git_create_branch
git_checkout
```

The modifying Git tools require human approval before execution.

Example:

```text
User
 ↓
"Stage .gitignore"
 ↓
Agent selects git_add
 ↓
Human Approval
 ↓
Approve
 ↓
git_add executes
```

---

# YouTube

```text
play_youtube_song
```

Searches YouTube using `yt-dlp` and opens the matching video in the default browser.

Example:

```text
Play Bohemian Rhapsody on YouTube.
```

The tool does not download audio or video.

---

# PDF RAG

The project includes an advanced local PDF RAG system.

MCP tools:

```text
index_document
search_documents
list_documents
delete_document
```

PDF files can be uploaded directly through the Streamlit interface.

Indexing starts automatically after upload.

---

# RAG Pipeline

```text
PDF Upload
   ↓
PDF Text Extraction
   ↓
Text Cleaning
   ↓
Page-Aware Chunking
   ↓
Local Embeddings
   ↓
SQLite Index
```

When the user asks a question:

```text
Question
   │
   ├── Dense Semantic Retrieval
   │
   └── BM25 Keyword Retrieval
            ↓
            RRF
            ↓
            MMR
            ↓
        Best Chunks
            ↓
   Filename + Page Number
            ↓
          Gemini
            ↓
      Grounded Answer
```

---

# Local Embeddings

PDF embeddings are generated locally using:

```text
intfloat/multilingual-e5-small
```

Embedding dimension:

```text
384
```

The project originally experimented with Gemini embeddings, but local embeddings are now used to avoid cloud embedding rate limits and make document indexing more reliable.

Benefits:

- no embedding API quota
- local document vector generation
- multilingual retrieval
- better privacy for document indexing
- predictable indexing behavior

---

# Hybrid Retrieval

The RAG engine combines multiple retrieval techniques.

## Dense Retrieval

Finds semantically similar passages using embedding similarity.

Example:

```text
Question:
What problems does Naive RAG have?
```

Dense retrieval can find related passages even if the exact wording is different.

---

## BM25

Provides lexical / keyword search.

This is useful when exact terminology, identifiers, or specific phrases are important.

---

## Reciprocal Rank Fusion

Dense and BM25 rankings are combined using:

```text
RRF
```

This allows semantic retrieval and keyword retrieval to contribute to the final ranking.

---

## MMR

Maximum Marginal Relevance is used to improve result diversity and reduce repetitive chunks.

Final retrieval architecture:

```text
Dense
+
BM25
+
RRF
+
MMR
```

---

# PDF Grounding

Document questions must be answered from retrieved PDF evidence.

Example:

```text
What are the three main RAG paradigms?
```

Possible result:

```text
1. Naive RAG
2. Advanced RAG
3. Modular RAG

[rag_survey.pdf, page 2]
```

If sufficient supporting evidence cannot be retrieved, the agent responds with an insufficient-evidence message instead of filling the answer with general model knowledge.

Example:

```text
I couldn't find sufficient evidence in the uploaded document to answer that.
```

---

# Page Citations

Retrieved document answers include user-facing citations such as:

```text
[rag_survey.pdf, page 3]
```

Internal hashed storage filenames are not exposed to the user.

---

# Multiple PDF Support

Multiple PDFs can be indexed in the same conversation.

Example:

```text
rag_survey.pdf
Troubleshooting SOP.pdf
```

The user can explicitly select one document:

```text
Using only rag_survey.pdf, what are the three RAG paradigms?
```

The agent:

```text
1. calls list_documents
2. resolves the requested filename
3. obtains its document ID
4. searches only that document
5. returns evidence only from that PDF
```

This prevents cross-document retrieval mixing.

---

# Thread Isolation

Documents are scoped to conversation threads.

```text
Chat A
 ├── document-a.pdf
 └── document-b.pdf

Chat B
 └── document-c.pdf
```

A document indexed in Chat A is not automatically treated as part of Chat B.

This allows each conversation to maintain its own document workspace.

---

# Follow-Up PDF Questions

Conversation context is preserved.

Example:

```text
User:
What are the three main RAG paradigms?

Agent:
Naive RAG, Advanced RAG, Modular RAG.

User:
What are the limitations of the first one?
```

The agent understands that:

```text
"the first one" = Naive RAG
```

but still performs a new document search before answering.

---

# Human Approval

Protected actions are routed through a LangGraph approval node.

Examples include:

```text
git_add
git_commit
git_create_branch
git_checkout

create_calendar_event
update_calendar_event
delete_calendar_event

delete_document
```

Workflow:

```text
User request
     ↓
Gemini selects protected tool
     ↓
LangGraph detects protected action
     ↓
Approval Node
     ↓
Streamlit displays action + arguments
     ↓
        ┌──────────┴──────────┐
        │                     │
     Approve                Cancel
        │                     │
        ▼                     ▼
   Tool executes        Tool rejected
```

The agent cannot silently execute protected chat actions.

---

# LangGraph Workflow

The graph contains three main nodes:

```text
agent
approval
tools
```

Flow:

```text
START
  ↓
agent
  ↓
route_after_agent()
  │
  ├── no tool call
  │      ↓
  │     END
  │
  ├── safe tool
  │      ↓
  │    tools
  │      ↓
  │    agent
  │
  └── protected tool
         ↓
      approval
         ↓
      approve?
       /     \
     yes      no
      │        │
    tools    agent
      │
    agent
```

---

# Conversation Memory

Conversation history is stored locally using:

```text
LangGraph
+
AsyncSqliteSaver
+
SQLite
```

Database:

```text
.langgraph_checkpoints.sqlite
```

Each conversation uses its own:

```text
thread_id
```

This enables:

- persistent chat history
- switching between conversations
- follow-up questions
- human approval resume
- thread-specific PDF context

---

# Tool Selection

Gemini receives:

```text
User request
+
System instructions
+
Available MCP tool schemas
+
Conversation context
```

The model then decides whether a tool is required.

Example:

```text
User:
What's the weather in Kathmandu?

Gemini:
→ get_current_weather
```

Another example:

```text
User:
Show my recent emails.

Gemini:
→ search_emails
```

Another example:

```text
User:
What does rag_survey.pdf say about Naive RAG?

Gemini:
→ list_documents
→ search_documents
```

---

# Capability Boundary

The application is intentionally tool-grounded.

Its supported knowledge sources are:

```text
Current conversation
Uploaded PDF documents
Weather
Gmail
Google Calendar
Public URL Fetch
Git
YouTube
```

The application is not designed as an unrestricted general-purpose chatbot.

Unsupported unrelated general-knowledge or creative-writing requests may be rejected.

This reduces unsupported answers and keeps responses grounded in the capabilities available to the application.

---

# Error Handling

The agent handles common model and tool failures.

Examples include:

```text
Gemini quota errors
Gemini service unavailable
Model timeout
WeatherAPI errors
Invalid weather location
Ambiguous weather location
Fetch network errors
OAuth errors
PDF extraction errors
RAG indexing errors
```

Model requests include retry handling for transient failures.

Current model timeout:

```text
35 seconds
```

Maximum full agent turn timeout:

```text
90 seconds
```

---

# Security

Sensitive configuration is not hard-coded.

The project uses:

```text
.env
```

for API credentials.

Local OAuth tokens are stored in:

```text
.gmail_token.json
.calendar_token.json
```

These files are ignored by Git.

Uploaded PDFs and local RAG indexes are stored under:

```text
.rag/
```

and are also ignored by Git.

Persistent conversation data is stored locally and ignored by Git.

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```env
WEATHER_API_KEY=your_weatherapi_key

GEMINI_API_KEY=your_gemini_api_key

GMAIL_OAUTH_CLIENT_ID=your_google_oauth_client_id
GMAIL_OAUTH_CLIENT_SECRET=your_google_oauth_client_secret

CALENDAR_TIMEZONE=Asia/Kathmandu
```

Do not commit `.env`.

---

# `.gitignore`

The project ignores local credentials, private documents, generated databases, environments, and temporary files.

Important entries include:

```gitignore
# Python cache / build
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
wheels/
.pytest_cache/
.ruff_cache/

# Virtual environment
.venv/

# Secrets / local credentials
.env
.gmail_token.json
.calendar_token.json

# LangGraph local persistence
.langgraph_checkpoints.sqlite
.langgraph_checkpoints.sqlite-shm
.langgraph_checkpoints.sqlite-wal

# Local RAG data
.rag/

# Local editor / OS files
.vscode/
.DS_Store

# Temporary backups
*.bak
*.prompt-backup
```

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd multi-mcp-agent
```

---

## 2. Install dependencies

The project uses `uv`.

```bash
uv sync
```

---

## 3. Activate the environment

On macOS/Linux:

```bash
source .venv/bin/activate
```

---

## 4. Configure `.env`

Create:

```text
.env
```

and add the required credentials.

---

# Run the Application

Start the Streamlit application:

```bash
uv run streamlit run src/multi_mcp_agent/app.py
```

Then open the local Streamlit URL in the browser.

---

# Test the MCP Server

Run MCP Inspector:

```bash
uv run mcp dev src/multi_mcp_agent/mcp_server.py
```

The server should expose the multi-capability MCP tools.

Current groups:

```text
Weather
Gmail
Google Calendar
Fetch
Git
YouTube
RAG
```

---

# Syntax Check

Before running the project, the main source files can be checked with:

```bash
uv run python -m py_compile \
  src/multi_mcp_agent/agent.py \
  src/multi_mcp_agent/app.py \
  src/multi_mcp_agent/mcp_server.py \
  src/multi_mcp_agent/rag.py
```

No output means the files compiled successfully.

---

# Example: Weather Flow

```text
User:
What's the current weather in Kathmandu?

        ↓

app.py

        ↓

agent.py

        ↓

Gemini decides:
get_current_weather

        ↓

LangGraph

        ↓

MCPAdapter

        ↓

mcp_server.py

        ↓

WeatherAPI

        ↓

Weather result

        ↓

agent.py

        ↓

Gemini final response

        ↓

app.py

        ↓

User
```

---

# Example: PDF Flow

## Indexing

```text
User uploads PDF
       ↓
app.py
       ↓
save_uploaded_file()
       ↓
.rag/uploads/
       ↓
index_document MCP tool
       ↓
mcp_server.py
       ↓
rag.py
       ↓
PDF extraction
       ↓
Smart chunking
       ↓
Local embeddings
       ↓
SQLite RAG index
```

## Question Answering

```text
User asks PDF question
       ↓
app.py
       ↓
agent.py
       ↓
Gemini selects search_documents
       ↓
mcp_server.py
       ↓
rag.py
       ↓
Dense + BM25
       ↓
RRF
       ↓
MMR
       ↓
Relevant chunks + pages
       ↓
agent.py
       ↓
Grounded final answer
       ↓
[filename.pdf, page N]
```

---

# Example: Calendar Write Flow

```text
User:
Create an event tomorrow at 5 PM.

        ↓

agent.py

        ↓

Current date/time injected

        ↓

Gemini resolves actual date

        ↓

create_calendar_event

        ↓

Protected action detected

        ↓

Human Approval

        ↓

User approves

        ↓

mcp_server.py

        ↓

Google Calendar API

        ↓

Event created

        ↓

Final confirmation
```

---

# Streamlit Interface

The current UI includes:

- responsive web layout
- light mode
- dark mode
- new chat
- saved conversations
- PDF uploader
- automatic PDF indexing
- multiple document support
- indexed document metadata
- document removal
- capability cards
- AI thinking state
- streamed responses
- tool usage information
- human approval controls

---

# Current Capabilities

```text
🌤 Weather
✉ Gmail
📅 Google Calendar
📄 PDF RAG
🌐 Fetch
⑂ Git
▶ YouTube
🧠 Persistent Memory
```

---

# Current Project Status

## Completed

```text
✓ Single unified MCP server
✓ Streamlit web interface
✓ Light / dark mode
✓ Google Gemini integration
✓ LangGraph agent workflow
✓ MCPAdapter integration
✓ Automatic tool selection
✓ Streaming responses
✓ Persistent chat history
✓ Multiple conversation threads
✓ Human approval workflow

✓ Weather current conditions
✓ Weather forecasting
✓ Weather location resolution
✓ Gmail search
✓ Gmail message retrieval
✓ Gmail draft creation
✓ Google Calendar listing
✓ Calendar event creation
✓ Calendar event update
✓ Calendar event deletion
✓ Runtime-aware relative date handling
✓ Public URL fetching
✓ Network safety validation
✓ Git read tools
✓ Git write tools with approval
✓ YouTube search / browser opening

✓ PDF upload
✓ Automatic PDF indexing
✓ Multiple PDF support
✓ Thread-scoped documents
✓ Local multilingual embeddings
✓ Dense semantic retrieval
✓ BM25 retrieval
✓ Reciprocal Rank Fusion
✓ MMR reranking
✓ Page-aware chunks
✓ Source/page citations
✓ Explicit per-document retrieval
✓ Follow-up PDF questions
✓ Insufficient-evidence guard
✓ General capability boundary

✓ Local secrets ignored by Git
✓ Local RAG data ignored by Git
✓ Persistent LangGraph SQLite memory
✓ Retry and timeout handling
```

---

# Known Limitations

The current project intentionally has several limitations:

- Gemini API response latency can vary
- the first local embedding model load may take longer because the model must be downloaded
- scanned/image-only PDFs do not yet have an OCR pipeline
- Gmail supports draft creation but not automatic sending
- Fetch supports public URLs only
- YouTube functionality opens the browser rather than controlling media playback internally
- RAG indexes are local to the machine
- the project is primarily designed for local learning and portfolio demonstration
- protected actions depend on the human approval flow
- the system is intentionally not an unrestricted general-purpose chatbot

---

# Key Learning Outcomes

This project demonstrates how to build a real tool-calling AI system where:

```text
LLM
 ≠ direct API access
```

Instead:

```text
LLM
 ↓
Agent
 ↓
MCP
 ↓
Controlled Tools
 ↓
External Systems
```

It demonstrates:

- MCP tool design
- one-server multi-tool architecture
- LangChain MCP integration
- LangGraph orchestration
- LLM tool selection
- human-in-the-loop approval
- persistent conversation state
- OAuth-based integrations
- API safety
- local Git automation
- local hybrid RAG
- multi-document retrieval
- source-grounded answers
- Streamlit agent UI design

---

# File Responsibilities — Quick Explanation

For demonstrations or presentations:

```text
app.py
= Frontend / UI

agent.py
= Brain / Orchestration

mcp_server.py
= All MCP Tools / Execution Layer

rag.py
= PDF Indexing + Retrieval Engine

__init__.py
= Python Package Setup
```

Complete connection:

```text
User
 ↓
app.py
 ↓
agent.py
 ↓
MCP
 ↓
mcp_server.py
 ↓
External APIs / Git / YouTube / rag.py
 ↓
Result
 ↓
agent.py
 ↓
app.py
 ↓
User
```

---

# Summary

`Multi MCP Agent` evolved from a basic weather MCP learning project into a multi-capability agentic application.

The final architecture separates concerns clearly:

```text
UI
↓
Agent Orchestration
↓
MCP
↓
Tool Execution
↓
External Services / Local RAG
```

This keeps the project modular, understandable, safer, and easier to extend with additional MCP capabilities in the future.