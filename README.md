# 🤖 ITAssist AI — Service Desk Copilot

> AI-powered IT support workflow built with **Streamlit + LangGraph + FAISS + Groq**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3--8b-orange)](https://console.groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector--DB-blue)](https://faiss.ai)

---

## 🎯 What It Does

ITAssist AI eliminates the back-and-forth in IT support tickets. Instead of:

```
User submits vague ticket → IT asks 10 questions → User responds → Ticket finally useful
```

It delivers:

```
User describes issue → AI classifies + searches Microsoft SOP docs → 
AI asks 2-3 diagnostic questions → AI analyzes root cause → 
Suggests self-resolution → Creates enriched ticket only if needed
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│   User Portal (Chat)          IT Engineer Dashboard          │
└────────────────────┬────────────────────────────────────────┘
                     │
            ┌────────▼────────┐
            │   LangGraph     │
            │  State Machine  │
            └────────┬────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌─────────────┐
   │ Intake  │  │Knowledge│  │Troubleshoot │
   │ Agent   │  │ Agent   │  │  Agent      │
   │(Groq    │  │(FAISS   │  │(Groq LLM)  │
   │ LLM)    │  │ RAG)    │  │             │
   └─────────┘  └─────────┘  └─────────────┘
                     │
              ┌──────▼──────┐
              │  FAISS DB   │
              │ (Microsoft  │
              │   PDF +     │
              │ SOP Docs)   │
              └─────────────┘
                     │
              ┌──────▼──────┐
              │   SQLite    │
              │  (Tickets,  │
              │  Convos)    │
              └─────────────┘
```

---

## 📁 Project Structure

```
itassist-ai/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env.example                    # Copy to .env and add GROQ_API_KEY
│
├── agents/
│   ├── graph.py                    # LangGraph state machine
│   ├── intake_agent.py             # Issue classification (Groq LLM)
│   ├── knowledge_agent.py          # RAG search (FAISS)
│   ├── troubleshoot_agent.py       # Diagnostic Q&A + root cause
│   └── resolution_agent.py        # Resolution steps + PowerShell scripts
│
├── rag/
│   ├── ingest.py                   # Build FAISS index from PDF + SOPs
│   ├── retriever.py                # Semantic search
│   └── sop_documents/              # IT SOP text documents
│       ├── performance_sop.txt
│       ├── networking_vpn_sop.txt
│       ├── printing_sop.txt
│       ├── security_bitlocker_sop.txt
│       └── backup_storage_sop.txt
│
├── database/
│   ├── models.py                   # SQLAlchemy ORM (4 tables)
│   └── crud.py                     # CRUD operations
│
├── ui/
│   ├── user_portal.py              # Interactive user chat (7 stages)
│   ├── it_dashboard.py             # IT engineer dashboard (4 tabs)
│   └── components.py               # Reusable UI components
│
└── config/
    └── settings.py                 # Centralized configuration
```

---

## 🚀 Setup & Run

### Step 1: Install Dependencies

```bash
cd itassist-ai
pip install -r requirements.txt
```

### Step 2: Configure API Key

```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your Groq API key
# Get free key at: https://console.groq.com
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 3: Build Knowledge Base (RAG Index)

Place the PDF at the path configured in `rag/ingest.py`, then run:

```bash
python rag/ingest.py
```

This will:
- Load the Microsoft Windows troubleshooting PDF (68 MB)
- Load all 5 SOP text documents
- Create FAISS vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Save the index to `faiss_index/`

⏱️ *First run takes 3-8 minutes depending on hardware.*

### Step 4: Start the App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🎭 Usage

### As a User:
1. Login with any username → select **"User"** role
2. Click **"Chat with AI"**
3. Describe your IT issue (or click a quick example)
4. Answer the AI's 2-3 diagnostic questions
5. Try the suggested self-resolution steps
6. If unresolved → approve the AI-generated ticket

### As an IT Engineer:
1. Login → select **"IT Engineer"** role
2. Go to **"IT Dashboard"**
3. View open tickets with full AI analysis
4. Read the generated PowerShell diagnostic scripts
5. Update ticket status and add resolution notes

---

## 🤖 The 4 AI Agents

| Agent | Input | Output |
|-------|-------|--------|
| **Intake Agent** | User message | Category + confidence score |
| **Knowledge Agent** | Category + message | Relevant SOP excerpts (FAISS RAG) |
| **Troubleshoot Agent** | Category + SOP context | Diagnostic questions → Root cause analysis |
| **Resolution Agent** | Ticket + root cause | Resolution steps + PowerShell script |

---

## 📚 Knowledge Base Sources

- **Microsoft Windows Client Troubleshooting PDF** (68 MB) — from Microsoft Learn
- **Performance SOP** — Slow performance, CPU/RAM/Disk diagnostics, Blue Screen
- **Networking & VPN SOP** — TCP/IP, DNS, DHCP, VPN errors, wireless
- **Printing SOP** — Print spooler, driver issues, network printers
- **Security & BitLocker SOP** — Recovery passwords, TPM, Credential Guard
- **Backup & Storage SOP** — VSS, disk management, OneDrive, file history

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Orchestration | LangGraph |
| LLM | Groq (llama3-8b-8192) — **Free tier** |
| Vector DB | FAISS (local, no server) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 — **Free, local** |
| Database | SQLite (zero setup) |
| Language | Python 3.11+ |

---

## 🔑 Environment Variables

```env
GROQ_API_KEY=gsk_...      # Required — get at console.groq.com (free)
GROQ_MODEL=llama3-8b-8192 # Optional — default model
```

---

*Built as a LinkedIn portfolio project demonstrating Agentic AI + RAG + IT Support workflow.*
