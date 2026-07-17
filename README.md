# ITAssist AI Service Desk Copilot

ITAssist AI is an automated IT support tool built with Streamlit, LangGraph, FAISS, and Groq. It helps users resolve common IT issues by analyzing their requests, searching documentation, asking diagnostic questions, and suggesting self-resolution steps. If the issue can't be resolved automatically, it creates an enriched ticket for the IT team.

## Project Structure
- `app.py`: The main Streamlit application
- `agents/`: The AI agents that handle classification, search, troubleshooting, and resolution
- `rag/`: The knowledge base ingestion and semantic search tools
- `database/`: Database models and operations
- `ui/`: The user and engineer portal interfaces
- `utils/guardrail.py`: The safety scanner checking both inputs and outputs
- `config/`: Configuration settings

## Setup Instructions

1. **Install Dependencies**
   Run this in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   Copy `.env.example` to `.env` and add your Groq API key:
   ```env
   GROQ_API_KEY=your_key_here
   ```

3. **Build the Knowledge Base**
   Run the ingestion script to build the local search index:
   ```bash
   python rag/ingest.py
   ```

4. **Start the Application**
   Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Then open your browser to `http://localhost:8501`.

---

## Security & Safety Guardrails

To prevent jailbreaks, prompt injections, and accidental credential leaks, the app has a built-in safety guardrail system that runs in multiple layers:

*   **Input Normalization & Regex (Layer 1)**: Automatically cleanses text, decodes URL/Base64/Hex encodings, and checks for API keys or credentials using strict code rules before calling any AI.
*   **AI Safety Classifier (Layer 2)**: Uses Llama 3.1 8B to check if the query is relevant to IT support and blocks off-topic requests (like coding, travel advice, or recipes), hacking requests, and jailbreak tricks.
*   **Output Moderation (Layer 3)**: Automatically scans the generated troubleshooting steps or support emails before displaying them to ensure no passwords or restricted command references leak out.
*   **Input Truncation Protection (Layer 4)**: Aligns character limits so the safety check and main agent see the exact same truncated input, neutralizing truncation bypasses.

---

## How to Use

### For Users:
Log in as a **User**, open the chat, and describe your IT issue. Answer any diagnostic questions the AI asks, and try the suggested steps. If it doesn't work, the AI will create a ticket for you.

### For IT Engineers:
Log in as an **IT Engineer** and go to the **IT Dashboard**. You can view open tickets, read the AI's analysis and recommended scripts, and update the ticket status.
