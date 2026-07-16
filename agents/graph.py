"""
Dynamic Cross-Questioning RAG + Groq Workflow Runner for ITAssist AI.
Uses a two-step verification flow:
1. Initial user issue -> Local search -> Generate dynamic cross-questions.
2. User answers -> Write session_finding.txt -> Call Groq for final resolution.
"""
import os
import re
from typing import TypedDict, Annotated
import operator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import GROQ_API_KEY, GROQ_MODEL
from agents.intake_agent import classify_issue
from rag.retriever import search_text

# ── State Definition ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_message: str
    session_id: str
    category: str
    intake_confidence: float
    issue_summary: str
    sop_context: str
    diagnostic_questions: str  # The dynamic cross-questions asked
    user_answers: str          # User's response to cross-questions
    analysis_text: str         # Final troubleshooting instructions
    problem_title: str
    probable_cause: str
    self_resolution_steps: list[str]
    conversation_history: list[dict]
    ticket_id: str
    resolution_notes: str      # Stores the email draft / supportive response


def _build_llm(temperature: float = 0.2, max_tokens: int = 1024) -> ChatGroq:
    import os
    key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    return ChatGroq(
        api_key=key,
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_intake_and_knowledge(user_message: str, session_id: str) -> AgentState:
    """
    Step 1: User problem input -> Local FAISS search -> Generate dynamic cross-questions.
    """
    # 1. Intake classification
    classification = classify_issue(user_message)
    category = classification.get("category", "Other")
    confidence = classification.get("confidence", 0.5)

    # 2. Perform semantic search (RAG) on local FAISS index
    rag_context = search_text(user_message)

    # 3. Call Groq to generate 1-2 targeted diagnostic cross-questions based on SOP
    llm = _build_llm(temperature=0.3)
    system_prompt = (
        "You are an IT support intake assistant. Analyze the user problem and the matched internal documentation findings. "
        "Your task is to generate 1 or 2 targeted diagnostic cross-questions to help narrow down the root cause and get specific details (e.g. error codes, OS, specific settings, symptoms).\n"
        "Keep the questions highly direct, short, conversational, and easy for a non-technical user. Ask ONLY the questions, do not add introductory greetings like 'Sure' or concluding notes."
    )

    prompt = f"""User problem: {user_message}
Matched SOP Findings: {rag_context}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        questions = response.content.strip()
    except Exception as e:
        print(f"Failed to generate dynamic questions via Groq: {e}")
        # Local fallback questions based on category
        from agents.troubleshoot_agent import get_local_category_questions
        fallback_questions = get_local_category_questions(category)
        questions = "\n".join([f"- {q}" for q in fallback_questions])

    state: AgentState = {
        "user_message": user_message,
        "session_id": session_id,
        "category": category,
        "intake_confidence": confidence,
        "issue_summary": user_message[:100],
        "sop_context": rag_context,
        "diagnostic_questions": questions,
        "user_answers": "",
        "analysis_text": "",
        "problem_title": category,
        "probable_cause": rag_context,
        "self_resolution_steps": [],
        "conversation_history": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": questions}
        ],
        "ticket_id": "",
        "resolution_notes": "",
    }
    return state


def generate_final_resolution(state: AgentState, user_answers_text: str) -> AgentState:
    """
    Step 2: User answers -> Write session_finding.txt -> Call Groq for final resolution.
    """
    state["user_answers"] = user_answers_text
    user_message = state["user_message"]
    rag_context = state["sop_context"]
    category = state["category"]

    # 1. Create text document in exact format: User problem: ___ internal doc finding: ___ Diagnostic verification details: ___
    finding_content = (
        f"User problem: {user_message}\n"
        f"internal doc finding: {rag_context}\n"
        f"Diagnostic verification details: {user_answers_text}\n"
    )

    # Save the context file locally in project root
    try:
        with open("session_finding.txt", "w", encoding="utf-8") as f:
            f.write(finding_content)
        print("📁 Successfully created local document 'session_finding.txt'")
    except Exception as e:
        print(f"Error writing session_finding.txt: {e}")

    # 2. Call Groq
    llm = _build_llm(temperature=0.2)
    system_prompt = (
        "You are an empathetic, highly skilled IT support technician helping an employee resolve an issue.\n"
        "Analyze their problem description and diagnostic responses, and provide direct, clean, step-by-step troubleshooting recommendations.\n"
        "RULES FOR CONVERSATION STYLE:\n"
        "1. Write in a clean, natural, professional helpdesk voice. Start directly with an empathetic greeting and transition straight to the steps.\n"
        "2. NEVER use robotic filler phrases like 'Based on the diagnostic details provided', 'I have reviewed the internal document findings', 'According to the context', or similar backend jargon.\n"
        "3. Cross-check all input details: do not recommend generic steps (e.g., VPN or Outlook) if the user's symptoms are completely different. Your suggestions must target their specific reported errors/setup.\n"
        "4. Keep explanations short, clear, and focused on action."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=finding_content),
    ]

    try:
        response = llm.invoke(messages)
        chat_reply = response.content.strip()
    except Exception as e:
        print(f"Groq API call failed: {e}")
        # Local fallback resolution
        chat_reply = (
            f"Unable to connect to Groq cloud. Based on local Microsoft Troubleshooting guidelines, "
            f"please perform standard troubleshoot steps for category '{category}'."
        )

    state["analysis_text"] = chat_reply
    state["self_resolution_steps"] = [chat_reply]
    state["conversation_history"].extend([
        {"role": "user", "content": user_answers_text},
        {"role": "assistant", "content": chat_reply}
    ])
    
    return state


def run_analysis_and_resolution(state: AgentState) -> AgentState:
    """Mock runner function for backwards compatibility."""
    return state


def draft_support_email(user_message: str, chat_reply: str) -> str:
    """
    Call Groq to draft a professional support email summarizing the issue and troubleshooting steps.
    """
    llm = _build_llm(temperature=0.3)
    prompt = f"""You are an IT helpdesk assistant. Based on the user's issue and your troubleshooting response, draft a professional support email.
The email must contain:
1. A polite greeting.
2. A brief summary of the issue.
3. The recommended troubleshooting steps.
4. A closing statement stating that an IT support ticket has been created.

User Issue: {user_message}
Troubleshooting Steps: {chat_reply}

Write ONLY the email content. Do not write any other introductory or concluding conversational text."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Failed to draft email: {e}")
        return f"""Subject: Support Ticket - {user_message[:30]}

Hi,

Thank you for reaching out. We have logged a ticket for your issue: "{user_message}".

Here are the recommended steps to resolve it:
{chat_reply}

If the issue persists, our IT Engineer will contact you shortly.

Best regards,
IT Helpdesk Team"""
