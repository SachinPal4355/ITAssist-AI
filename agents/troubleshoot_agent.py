
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import GROQ_API_KEY, GROQ_MODEL


def _build_llm() -> ChatGroq:
    import os
    key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    return ChatGroq(
        api_key=key,
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
    )


# ── Phase A: Generate Questions ───────────────────────────────────────────────

QUESTIONS_SYSTEM_PROMPT = """You are a senior IT support engineer. Based on the 
user's issue category, their description, and the relevant documentation provided,
generate exactly 2-3 targeted diagnostic questions to identify the root cause.

Rules:
- Questions must be concise and specific (not open-ended)
- Each question should help narrow down the root cause
- Use yes/no or short-answer format where possible
- Reference specific symptoms from the documentation context

Respond ONLY with valid JSON:
{
  "questions": [
    "Question 1 here?",
    "Question 2 here?",
    "Question 3 here?"
  ]
}"""


def get_local_category_questions(category: str) -> list[str]:
    """Retrieve pre-defined targeted questions for a specific category locally."""
    fallbacks = {
        "Performance": [
            "Is the device freezing or just running slowly?",
            "What is the disk usage shown in Task Manager? (e.g. 100% disk usage)",
            "When did this problem start — after an update or recent change?"
        ],
        "VPN / Remote Access": [
            "What error code or message does the VPN client show? (e.g. Error 800, 691, 812)",
            "Are you able to connect to the internet without VPN?",
            "Has the VPN worked successfully on this device before?"
        ],
        "Network": [
            "Are other devices on the same network also affected?",
            "Can you ping 8.8.8.8? (Open CMD → type: ping 8.8.8.8)",
            "Is this a WiFi or wired (Ethernet) connection?"
        ],
        "Email": [
            "Does Outlook open in Safe Mode? (Press Win+R → type: outlook.exe /safe)",
            "Are emails stuck in the Outbox or is the calendar not syncing?",
            "Have you tried rebuilding the Outlook profile or repairing Office?"
        ],
        "Access / Permissions": [
            "Is the user account locked in Active Directory or is the password expired?",
            "What error message do you get? (e.g. 'Access Denied', 'Trust relationship failed')",
            "Are you trying to access a network shared folder or a specific domain controller?"
        ],
        "Software": [
            "What error code does the installer show? (e.g. MSI error 1603, 1618)",
            "Does the application crash immediately on launch or show missing DLL errors?",
            "Have you tried running it in compatibility mode or as Administrator?"
        ],
        "Hardware": [
            "Is there a yellow exclamation mark or 'Unknown device' in Device Manager?",
            "Does the USB device, display monitor, or keyboard/mouse work on another port/PC?",
            "If it's a laptop battery/power issue, is the charger recognized and charging?"
        ],
        "Backup / Storage": [
            "What error code does the backup show? (e.g. VSS writer error)",
            "Is the backup destination drive (local, network share, or USB) accessible?",
            "Are you unable to restore previous versions or create shadow copies?"
        ],
        "Printer": [
            "Is the printer showing as 'Offline' in Windows or are jobs stuck in the queue?",
            "Is this a network printer or connected via USB cable?",
            "Does the print spooler service keep crashing?"
        ],
        "Security / BitLocker": [
            "Is Windows prompting for a 48-digit BitLocker recovery password at boot?",
            "Did this prompt start after a recent Windows Update or UEFI/TPM firmware update?",
            "Is Windows Defender reporting real-time threat detections or blocks?"
        ],
        "Windows Update": [
            "Is the Windows Update stuck downloading/installing at a specific percentage?",
            "What error code does it show? (e.g. 0x80070002, 0x800F0922)",
            "Is this device configured to retrieve updates from a WSUS server or directly?"
        ],
        "Remote Desktop": [
            "What error message does RDP show? (e.g. Error 0x204, NLA error, credentials failed)",
            "Is RDP enabled on the target machine and allowed through the Windows Firewall?",
            "Are you experiencing Teams call quality drops or microphone/camera failures?"
        ],
        "Wireless / WiFi": [
            "Is this a corporate WiFi connection using 802.1X enterprise authentication?",
            "Is the WiFi adapter missing from Device Manager or does the signal keep dropping?",
            "Have you tried deleting the wireless profile and reconnecting?"
        ],
        "System Recovery": [
            "Is the PC stuck in an Automatic Repair loop or booting to a black screen?",
            "Can you access the Windows Recovery Environment (WinRE) or Safe Mode?",
            "Are you getting a boot error like 'BOOTMGR is missing' or 'No boot device found'?"
        ],
        "OneDrive / Cloud": [
            "Is OneDrive showing sync errors or stuck in a 'Syncing' state?",
            "Are you getting error 0x8007016A when trying to open Files On-Demand?",
            "Are you having permissions issues with a SharePoint document library?"
        ],
        "Power / Sleep": [
            "Is the PC failing to sleep, or waking up immediately from sleep?",
            "What device or request is keeping the PC awake? (run: powercfg /requests)",
            "Is this a laptop using Modern Standby (S0 low power idle) draining battery?"
        ],
    }
    return fallbacks.get(
        category,
        [
            "When did this issue first occur?",
            "Has anything changed recently (update, new software, new hardware)?",
            "Is this affecting just your device or multiple users?"
        ]
    )


def generate_questions(
    category: str,
    user_message: str,
    sop_context: str,
    intake_confidence: float = 0.5
) -> list[str]:
    """
    Generate diagnostic questions for the user.
    Uses pre-defined specific local questions if local intake confidence is high (>= 80%).
    Otherwise, calls Groq LLM to generate custom dynamic questions.
    """
    # ── High Local Confidence Check ───────────────────────────────────────────
    if intake_confidence >= 0.80 and category != "Other":
        print(f"[Troubleshoot] Intake matched locally with {intake_confidence:.0%} confidence. Loading pre-defined questions for category '{category}'...")
        return get_local_category_questions(category)

    # ── Dynamic LLM Question Generation ────────────────────────────────────────
    print("[Troubleshoot] Low local confidence — calling Groq LLM to generate diagnostic questions...")
    llm = _build_llm()

    prompt = f"""Category: {category}
User Issue: {user_message}

Relevant Documentation:
{sop_context[:2000]}

Generate 2-3 targeted diagnostic questions."""

    messages = [
        SystemMessage(content=QUESTIONS_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("questions", [])[:3]
    except Exception:
        pass

    # Fallback to local questions if LLM fails
    return get_local_category_questions(category)


# ── Phase B: Analyze Root Cause ───────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """You are a senior IT support engineer. Based on the 
user's issue, the documentation, and the user's answers to your diagnostic questions,
provide a root cause analysis.

Severity levels: Low, Medium, High, Critical
Confidence: 0.0 to 1.0

Respond ONLY with valid JSON:
{
  "problem": "Short problem title (5-10 words)",
  "probable_cause": "Specific root cause explanation (1-2 sentences)",
  "severity": "High",
  "confidence": 0.87,
  "analysis": "Detailed technical analysis (3-5 sentences) referencing the documentation",
  "self_resolvable": true,
  "self_resolution_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ]
}"""


def analyze_root_cause(
    category: str,
    user_message: str,
    sop_context: str,
    questions: list[str],
    answers: list[str],
) -> dict:
    """
    Analyze root cause after collecting user answers.

    Returns:
        dict with problem, probable_cause, severity, confidence, analysis, self_resolution_steps
    """
    llm = _build_llm()

    qa_pairs = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
    )

    prompt = f"""Category: {category}
Original Issue: {user_message}

Diagnostic Q&A:
{qa_pairs}

Relevant Documentation:
{sop_context[:2000]}

Analyze the root cause and determine if the user can self-resolve."""

    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.7))))
            if "self_resolution_steps" not in result:
                result["self_resolution_steps"] = []
            return result
    except Exception as e:
        pass

    # Dynamic local RAG fallback: parse matching SOP document directly
    from rag.retriever import parse_sop_file_locally
    local_diag = parse_sop_file_locally(category, user_message)
    if local_diag:
        return {
            "problem": local_diag["problem"],
            "probable_cause": local_diag["probable_cause"],
            "severity": local_diag["severity"],
            "confidence": local_diag["confidence"],
            "analysis": local_diag["analysis"],
            "self_resolvable": local_diag["self_resolvable"],
            "self_resolution_steps": local_diag["self_resolution_steps"],
        }

    # Absolute minimal fallback if SOP parsing fails
    return {
        "problem": f"{category} Issue",
        "probable_cause": "Root cause requires further investigation by an IT engineer.",
        "severity": "Medium",
        "confidence": 0.6,
        "analysis": "Based on the symptoms described, this requires escalation.",
        "self_resolvable": False,
        "self_resolution_steps": [],
    }
