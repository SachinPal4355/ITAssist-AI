"""
Agent 1: Intake Agent
Responsibility: Classify the user's IT issue into a category with confidence score.
Input:  User's raw message (str)
Output: {"category": str, "confidence": float, "summary": str}
"""
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import GROQ_API_KEY, GROQ_MODEL, ISSUE_CATEGORIES


def _build_llm() -> ChatGroq:
    import os
    key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    return ChatGroq(
        api_key=key,
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=512,
    )





# Local keywords dictionary mapping categories to specific search terms
LOCAL_KEYWORDS = {
    "Performance": ["slow", "freeze", "freezing", "lag", "cpu", "ram", "memory", "disk usage", "task manager", "sluggish", "bsod", "blue screen", "hang", "speed"],
    "VPN / Remote Access": ["vpn", "forticlient", "anyconnect", "remote access", "pulse secure", "sstp", "pptp", "error 800", "error 691", "error 812", "error 720", "globalprotect"],
    "Network": ["network", "dns", "dhcp", "ip", "ping", "ethernet", "router", "gateway", "cable", "connected", "internet", "offline", "share", "smb", "unc"],
    "Email": ["outlook", "email", "mail", "pst", "ost", "inbox", "send/receive", "calendar", "sync", "office 365", "m365", "exchange"],
    "Access / Permissions": ["active directory", "lock", "locked", "password", "reset", "login", "domain", "credentials", "gpo", "group policy", "permission", "access denied"],
    "Software": ["install", "uninstall", "msi", "error 1603", "crash", "dll", "vc++", "redistributable", "compatibility", "office", "activation"],
    "Hardware": ["driver", "usb", "monitor", "display", "screen", "keyboard", "mouse", "battery", "charge", "power", "hardware", "port"],
    "Backup / Storage": ["backup", "vss", "shadow copy", "restore point", "disk", "drive", "volume", "chkdsk", "hard drive"],
    "Printer": ["printer", "print", "spooler", "queue", "offline", "paper", "driver isolation", "error 0x00000709", "toner"],
    "Security / BitLocker": ["bitlocker", "tpm", "encrypt", "recovery key", "recovery password", "defender", "antivirus", "virus", "malware", "firewall", "blocked"],
    "Windows Update": ["windows update", "wsus", "kb", "update stuck", "cumulative", "error 0x8007", "servicing stack"],
    "Remote Desktop": ["rdp", "remote desktop", "teams call", "teams meeting", "call drop", "quick assist", "screen sharing", "wake on lan", "wol"],
    "Wireless / WiFi": ["wifi", "wi-fi", "wireless", "802.1x", "wpa2", "wpa3", "radius", "nps", "ssid"],
    "System Recovery": ["boot", "reboot", "restart loop", "winre", "recovery mode", "safe mode", "reset pc", "factory reset", "startup repair", "bootmgr"],
    "OneDrive / Cloud": ["onedrive", "sharepoint", "sync error", "reparse point", "files on demand", "cloud storage"],
    "Power / Sleep": ["sleep", "hibernate", "lid", "battery drain", "power plan", "modern standby", "powercfg"],
}


def classify_issue_locally(user_message: str) -> dict | None:
    """
    Attempt to classify the user's issue locally using keyword matching.
    Returns classification results if confidence is high (>= 80%), else None.
    """
    msg_lower = user_message.lower()
    category_scores = {}
    
    for category, keywords in LOCAL_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # Count word matches specifically
            matches = len(re.findall(r'\b' + re.escape(kw) + r'\b', msg_lower))
            # Also support substring matches for compound terms like 'wi-fi'
            if matches == 0 and kw in msg_lower:
                score += 0.5
            else:
                score += matches
        if score > 0:
            category_scores[category] = score
            
    if not category_scores:
        return None
        
    # Sort categories by match score descending
    sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    best_cat, best_score = sorted_cats[0]
    
    # Calculate confidence:
    # If there is a dominant category match with a direct keyword match, confidence is high
    if len(sorted_cats) == 1:
        confidence = 0.90 if best_score >= 1.0 else 0.80
    else:
        # Multiple matching categories, compare best vs second best
        second_cat, second_score = sorted_cats[1]
        diff = best_score - second_score
        if diff >= 1.0:
            confidence = 0.85
        else:
            confidence = 0.70 # Low confidence due to ambiguity
            
    # Return classification only if confidence is >= 80% (0.80)
    if confidence >= 0.80:
        return {
            "category": best_cat,
            "confidence": confidence,
            "summary": f"[Local Classify] {user_message[:80]}",
            "local": True
        }
        
    return None


def classify_issue(user_message: str) -> dict:
    """
    Classify a user's IT issue.
    First tries local keyword classification. If confidence is >= 80%, uses it.
    Otherwise, falls back to Groq LLM.
    """
    # ── Try Local Classifier First ─────────────────────────────────────────────
    local_result = classify_issue_locally(user_message)
    if local_result:
        print(f"[Intake] Local classification matched: {local_result['category']} (confidence: {local_result['confidence']:.0%})")
        return local_result

    # ── Fallback to Groq LLM ───────────────────────────────────────────────────
    print("[Intake] Low local confidence — calling Groq LLM for classification...")
    llm = _build_llm()
    
    categories_list = "\n".join([f"- {cat}" for cat in ISSUE_CATEGORIES])
    system_prompt = f"""You are an IT helpdesk intake specialist. Your job is to classify 
user IT issues into exactly one category and provide a confidence score.

Available categories:
{categories_list}

Respond ONLY with valid JSON in this exact format:
{{
  "category": "<category from list above>",
  "confidence": <float between 0.0 and 1.0>,
  "summary": "<1-sentence summary of the issue>"
}}

Do not include any text outside the JSON object."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User reported: {user_message}"),
    ]
    
    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        
        # Extract JSON even if wrapped in markdown code blocks
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw)
        
        # Validate category
        if result.get("category") not in ISSUE_CATEGORIES:
            result["category"] = "Other"
        
        # Clamp confidence
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.7))))
        result["local"] = False
        
        return result
        
    except (json.JSONDecodeError, Exception) as e:
        # Fallback classification
        return {
            "category": "Other",
            "confidence": 0.5,
            "summary": user_message[:100],
            "local": False
        }
