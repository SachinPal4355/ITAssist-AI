
from rag.retriever import search, is_index_ready


# Category-to-query mapping for better RAG recall — maps to 16 SOP documents
CATEGORY_QUERIES = {
    # Existing categories
    "Performance": "slow performance high CPU disk usage memory Windows blue screen BSOD shutdown",
    "VPN / Remote Access": "VPN remote access connection failed error Windows RDP remote desktop",
    "Network": "network connectivity DNS DHCP TCP/IP SMB file share wireless adapter",
    "Email": "Outlook email not working sync issues calendar PST Office 365 Exchange",
    "Access / Permissions": "access denied permissions Active Directory login account lockout GPO domain",
    "Software": "software installation application crash update error MSI Microsoft 365 Office",
    "Hardware": "hardware device driver not recognized USB monitor keyboard mouse battery",
    "Backup / Storage": "backup storage VSS shadow copy disk drive file access OneDrive SharePoint",
    "Printer": "printer offline print spooler not responding driver network printer",
    "Security / BitLocker": "BitLocker recovery password TPM encryption security Windows Defender malware",
    # New expanded categories
    "Windows Update": "Windows Update stuck downloading error 0x80070002 WSUS patch management",
    "Remote Desktop": "Remote Desktop RDP connection refused Teams video call quality screen sharing",
    "Wireless / WiFi": "WiFi wireless cannot connect 802.1X enterprise RADIUS WPA2 adapter missing",
    "System Recovery": "Windows not booting system restore safe mode WinRE factory reset BOOTMGR",
    "OneDrive / Cloud": "OneDrive sync error SharePoint access files on demand cloud storage",
    "Power / Sleep": "sleep hibernate wake black screen power plan battery laptop lid close",
    "Other": "Windows troubleshooting general IT support help desk",
}


def search_knowledge(category: str, user_message: str) -> dict:
    """
    Search the knowledge base for relevant SOP content.

    Args:
        category: Classified issue category
        user_message: Original user message

    Returns:
        dict: {sop_context: str, articles_used: list[str], index_ready: bool}
    """
    if not is_index_ready():
        return {
            "sop_context": (
                "Knowledge base is not yet built. "
                "Please run 'python rag/ingest.py' to build the index."
            ),
            "articles_used": [],
            "index_ready": False,
        }

    # Build a rich query combining category hint + user message
    category_hint = CATEGORY_QUERIES.get(category, "Windows troubleshooting")
    combined_query = f"{category_hint}. User issue: {user_message}"

    results = search(combined_query, k=3)

    if not results:
        return {
            "sop_context": "No specific documentation found for this issue.",
            "articles_used": [],
            "index_ready": True,
        }

    # Format context for agent consumption
    sections = []
    articles_used = []
    for i, r in enumerate(results, 1):
        source = r["source"]
        # Human-readable article name
        article_name = (
            source.replace("_sop.txt", "")
            .replace("_", " ")
            .replace(".txt", "")
            .title()
        )
        if article_name not in articles_used:
            articles_used.append(article_name)

        sections.append(f"[Documentation {i} – {article_name}]\n{r['content']}")

    sop_context = "\n\n---\n\n".join(sections)

    return {
        "sop_context": sop_context,
        "articles_used": articles_used,
        "index_ready": True,
    }
