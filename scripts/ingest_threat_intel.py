import os
import csv
import requests
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# Configuration Constants
MILVUS_URI = "./data/milvus_threat_intel.db"
COLLECTION_NAME = "phishing_threat_intel"

# Phishing & Live Payload Endpoints
OPENPHISH_URL = "https://openphish.com/feed.txt"
URLHAUS_CSV_URL = "https://urlhaus.abuse.ch/downloads/csv_recent/"


def fetch_phishing_and_live_iocs() -> List[Document]:
    """Fetch real-time phishing and payload URLs from OpenPhish and URLhaus."""
    documents = []
    
    # 1. OpenPhish Live Links
    try:
        print("[*] Fetching live phishing indicators from OpenPhish...")
        resp = requests.get(OPENPHISH_URL, timeout=10)
        if resp.status_code == 200:
            lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
            for i, phish_url in enumerate(lines[:30]):
                documents.append(Document(
                    page_content=(
                        f"Threat Vector: Active Phishing URL / Link\n"
                        f"Target URL: {phish_url}\n"
                        f"Source: OpenPhish Live Feed\n"
                        f"Risk Assessment: High probability of credential harvest or account compromise."
                    ),
                    metadata={
                        "indicator_id": f"PHISH-OPENPHISH-{i+1}",
                        "risk_score": 0.95,
                        "source": "openphish_live"
                    }
                ))
            print(f"[+] Ingested {len(documents)} live OpenPhish records.")
    except Exception as e:
        print(f"[!] OpenPhish Fetch Warning: {e}")

    # 2. Abuse.ch URLhaus Payloads
    try:
        print("[*] Fetching live malicious payloads from Abuse.ch URLhaus...")
        resp = requests.get(URLHAUS_CSV_URL, timeout=15)
        if resp.status_code == 200:
            decoded = resp.content.decode('utf-8', errors='ignore')
            lines = [l for l in decoded.splitlines() if not l.startswith('#')]
            reader = csv.reader(lines)
            count = 0
            for row in reader:
                if count >= 30:
                    break
                if len(row) > 4:
                    date_added, target_url, threat_type = row[1], row[2], row[3]
                    documents.append(Document(
                        page_content=(
                            f"Threat Vector: Malicious Payload URL\n"
                            f"Type: {threat_type}\n"
                            f"Target URL: {target_url}\n"
                            f"Reported Date: {date_added} via Abuse.ch URLhaus."
                        ),
                        metadata={
                            "indicator_id": f"PHISH-URLHAUS-{count+1}",
                            "risk_score": 0.90,
                            "source": "urlhaus_live"
                        }
                    ))
                    count += 1
            print(f"[+] Ingested {count} URLhaus malicious payload records.")
    except Exception as e:
        print(f"[!] URLhaus Fetch Warning: {e}")

    return documents


def build_social_engineering_and_spam_docs() -> List[Document]:
    """Generates rich behavioral templates for recruitment scams, BEC, and general email spam."""
    behavioral_patterns = [
        # Recruitment & Practicum Fraud
        {
            "id": "SOCENG-JOB-001",
            "category": "Recruitment Fraud / Practicum Pivot",
            "content": (
                "Employment Scam Pattern: Candidate receives an initial polite corporate job rejection letter "
                "praising their analytical or technical skills, followed closely by an unrequested follow-up email "
                "proposing an exclusive paid or trial 4-week remote Data Analyst Practicum, mentorship cohort, or training program "
                "promising priority hiring or future placement at the firm."
            ),
            "score": 0.96
        },
        {
            "id": "SOCENG-JOB-002",
            "category": "Task Scam / Fake HR Onboarding",
            "content": (
                "Fake HR & Task Scam Pattern: Candidate is offered high daily wages for trivial tasks (app reviews, rating products). "
                "Requires preliminary platform deposits, purchasing home office gear via fake checks, or transferring cryptocurrency."
            ),
            "score": 0.92
        },

        # Executive & BEC Impersonation
        {
            "id": "SOCENG-BEC-001",
            "category": "Business Email Compromise (BEC)",
            "content": (
                "Executive Impersonation Pattern: Unsolicited internal email from senior leadership or HR strategy "
                "demanding urgent wire transfers, purchasing gift cards, or sending confidential employee records outside approved channels."
            ),
            "score": 0.95
        },

        # General High-Volume Email Spam
        {
            "id": "SPAM-BULK-001",
            "category": "Unsolicited Commercial Bulk Email",
            "content": (
                "General Commercial Spam Pattern: Bulk unsolicited marketing proposing unverified B2B lead databases, "
                "SEO ranking services, unrequested software licenses, or high-pressure sales calls lacking standard unsubscribe headers."
            ),
            "score": 0.65
        },
        {
            "id": "SPAM-FINANCE-001",
            "category": "Financial / Crypto Promotion Spam",
            "content": (
                "Financial Scam Pattern: Mass mailing promoting guaranteed crypto trading returns, penny stock alerts, "
                "unauthorized loan approvals, or offshore wealth transfer schemes."
            ),
            "score": 0.80
        }
    ]

    documents = []
    for item in behavioral_patterns:
        documents.append(Document(
            page_content=(
                f"Threat Domain: {item['category']}\n"
                f"Pattern Reference: {item['id']}\n"
                f"Behavioral Narrative: {item['content']}\n"
                f"Risk Classification: High psychological / manipulative threat vector."
            ),
            metadata={
                "indicator_id": item["id"],
                "risk_score": item["score"],
                "source": "social_engineering_and_spam_registry"
            }
        ))
    return documents


def build_unified_intel_documents() -> List[Document]:
    """Builds unified threat intel corpus across live phishing links, social engineering, and spam."""
    documents = []

    # 1. Fetch Live Phishing & Payload Indicators
    documents.extend(fetch_phishing_and_live_iocs())

    # 2. Append Social Engineering & General Spam Behavioral Patterns
    documents.extend(build_social_engineering_and_spam_docs())

    return documents


def sync_to_milvus():
    docs = build_unified_intel_documents()
    if not docs:
        print("[!] No threat documents generated. Aborting sync.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    Milvus.from_documents(
        docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=True
    )
    print(f"[+] Successfully synchronized {len(docs)} email threat records into Milvus collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    sync_to_milvus()