"""Ingest threat intelligence.

What changed and why
--------------------
* Live feed URLs are written to a JSON snapshot (data/ioc_feed.json) and matched
  deterministically (src/ioc.py). They are no longer embedded: the old feed documents shared
  boilerplate text that dominated their embeddings, so they matched unrelated emails.
* The feed is only re-downloaded with --refresh-feeds. Evaluations therefore run against a
  fixed snapshot (reproducible) whose fetch time is recorded in the file.
* The hand-written social-engineering patterns go into their OWN collection with explicit
  sop_id/category/label metadata (no keyword guessing).
* This script never touches the `email_history` collection. Previously both scripts shared a
  collection name with drop_old=True, and run_evals re-ran this script first, which wiped the
  indexed training emails before every evaluation.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MILVUS_URI = "./data/milvus_threat_intel.db"
IOC_PATH = "./data/ioc_feed.json"
PATTERN_COLLECTION = "social_engineering_patterns"
OPENPHISH_URL = "https://openphish.com/feed.txt"
URLHAUS_API_URL = "https://urlhaus-api.abuse.ch/v1"


# --------------------------------------------------------------------------- IOC feeds
def fetch_openphish(limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        resp = requests.get(OPENPHISH_URL, timeout=15)
        resp.raise_for_status()
        for i, url in enumerate([l.strip() for l in resp.text.splitlines() if l.strip()][:limit], 1):
            out.append({"indicator_id": f"OPENPHISH-{i}", "url": url, "source": "openphish_live",
                        "threat_type": "phishing"})
        print(f"[+] OpenPhish: {len(out)} URLs")
    except Exception as exc:
        print(f"[!] OpenPhish fetch failed: {exc}")
    return out


def fetch_urlhaus(limit: int) -> List[Dict[str, Any]]:
    """URLhaus now requires a free abuse.ch Auth-Key; the old anonymous csv_recent download returns 410.
    Optional: set URLHAUS_AUTH_KEY to enable. Without it this feed is skipped (OpenPhish still works)."""
    key = os.environ.get("URLHAUS_AUTH_KEY", "").strip()
    if not key:
        print("[=] URLhaus skipped: set URLHAUS_AUTH_KEY (free key from the abuse.ch auth portal) to enable it.")
        return []
    out: List[Dict[str, Any]] = []
    try:
        resp = requests.get(f"{URLHAUS_API_URL}/urls/recent/limit/{max(1, min(limit, 1000))}/",
                            headers={"Auth-Key": key}, timeout=20)
        resp.raise_for_status()
        for u in resp.json().get("urls", []):
            if u.get("url"):
                out.append({"indicator_id": f"URLHAUS-{u.get('id')}", "url": u["url"], "source": "urlhaus_live",
                            "threat_type": u.get("threat", "malware_download"),
                            "date_added": u.get("date_added", "")})
        print(f"[+] URLhaus: {len(out)} URLs")
    except Exception as exc:
        print(f"[!] URLhaus fetch failed: {exc}")
    return out


def sync_ioc_snapshot(refresh: bool, limit: int, path: str = IOC_PATH) -> None:
    p = Path(path)
    if p.exists() and not refresh:
        snap = json.loads(p.read_text(encoding="utf-8"))
        print(f"[=] Using cached IOC snapshot ({len(snap.get('records', []))} records, "
              f"fetched {snap.get('fetched_at', '?')}). Pass --refresh-feeds to re-download.")
        return
    records = fetch_openphish(limit) + fetch_urlhaus(limit)
    if not records and p.exists():
        print("[!] Refresh returned nothing; keeping the existing snapshot.")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(),
                             "records": records}, indent=1), encoding="utf-8")
    print(f"[+] Wrote IOC snapshot with {len(records)} records to {path}")


# --------------------------------------------------------------------------- patterns
PATTERNS = [
    {"id": "SOCENG-JOB-001", "sop_id": "SOP-104", "label": "PHISHING",
     "category": "Recruitment Fraud / Practicum Pivot", "indicator_type": "social_engineering",
     "content": ("Candidate receives a polite corporate job rejection praising their skills, followed closely by an "
                 "unrequested email proposing an exclusive paid or trial 4-week remote Data Analyst Practicum, "
                 "mentorship cohort, or training program promising priority hiring or future placement.")},
    {"id": "SOCENG-JOB-002", "sop_id": "SOP-104", "label": "PHISHING",
     "category": "Task Scam / Fake HR Onboarding", "indicator_type": "social_engineering",
     "content": ("Candidate is offered high daily wages for trivial tasks (app reviews, rating products), then "
                 "asked for platform deposits, to buy home-office gear via fake checks, or to transfer cryptocurrency.")},
    {"id": "SOCENG-BEC-001", "sop_id": "SOP-103", "label": "PHISHING",
     "category": "Business Email Compromise (BEC)", "indicator_type": "social_engineering",
     "content": ("Unsolicited email claiming to be from senior leadership or HR demanding urgent wire transfers, "
                 "gift-card purchases, or confidential employee records sent outside approved channels.")},
    {"id": "SPAM-BULK-001", "sop_id": "SOP-102", "label": "SPAM",
     "category": "Unsolicited Commercial Bulk Email", "indicator_type": "bulk_marketing",
     "content": ("Bulk unsolicited marketing proposing unverified B2B lead databases, SEO ranking services, "
                 "unrequested software licenses, or high-pressure sales calls lacking an unsubscribe option.")},
    {"id": "SPAM-FINANCE-001", "sop_id": "SOP-102", "label": "SPAM",
     "category": "Financial / Crypto Promotion Spam", "indicator_type": "bulk_marketing",
     "content": ("Mass mailing promoting guaranteed crypto trading returns, penny-stock alerts, unauthorised loan "
                 "approvals, or offshore wealth-transfer schemes.")},
]


def sync_patterns() -> None:
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus

    docs = [Document(page_content=f"{p['category']}: {p['content']}",
                     metadata={"indicator_id": p["id"], "sop_id": p["sop_id"], "category": p["category"],
                               "label": p["label"], "indicator_type": p["indicator_type"],
                               "source": "social_engineering_and_spam_registry"})
            for p in PATTERNS]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                       encode_kwargs={"normalize_embeddings": True})
    os.makedirs(os.path.dirname(MILVUS_URI), exist_ok=True)
    Milvus.from_documents(docs, embeddings, collection_name=PATTERN_COLLECTION,
                          connection_args={"uri": MILVUS_URI}, drop_old=True)
    print(f"[+] Indexed {len(docs)} policy patterns into '{PATTERN_COLLECTION}'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-feeds", action="store_true", help="re-download OpenPhish/URLhaus")
    ap.add_argument("--ioc-limit", type=int, default=2000, help="max records per feed")
    ap.add_argument("--skip-patterns", action="store_true")
    args = ap.parse_args()
    os.makedirs("./data", exist_ok=True)
    sync_ioc_snapshot(args.refresh_feeds, args.ioc_limit)
    if not args.skip_patterns:
        sync_patterns()