"""Index the labelled TRAIN split into the `email_history` collection.

* Uses its own collection; scripts/ingest_threat_intel.py can no longer wipe it.
* Asserts the split is leak-free before indexing and never indexes test emails.
* Writes data/history_manifest.json; run_evals.py refuses to run if the manifest does not match
  the dataset/seed/fold it is about to evaluate (stale or mismatched index).
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from adapters.phishfuzzer_loader import PhishFuzzerDatasetAdapter
from adapters.rag_spam_adapter import HISTORY_COLLECTION, normalize_label

MILVUS_URI = "./data/milvus_threat_intel.db"
MANIFEST_PATH = "./data/history_manifest.json"


def populate_history(file_path: str = None, outer_fold: int = 0) -> None:
    print("Loading leakage-checked dataset split (no model training)...")
    adapter = PhishFuzzerDatasetAdapter(file_path=file_path, fit_models=False, outer_fold=outer_fold)
    adapter.assert_no_leakage()
    report = adapter.leakage_report()
    print("Leakage report:", json.dumps(report, indent=1))

    documents = []
    for _, row in adapter.train_df.iterrows():  # proper-train + calibration; NEVER test
        content = (f"Subject: {row['Subject']}\nSender: {row['Sender']}\nLinks: {row['Url']}\n"
                   f"Attachments: {row['File']}\nBody: {row['Body']}")[:60000]
        documents.append(Document(
            page_content=content,
            metadata={"label": normalize_label(row["Type"]), "type": str(row["Type"]),
                      "subject": str(row["Subject"])[:200], "sender": str(row["Sender"])[:200],
                      "split": "train"}))

    print(f"Embedding and indexing {len(documents)} training emails into '{HISTORY_COLLECTION}'...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                       encode_kwargs={"normalize_embeddings": True})
    os.makedirs(os.path.dirname(MILVUS_URI), exist_ok=True)
    Milvus.from_documents(documents=documents, embedding=embeddings, collection_name=HISTORY_COLLECTION,
                          connection_args={"uri": MILVUS_URI}, drop_old=True)

    Path(MANIFEST_PATH).write_text(json.dumps({
        "collection": HISTORY_COLLECTION,
        "dataset_path": adapter.file_path,
        "seed": adapter.seed,
        "outer_fold": outer_fold,
        "n_docs": len(documents),
        "split_fingerprint": adapter.split_fingerprint(),
        "leakage_report": report,
    }, indent=1), encoding="utf-8")
    print(f"Done. Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=str, default=None)
    ap.add_argument("--fold", type=int, default=0, help="outer CV fold used as the held-out test split")
    a = ap.parse_args()
    populate_history(a.dataset_path, a.fold)
