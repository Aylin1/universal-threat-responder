import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from adapters.phishfuzzer_loader import PhishFuzzerDatasetAdapter

MILVUS_URI = "./data/milvus_threat_intel.db"
COLLECTION_NAME = "phishing_threat_intel"

def populate_milvus_from_phishfuzzer():
    print("Loading leakage-free dataset split...")
    adapter = PhishFuzzerDatasetAdapter()
    
    train_df = adapter.train_df
    
    documents = []
    for idx, row in train_df.iterrows():
        content = (
            f"Subject: {row.get('Subject', '')}\n"
            f"Sender: {row.get('Sender', '')}\n"
            f"Links: {row.get('Url', '')}\n"
            f"Attachments: {row.get('File', '')}\n"
            f"Body: {row.get('Body', '')}"
        )
        
        # Truncate content to fit Milvus default max_length (65535)
        if len(content) > 60000:
            content = content[:60000]
        
        metadata = {
            "type": str(row.get('Type', '')),
            "motivation": str(row.get('Motivation', '')),
            "sender": str(row.get('Sender', '')),
            "url": str(row.get('Url', '')),
            "file": str(row.get('File', ''))
        }
        
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Embedding and indexing {len(documents)} training documents into Milvus...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    os.makedirs(os.path.dirname(MILVUS_URI), exist_ok=True)
    
    vector_store = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=True
    )
    
    print(f"Successfully indexed {len(documents)} training samples into '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    populate_milvus_from_phishfuzzer()