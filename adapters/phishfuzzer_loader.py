import os
import re
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False

from config import cfg
from src.domain_interface import ThreatSignal

SUSPICIOUS_EXTENSIONS = {
    '.exe', '.zip', '.html', '.htm', '.js', '.scr', '.vbs', 
    '.iso', '.img', '.rar', '.7z', '.bat', '.cmd', '.ps1', '.docm', '.xlsm'
}
FREEMAIL_DOMAINS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'icloud.com'}


def load_and_clean_json(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read().strip()
    
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        last_valid_pos = raw_content.rfind('}')
        if last_valid_pos != -1:
            fixed_content = raw_content[:last_valid_pos + 1] + ']'
            data = json.loads(fixed_content)
        else:
            raise ValueError(f"Failed to parse JSON dataset from {file_path}")

    df = pd.DataFrame(data)

    col_map = {}
    for col in df.columns:
        c_lower = str(col).strip().lower()
        if c_lower in ['type', 'label', 'class']:
            col_map[col] = 'Type'
        elif c_lower in ['subject']:
            col_map[col] = 'Subject'
        elif c_lower in ['body', 'text', 'content']:
            col_map[col] = 'Body'
        elif c_lower in ['sender', 'from']:
            col_map[col] = 'Sender'
        elif c_lower in ['url', 'urls', 'link', 'links']:
            col_map[col] = 'Url'
        elif c_lower in ['file', 'attachment', 'attachments']:
            col_map[col] = 'File'
        elif c_lower in ['motivation']:
            col_map[col] = 'Motivation'
        elif c_lower in ['original_id', 'seed_id', 'template_id']:
            col_map[col] = 'Original_ID'

    return df.rename(columns=col_map)


def create_group_signature(df: pd.DataFrame) -> pd.Series:
    if 'Original_ID' in df.columns and df['Original_ID'].notna().any():
        return df['Original_ID'].fillna("unknown_group").astype(str)

    def extract_domain(sender: str) -> str:
        if not isinstance(sender, str) or not sender:
            return "unknown_domain"
        match = re.search(r'@([\w.-]+)', sender)
        return match.group(1).lower() if match else sender.lower()

    sender_domains = df['Sender'].apply(extract_domain)
    motivations = df['Motivation'].fillna('none').astype(str).str.lower()
    subject_stems = (
        df['Subject'].fillna('')
        .astype(str)
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str[:25]
    )

    return sender_domains + "_" + motivations + "_" + subject_stems


def extract_structural_features(df: pd.DataFrame) -> np.ndarray:
    features = []
    ip_url_regex = re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}')
    url_finder = re.compile(r'https?://[^\s]+')

    for _, row in df.iterrows():
        subject = str(row.get('Subject', ''))
        body = str(row.get('Body', ''))
        sender = str(row.get('Sender', '')).lower()
        url_str = str(row.get('Url', '')).lower()
        file_str = str(row.get('File', '')).lower()
        
        full_text = f"{subject} {body} {url_str} {file_str}".lower()

        urls_in_text = url_finder.findall(full_text)
        url_count = len(urls_in_text) + (1 if url_str and url_str != "nan" else 0)
        has_ip_url = 1.0 if bool(ip_url_regex.search(full_text)) else 0.0

        has_suspicious_ext = 0.0
        for ext in SUSPICIOUS_EXTENSIONS:
            if ext in full_text or ext in file_str:
                has_suspicious_ext = 1.0
                break

        sender_domain = sender.split('@')[-1].strip() if '@' in sender else ""
        is_freemail = 1.0 if sender_domain in FREEMAIL_DOMAINS else 0.0

        url_mismatch = 0.0
        if sender_domain and url_str:
            try:
                parsed_host = urlparse(url_str if url_str.startswith('http') else f"http://{url_str}").netloc
                if parsed_host and sender_domain not in parsed_host and not is_freemail:
                    url_mismatch = 1.0
            except Exception:
                pass

        text_len = float(len(full_text))
        exclamation_count = float(full_text.count('!'))

        features.append([
            float(url_count),
            has_ip_url,
            has_suspicious_ext,
            is_freemail,
            url_mismatch,
            np.log1p(text_len),
            exclamation_count
        ])

    return np.array(features, dtype=np.float32)


class PhishFuzzerDatasetAdapter:
    def __init__(self, file_path: str = None):
        target_path = file_path or cfg.dataset.file_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Dataset not found at: {target_path}")

        self.df = load_and_clean_json(target_path)

        for col in ['Subject', 'Body', 'Sender', 'Url', 'File', 'Type', 'Motivation']:
            if col not in self.df.columns:
                self.df[col] = ""
            self.df[col] = self.df[col].fillna("").astype(str)

        self.df = self.df[self.df['Type'].str.strip() != ""].reset_index(drop=True)
        self.df['full_text'] = self.df['Subject'] + "\n" + self.df['Body']

        groups = create_group_signature(self.df)
        sgkf = StratifiedGroupKFold(n_splits=5)
        
        train_idx, test_idx = next(sgkf.split(self.df, self.df['Type'], groups=groups))

        self.train_df = self.df.iloc[train_idx].reset_index(drop=True)
        self.test_df = self.df.iloc[test_idx].reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            max_features=cfg.dataset.tfidf_max_features, stop_words='english', ngram_range=(1, 2)
        )
        X_train_tfidf = self.vectorizer.fit_transform(self.train_df['full_text'])
        X_train_struct = extract_structural_features(self.train_df)
        X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_struct)]).tocsr()

        def get_class_id(label_str):
            l = str(label_str).lower()
            if 'phish' in l: return 2
            if 'spam' in l: return 1
            return 0 # valid

        y_train = np.array([get_class_id(t) for t in self.train_df['Type']])

        base_model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
            eval_metric='mlogloss', random_state=cfg.dataset.seed
        ) if HAS_XGBOOST else RandomForestClassifier(n_estimators=100, max_depth=10, random_state=cfg.dataset.seed)

        self.ml_classifier = CalibratedClassifierCV(estimator=base_model, cv=3, method='sigmoid')
        self.ml_classifier.fit(X_train_combined, y_train)

    def _map_expected_action(self, label: str) -> str:
        lbl = str(label).lower().strip()
        if 'phish' in lbl: return "PHISHING"
        elif 'spam' in lbl: return "SPAM"
        elif 'valid' in lbl or 'legit' in lbl or 'benign' in lbl: return "VALID"
        return "QUARANTINE"

    def load_as_test_items(self, sample_fraction: float = None, seed: int = None, high_bound: float = None, low_bound: float = None) -> List[Dict[str, Any]]:
        frac = sample_fraction if sample_fraction is not None else cfg.dataset.sample_fraction
        s_seed = seed if seed is not None else cfg.dataset.seed
        hb = high_bound if high_bound is not None else cfg.conformal.high_bound

        sampled_df = self.test_df.sample(frac=frac, random_state=s_seed)
        test_items = []

        X_test_tfidf = self.vectorizer.transform(sampled_df['full_text'])
        X_test_struct = extract_structural_features(sampled_df)
        X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_struct)]).tocsr()

        classes = ["VALID", "SPAM", "PHISHING"]
        probabilities = self.ml_classifier.predict_proba(X_test_combined)

        for idx, (_, row) in enumerate(sampled_df.iterrows()):
            expected_action = self._map_expected_action(row['Type'])
            
            probs = probabilities[idx]
            pred_idx = np.argmax(probs)
            ml_pred_class = classes[pred_idx]
            max_prob = float(probs[pred_idx])

            raw_payload = f"Subject: {row['Subject']}\n"
            if row['Sender']:
                raw_payload += f"Sender: {row['Sender']}\n"
            if row['Url']:
                raw_payload += f"Links: {row['Url']}\n"
            if row['File']:
                raw_payload += f"Attachments: {row['File']}\n"
            raw_payload += f"\nBody:\n{row['Body']}"

            # Conformal bounds check for fast-path routing decision
            if cfg.conformal.use_fast_path and max_prob >= hb:
                eval_mode = "FAST_PATH_BYPASS"
            else:
                eval_mode = "RAG_LLM_EVALUATION"

            def build_signal(ctx: Dict[str, Any], rp=max_prob, rm=eval_mode, rp_text=raw_payload, pc=ml_pred_class) -> ThreatSignal:
                merged_context = {**(ctx or {}), "evaluation_mode": rm, "ml_predicted_class": pc}

                return ThreatSignal(
                    signal_id=f"PF-{random.randint(100000, 999999)}",
                    domain="spam_phishing",
                    raw_payload=rp_text,
                    risk_score=rp,
                    confidence=rp,
                    context=merged_context
                )

            test_items.append({
                "input": raw_payload,
                "expected_action": expected_action,
                "signal_builder": build_signal
            })

        return test_items