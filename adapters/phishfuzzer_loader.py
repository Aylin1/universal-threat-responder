import hashlib
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False

from src.config_access import cfg_get
from src.conformal import SplitConformalRouter
from src.domain_interface import ThreatSignal
from config import cfg

CLASSES = ["VALID", "SPAM", "PHISHING"]

# Real attachment / URL-path extensions only (the old check was a substring match on the whole
# text, so ".exe" fired on e.g. "mail.executivegroup.com").
_EXT_RE = re.compile(
    r"\.(?:exe|zip|html?|js|scr|vbs|iso|img|rar|7z|bat|cmd|ps1|docm|xlsm)(?=$|[\s?#;,)/])",
    re.IGNORECASE,
)
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
            warnings.warn(f"{file_path} was truncated/malformed; recovered up to the last complete record.")
        else:
            raise ValueError(f"Failed to parse JSON dataset from {file_path}")

    df = pd.DataFrame(data)
    col_map = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ['type', 'label', 'class']:
            col_map[col] = 'Type'
        elif c in ['subject']:
            col_map[col] = 'Subject'
        elif c in ['body', 'text', 'content']:
            col_map[col] = 'Body'
        elif c in ['sender', 'from']:
            col_map[col] = 'Sender'
        elif c in ['url', 'urls', 'link', 'links']:
            col_map[col] = 'Url'
        elif c in ['file', 'attachment', 'attachments']:
            col_map[col] = 'File'
        elif c in ['motivation']:
            col_map[col] = 'Motivation'
        elif c in ['original_id', 'seed_id', 'template_id']:
            col_map[col] = 'Original_ID'
    return df.rename(columns=col_map)


def create_group_signature(df: pd.DataFrame) -> pd.Series:
    """Group key for leakage-free splitting. Prefers a real template/seed id column."""
    if 'Original_ID' in df.columns and df['Original_ID'].notna().any():
        return df['Original_ID'].fillna("unknown_group").astype(str)

    def extract_domain(sender: str) -> str:
        if not isinstance(sender, str) or not sender:
            return "unknown_domain"
        m = re.search(r'@([\w.-]+)', sender)
        return m.group(1).lower() if m else sender.lower()

    sender_domains = df['Sender'].apply(extract_domain)
    motivations = df['Motivation'].fillna('none').astype(str).str.lower()
    subject_stems = (df['Subject'].fillna('').astype(str).str.lower()
                     .str.replace(r'[^\w\s]', '', regex=True).str[:25])
    return sender_domains + "_" + motivations + "_" + subject_stems


def _first_url(url_field: str) -> str:
    toks = str(url_field).split()
    return toks[0] if toks else ""


def extract_structural_features(df: pd.DataFrame) -> np.ndarray:
    features = []
    ip_url_regex = re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}')
    url_finder = re.compile(r'https?://[^\s]+')

    for _, row in df.iterrows():
        subject = str(row.get('Subject', ''))
        body = str(row.get('Body', ''))
        sender = str(row.get('Sender', '')).lower()
        url_str = str(row.get('Url', '')).lower()
        url_str = "" if url_str == "nan" else url_str
        file_str = str(row.get('File', '')).lower()
        file_str = "" if file_str == "nan" else file_str

        full_text = f"{subject} {body} {url_str} {file_str}".lower()
        urls = set(url_finder.findall(full_text))
        url_count = len(urls) if urls else (1 if url_str else 0)
        has_ip_url = 1.0 if ip_url_regex.search(full_text) else 0.0

        ext_haystacks = [file_str, url_str] + list(urls)
        has_suspicious_ext = 1.0 if any(_EXT_RE.search(h) for h in ext_haystacks if h) else 0.0

        sender_domain = sender.split('@')[-1].strip(' <>') if '@' in sender else ""
        is_freemail = 1.0 if sender_domain in FREEMAIL_DOMAINS else 0.0

        url_mismatch = 0.0
        first = _first_url(url_str)
        if sender_domain and first and not is_freemail:
            try:
                host = urlparse(first if first.startswith('http') else f"http://{first}").hostname or ""
                related = (host == sender_domain or host.endswith("." + sender_domain)
                           or sender_domain.endswith("." + host))
                if host and not related:
                    url_mismatch = 1.0
            except Exception:
                pass

        features.append([
            float(url_count), has_ip_url, has_suspicious_ext, is_freemail, url_mismatch,
            float(np.log1p(len(full_text))), float(full_text.count('!')),
        ])
    return np.array(features, dtype=np.float32)


def _row_hash(subject: str, body: str) -> str:
    norm = re.sub(r"\s+", " ", f"{subject}\n{body}".strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def get_class_id(label_str: str) -> int:
    l = str(label_str).lower()
    if 'phish' in l:
        return 2
    if 'spam' in l:
        return 1
    return 0


class PhishFuzzerDatasetAdapter:
    """Loads the dataset and builds three DISJOINT, group-aware partitions:

        proper-train  : fits TF-IDF + calibrated classifier
        calibration   : fits the split-conformal router (never used for training)
        test          : evaluation only

    `train_df` (= proper-train + calibration) is what gets indexed as retrieval history.
    The test partition is never indexed. `split_fingerprint()` lets the eval verify that the
    vector store was built from exactly this split.
    """

    def __init__(
        self,
        file_path: str = None,
        fit_models: bool = True,
        alpha: Optional[float] = None,
        class_conditional: bool = True,
        outer_fold: int = 0,
        n_folds: int = 5,
        seed: Optional[int] = None,
    ):
        target_path = file_path or cfg.dataset.file_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Dataset not found at: {target_path}")
        self.file_path = target_path
        self.seed = cfg.dataset.seed if seed is None else seed
        self.outer_fold = outer_fold
        self.classes = list(CLASSES)
        self.alpha = cfg_get("conformal", "alpha", 0.05) if alpha is None else alpha

        self.df = load_and_clean_json(target_path)
        for col in ['Subject', 'Body', 'Sender', 'Url', 'File', 'Type', 'Motivation']:
            if col not in self.df.columns:
                self.df[col] = ""
            self.df[col] = self.df[col].fillna("").astype(str)
        self.df = self.df[self.df['Type'].str.strip() != ""].reset_index(drop=True)
        self.df['full_text'] = self.df['Subject'] + "\n" + self.df['Body']
        self.df['row_hash'] = [_row_hash(s, b) for s, b in zip(self.df['Subject'], self.df['Body'])]

        self.grouping = ("Original_ID" if 'Original_ID' in self.df.columns and self.df['Original_ID'].notna().any()
                         else "heuristic(sender-domain+motivation+subject-stem)")
        if self.grouping.startswith("heuristic"):
            warnings.warn("No Original_ID/seed column found: leakage protection relies on a heuristic "
                          "group key. Check `leakage_report()` before trusting results.")
        groups = create_group_signature(self.df)
        self.df['group'] = groups.values

        # ---- outer split: train(+calibration) vs test --------------------------------
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        splits = list(sgkf.split(self.df, self.df['Type'], groups=groups))
        train_idx, test_idx = splits[outer_fold % n_folds]
        self.train_df = self.df.iloc[train_idx].reset_index(drop=True)
        self.test_df = self.df.iloc[test_idx].reset_index(drop=True)

        # ---- inner split: proper-train vs calibration (group-aware) -------------------
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed + 1)
        p_idx, c_idx = next(inner.split(self.train_df, self.train_df['Type'], groups=self.train_df['group']))
        self.proper_df = self.train_df.iloc[p_idx].reset_index(drop=True)
        self.calib_df = self.train_df.iloc[c_idx].reset_index(drop=True)

        self.vectorizer = None
        self.ml_classifier = None
        self.router: Optional[SplitConformalRouter] = None
        self.conformal_report: Dict[str, Any] = {}
        if fit_models:
            self._fit_models(class_conditional)

    # ------------------------------------------------------------------ modelling
    def _featurize(self, frame: pd.DataFrame):
        tfidf = self.vectorizer.transform(frame['full_text'])
        struct = extract_structural_features(frame)
        return hstack([tfidf, csr_matrix(struct)]).tocsr()

    def _fit_models(self, class_conditional: bool):
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.dataset.tfidf_max_features, stop_words='english', ngram_range=(1, 2))
        X_tfidf = self.vectorizer.fit_transform(self.proper_df['full_text'])
        X_struct = extract_structural_features(self.proper_df)
        X = hstack([X_tfidf, csr_matrix(X_struct)]).tocsr()
        y = np.array([get_class_id(t) for t in self.proper_df['Type']])

        base = (XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
                              eval_metric='mlogloss', random_state=self.seed)
                if HAS_XGBOOST else
                RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.seed))
        self.ml_classifier = CalibratedClassifierCV(estimator=base, cv=3, method='sigmoid')
        self.ml_classifier.fit(X, y)
        if list(self.ml_classifier.classes_) != [0, 1, 2]:
            raise ValueError(f"Expected all three classes in proper-train, got {self.ml_classifier.classes_}")

        # ---- conformal calibration on the held-out calibration fold -------------------
        Xc = self._featurize(self.calib_df)
        yc = np.array([get_class_id(t) for t in self.calib_df['Type']])
        self.router = SplitConformalRouter(self.classes, alpha=self.alpha, class_conditional=class_conditional)
        self.router.fit(self.ml_classifier.predict_proba(Xc), yc)

        # ---- sanity: empirical coverage on the (unused-for-training) test partition ----
        Xt = self._featurize(self.test_df)
        yt = np.array([get_class_id(t) for t in self.test_df['Type']])
        self.conformal_report = self.router.evaluate(self.ml_classifier.predict_proba(Xt), yt)

    # ------------------------------------------------------------------ integrity
    def split_fingerprint(self) -> str:
        """Hash of the retrieval-history membership; compared against data/history_manifest.json."""
        h = hashlib.sha1()
        for x in sorted(self.train_df['row_hash']):
            h.update(x.encode())
        h.update(f"{self.seed}|{self.outer_fold}".encode())
        return h.hexdigest()

    def assert_no_leakage(self) -> None:
        train_groups, test_groups = set(self.train_df['group']), set(self.test_df['group'])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"{len(overlap)} groups appear in both train and test splits")
        dup = set(self.train_df['row_hash']) & set(self.test_df['row_hash'])
        if dup:
            raise AssertionError(f"{len(dup)} identical emails appear in both train and test splits")

    def leakage_report(self, sim_threshold: float = 0.90) -> Dict[str, Any]:
        """Quantifies how close test emails are to training emails (TF-IDF cosine NN).

        A high near-duplicate rate means 'zero leakage' by group id is not enough and the
        benchmark mostly measures template memorisation.
        """
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50000)
        Xtr = vec.fit_transform(self.train_df['full_text'])
        Xte = vec.transform(self.test_df['full_text'])
        nn = np.asarray((Xte @ Xtr.T).max(axis=1).todense()).ravel()
        return {
            "grouping": self.grouping,
            "n_train": int(len(self.train_df)), "n_test": int(len(self.test_df)),
            "group_overlap": int(len(set(self.train_df['group']) & set(self.test_df['group']))),
            "exact_duplicate_overlap": int(len(set(self.train_df['row_hash']) & set(self.test_df['row_hash']))),
            "test_nn_cosine_mean": float(nn.mean()),
            f"test_nn_cosine_ge_{sim_threshold:.2f}_rate": float((nn >= sim_threshold).mean()),
        }

    # ------------------------------------------------------------------ eval items
    def _map_expected_action(self, label: str) -> str:
        lbl = str(label).lower().strip()
        if 'phish' in lbl:
            return "PHISHING"
        if 'spam' in lbl:
            return "SPAM"
        if 'valid' in lbl or 'legit' in lbl or 'benign' in lbl:
            return "VALID"
        return "QUARANTINE"

    def load_as_test_items(self, sample_fraction: float = None, seed: int = None,
                           high_bound: float = None, low_bound: float = None) -> List[Dict[str, Any]]:
        """high_bound/low_bound are accepted for backward compatibility and ignored:
        routing now uses the conformal prediction set, not a fixed probability threshold."""
        if self.router is None:
            raise RuntimeError("Models not fitted (fit_models=False)")
        frac = sample_fraction if sample_fraction is not None else cfg.dataset.sample_fraction
        s_seed = seed if seed is not None else self.seed

        sampled = self.test_df.sample(frac=frac, random_state=s_seed)
        probabilities = self.ml_classifier.predict_proba(self._featurize(sampled))
        use_fast = bool(cfg_get("conformal", "use_fast_path", True))

        items = []
        for pos, (row_idx, row) in enumerate(sampled.iterrows()):
            probs = probabilities[pos]
            pred_idx = int(np.argmax(probs))
            ml_pred = self.classes[pred_idx]
            max_prob = float(probs[pred_idx])
            decision = self.router.predict_one(probs)
            ml_probs = {c: float(p) for c, p in zip(self.classes, probs)}

            payload = f"Subject: {row['Subject']}\n"
            if row['Sender']:
                payload += f"Sender: {row['Sender']}\n"
            if row['Url']:
                payload += f"Links: {row['Url']}\n"
            if row['File']:
                payload += f"Attachments: {row['File']}\n"
            payload += f"\nBody:\n{row['Body']}"

            eval_mode = "FAST_PATH_BYPASS" if (use_fast and decision.is_singleton) else "RAG_LLM_EVALUATION"
            signal_id = f"PF-{row_idx}-{row['row_hash'][:8]}"  # deterministic and unique within the test split

            def build_signal(ctx, rp=max_prob, rm=eval_mode, text=payload, pc=ml_pred, mp=ml_probs,
                             d=decision, sid=signal_id) -> ThreatSignal:
                merged = {**(ctx or {}), "evaluation_mode": rm, "ml_predicted_class": pc, "ml_probs": mp,
                          "conformal_set": list(d.prediction_set), "conformal_singleton": bool(d.is_singleton)}
                return ThreatSignal(signal_id=sid, domain="spam_phishing", raw_payload=text,
                                    risk_score=rp, confidence=rp, context=merged)

            items.append({"input": payload, "expected_action": self._map_expected_action(row['Type']),
                          "signal_builder": build_signal})
        return items
