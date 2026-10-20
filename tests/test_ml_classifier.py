import numpy as np
import pandas as pd

from adapters.phishfuzzer_loader import CLASSES, EmailThreatClassifier, parse_raw_email


def test_parse_raw_email_reads_evaluation_payload_layout():
    text = (
        "Subject: Verify now\nSender: it@corp.invalid\nLinks: http://evil.invalid\n"
        "Attachments: form.exe\n\nBody:\nPlease log in."
    )

    fields = parse_raw_email(text)

    assert fields == {
        "Subject": "Verify now",
        "Sender": "it@corp.invalid",
        "Url": "http://evil.invalid",
        "File": "form.exe",
        "Body": "Please log in.",
    }


def test_parse_raw_email_treats_plain_text_as_body():
    fields = parse_raw_email("Hello, are we still meeting at 3?")

    assert fields["Body"] == "Hello, are we still meeting at 3?"
    assert fields["Subject"] == "" and fields["Sender"] == ""


def make_training_df():
    rows = [
        ("Win a free prize now", "Click http://spam.invalid to claim your prize!!!", "SPAM"),
        ("Cheap pills discount", "Buy now, limited offer, unsubscribe here", "SPAM"),
        ("Verify your account", "Your password expires, login at http://phish.invalid/login", "PHISHING"),
        ("Urgent: mailbox full", "Confirm your credentials at http://phish.invalid/verify", "PHISHING"),
        ("Meeting notes", "Attached are the notes from our planning meeting.", "VALID"),
        ("Lunch tomorrow", "Are you free for lunch tomorrow around noon?", "VALID"),
    ] * 6
    df = pd.DataFrame(rows, columns=["Subject", "Body", "Type"])
    df["Sender"] = ""
    df["Url"] = ""
    df["File"] = ""
    df["full_text"] = df["Subject"] + "\n" + df["Body"]
    return df


def test_classifier_round_trips_through_save_and_load(tmp_path):
    model = EmailThreatClassifier.fit(make_training_df())
    path = str(tmp_path / "models" / "clf.joblib")

    model.save(path)
    loaded = EmailThreatClassifier.load(path)

    text = "Subject: Verify your account\n\nBody:\nLogin at http://phish.invalid/login now"
    predicted_class, probability = loaded.predict_text(text)
    assert predicted_class in CLASSES
    assert 0.0 < probability <= 1.0
    assert (predicted_class, probability) == model.predict_text(text)


def test_predict_proba_rows_sum_to_one():
    df = make_training_df()
    model = EmailThreatClassifier.fit(df)

    probs = model.predict_proba_df(df)

    assert probs.shape == (len(df), len(CLASSES))
    assert np.allclose(probs.sum(axis=1), 1.0)
