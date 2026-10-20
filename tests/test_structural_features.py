import pandas as pd

from adapters.phishfuzzer_loader import extract_structural_features

URL_COUNT, _, SUSPICIOUS_EXT = 0, 1, 2


def features(**row):
    base = {"Subject": "", "Body": "", "Sender": "", "Url": "", "File": ""}
    return extract_structural_features(pd.DataFrame([{**base, **row}]))[0]


def test_json_and_html_links_are_not_flagged_as_suspicious_extensions():
    assert features(Body="See data.json and https://example.com/page.html?x=1")[SUSPICIOUS_EXT] == 1.0
    assert features(Body="See data.json and notes.jsonl")[SUSPICIOUS_EXT] == 0.0


def test_real_suspicious_extension_is_flagged():
    assert features(File="invoice.exe")[SUSPICIOUS_EXT] == 1.0
    assert features(Body="open archive.zip now")[SUSPICIOUS_EXT] == 1.0


def test_url_is_not_double_counted():
    assert features(Body="http://a.invalid", Url="http://a.invalid")[URL_COUNT] == 1.0
    assert features(Url="a.invalid/login")[URL_COUNT] == 1.0
    assert features(Body="no links")[URL_COUNT] == 0.0
