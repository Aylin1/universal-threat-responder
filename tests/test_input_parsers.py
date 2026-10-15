import pytest

from src.input_parsers import parse_csv_samples, parse_json_samples


def test_json_parser_accepts_bom_envelopes_strings_and_metadata():
    samples = parse_json_samples(
        b'\xef\xbb\xbf{"samples":["  plain text  ",'
        b'{"email":" message ","source":"test"},{"payload":"payload"}]}'
    )

    assert [sample["text"] for sample in samples] == [
        "plain text",
        "message",
        "payload",
    ]
    assert samples[1]["metadata"]["source"] == "test"
    assert samples[2]["id"] == "sample-3"


def test_json_parser_skips_empty_samples_and_rejects_invalid_items():
    assert parse_json_samples(b'{"text":"  "}') == []

    with pytest.raises(ValueError, match="sample 2 must be a string or an object"):
        parse_json_samples(b'["valid", null]')


def test_csv_parser_handles_bom_fallback_and_metadata():
    samples = parse_csv_samples(
        b"\xef\xbb\xbfsubject,category\r\n"
        b'"  Suspicious message  ",phishing\r\n'
    )

    assert len(samples) == 1
    assert samples[0]["text"] == "Suspicious message"
    assert samples[0]["metadata"]["category"] == "phishing"


def test_csv_parser_returns_no_samples_for_header_only_input():
    assert parse_csv_samples(b"text,email\n") == []
